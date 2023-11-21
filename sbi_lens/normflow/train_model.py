from functools import partial

import jax
import jax.numpy as jnp
import optax
import tensorflow_probability as tfp

tfp = tfp.experimental.substrates.jax
tfb = tfp.bijectors
tfd = tfp.distributions


class TrainModel:
    def __init__(
        self,
        compressor,
        nf,
        optimizer,
        loss_name,
        dim=None,
        info_compressor=None,
    ):
        self.compressor = compressor
        self.nf = nf
        self.optimizer = optimizer
        self.dim = dim # summary statistic dimension

        if loss_name == "train_compressor_mse":
            self.loss = self.loss_mse
        elif loss_name == "train_compressor_vmim":
            self.loss = self.loss_vmim
        elif loss_name == "train_compressor_gnll":
            self.loss = self.loss_gnll
            if self.dim is None:
                raise ValueError("dim should be specified when using gnll compressor")
        elif loss_name == "loss_for_sbi":
            if info_compressor is None:
                raise ValueError("sbi loss needs compressor informations")
            else:
                self.info_compressor = info_compressor
                self.loss = self.loss_nll

    def loss_mse(self, params, theta, x, state_resnet):
        """Compute the Mean Squared Error loss
        """
        y, opt_state_resnet = self.compressor.apply(params, state_resnet, None, x)

        loss = jnp.mean(jnp.sum((y - theta) ** 2, axis=1))

        return loss, opt_state_resnet

    def loss_mae(self, params, theta, x, state_resnet):
        """Compute the Mean Absolute Error loss
        """
        y, opt_state_resnet = self.compressor.apply(params, state_resnet, None, x)

        loss = jnp.mean(jnp.sum(jnp.absolute(y - theta), axis=1))

        return loss, opt_state_resnet

    def loss_vmim(self, params, theta, x, state_resnet):
        """Compute the Variational Mutual Information Maximization loss
        """
        y, opt_state_resnet = self.compressor.apply(params, state_resnet, None, x)
        log_prob = self.nf.apply(params, theta, y)

        return -jnp.mean(log_prob), opt_state_resnet

    def loss_gnll(self, params, theta, x, state_resnet):
        """Compute the Gaussian Negative Log Likelihood loss
        """
        y, opt_state_resnet = self.compressor.apply(params, state_resnet, None, x)
        y_mean = y[..., : self.dim]
        y_var = y[..., self.dim :]
        y_var = tfb.FillScaleTriL(diag_bijector=tfb.Softplus(low=1e-3)).forward(y_var)

        @jax.jit
        @jax.vmap
        def _get_log_prob(y_mean, y_var, theta):
            likelihood = tfd.MultivariateNormalTriL(y_mean, y_var)
            return likelihood.log_prob(theta)

        loss = -jnp.mean(_get_log_prob(y_mean, y_var, theta))

        return loss, opt_state_resnet

    def loss_nll(self, params, theta, x, _):
        """Compute the Negative Log Likelihood loss.
        This loss is for inference so it requires to have a trained compressor.
        """
        y, _ = self.compressor.apply(
            self.info_compressor[0], self.info_compressor[1], None, x
        )
        log_prob = self.nf.apply(params, theta, y)

        return -jnp.mean(log_prob), _

    @partial(jax.jit, static_argnums=(0,))
    def update(self, model_params, opt_state, theta, x, state_resnet=None):
        (loss, opt_state_resnet), grads = jax.value_and_grad(self.loss, has_aux=True)(
            model_params, theta, x, state_resnet
        )

        updates, new_opt_state = self.optimizer.update(grads, opt_state)

        new_params = optax.apply_updates(model_params, updates)

        return loss, new_params, new_opt_state, opt_state_resnet
