import jax
import jax.numpy as jnp
import optax
from functools import partial


class train_model():

  def loss_mse(self, params, theta, x, state_resnet):

    y, opt_state_resnet = self.compressor.apply(
      params,
      state_resnet,
      None,
      x.reshape([-1, self.nb_pixels, self.nb_pixels, self.nb_bins])
    )

    loss = jnp.mean(jnp.sum((y - theta)**2, axis=1))

    return loss, opt_state_resnet

  def loss_vmim(self, params, theta, x, state_resnet):

    y, opt_state_resnet = self.compressor.apply(
      params,
      state_resnet,
      None,
      x.reshape([-1, self.nb_pixels, self.nb_pixels, self.nb_bins])
    )
    log_prob = jax.vmap(
      lambda theta, x: self.apply(
        params,
        theta.reshape([1, 6]),
        x.reshape([1, 6])
      ).squeeze()
    )(theta, y)

    return -jnp.mean(log_prob), opt_state_resnet

  def __init__(
    self,
    compressor,
    nf,
    optimizer,
    loss_name,
    nb_pixels,
    nb_bins
  ):

    self.compressor = compressor
    self.nf = nf
    self.optimizer = optimizer
    self.nb_pixels = nb_pixels
    self.nb_bins = nb_bins

    if loss_name == ' mse':
      self.loss = self.loss_mse
    elif loss_name == 'vmim':
      self.loss = self.loss_vmim

  @partial(jax.jit, static_argnums=(0,))
  def update(
    self,
    model_params,
    opt_state,
    theta,
    x,
    state_resnet
  ):

    (loss, opt_state_resnet), grads = jax.value_and_grad(
      self.loss,
      has_aux=True
    )(model_params, theta, x, state_resnet)

    updates, new_opt_state = self.optimizer.update(
      grads,
      opt_state
    )

    new_params = optax.apply_updates(
      model_params,
      updates
    )

    return loss, new_params, new_opt_state, opt_state_resnet
