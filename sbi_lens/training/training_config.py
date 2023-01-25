from functools import partial
import jax
import tensorflow_probability as tfp

tfp = tfp.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors
import haiku as hk
from sbi_lens.normflow.models import AffineSigmoidCoupling, ConditionalRealNVP

import optax
from sbi_lens.training.losses import loss_vmim, loss_fn


@jax.jit
def update_compressor_with_vmim(params, opt_state, mu, batch, state_resnet,
                                optimizer_c):
    """Single SGD update step."""
    (loss,
     opt_state_resnet), grads = jax.value_and_grad(loss_vmim,
                                                   has_aux=True)(params, mu,
                                                                 batch,
                                                                 state_resnet)
    updates, new_opt_state = optimizer_c.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    return loss, new_params, new_opt_state, opt_state_resnet


@jax.jit
def update(params, parameters_compressor, opt_state, opt_state_resnet, weight,
           mu, batch, score, optimizer):
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(loss_fn)(params, parameters_compressor,
                                              opt_state_resnet, weight, mu,
                                              batch, score)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return loss, new_params, new_opt_state


def compressor_conf():
    bijector_layers_compressor = [128] * 2
    bijector_compressor = partial(AffineSigmoidCoupling,
                                  layers=bijector_layers_compressor,
                                  n_components=16,
                                  activation=jax.nn.silu)
    NF_compressor = partial(ConditionalRealNVP,
                            n_layers=4,
                            bijector_fn=bijector_compressor)

    class Flow_nd_Compressor(hk.Module):

        def __call__(self, y):
            nvp = NF_compressor(2)(y)
            return nvp

    nf = hk.without_apply_rng(
        hk.transform(lambda theta, y: Flow_nd_Compressor()
                     (y).log_prob(theta).squeeze()))
    return nf


def estimator_conf(scale_theta, shift_theta, sample):
    # create model for inference
    bijector_layers = [128] * 2
    bijector_npe = partial(AffineSigmoidCoupling,
                           layers=bijector_layers,
                           n_components=16,
                           activation=jax.nn.silu)
    NF_npe = partial(ConditionalRealNVP, n_layers=4, bijector_fn=bijector_npe)

    class SmoothNPE(hk.Module):

        def __call__(self, y):
            net = y
            nvp = NF_npe(2)(net)
            return tfd.TransformedDistribution(
                nvp,
                tfb.Chain([tfb.Scale(scale_theta),
                           tfb.Shift(shift_theta)]))

    nvp_nd = hk.without_apply_rng(
        hk.transform(lambda theta, y: SmoothNPE()
                     (y).log_prob(theta).squeeze()))
    nvp_sample_nd = hk.transform(
        lambda x: SmoothNPE()(x).sample(len(sample), seed=hk.next_rng_key()))
    return nvp_nd, nvp_sample_nd
