from functools import partial
import jax
import tensorflow_probability as tfp

tfp = tfp.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors
import haiku as hk
from sbi_lens.normflow.models import AffineSigmoidCoupling, ConditionalRealNVP
import jax.numpy as jnp
from haiku._src.nets.resnet import ResNet18


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
    params_nf = nf.init(jax.random.PRNGKey(8), 0.5 * jnp.ones([1, 2]),
                        0.5 * jnp.ones([1, 2]))
    compressor = hk.transform_with_state(lambda x: ResNet18(2)
                                         (x, is_training=True))
    parameters_resnet, opt_state_resnet = compressor.init(
        jax.random.PRNGKey(873457568), 0.5 * jnp.ones([1, 128, 128, 1]))
    parameters_compressor = hk.data_structures.merge(parameters_resnet,
                                                     params_nf)
    return nf, compressor, parameters_compressor, opt_state_resnet


def estimator_conf(scale_theta, shift_theta, sample):
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




