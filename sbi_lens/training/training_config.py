from functools import partial
import jax
import tensorflow_probability as tfp
tfp = tfp.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors
import haiku as hk
from sbi_lens.normflow.models import AffineSigmoidCoupling, ConditionalRealNVP


def compressor_conf():
    bijector_layers_compressor = [128] * 2
    bijector_compressor = partial(AffineSigmoidCoupling,
                              layers=bijector_layers_compressor,
                              n_components=16,
                              activation=jax.nn.silu)
    NF_compressor = partial(ConditionalRealNVP,
                        n_layers=4,
                        bijector_fn=bijector_compressor)
        
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
                nvp, tfb.Chain([tfb.Scale(scale_theta),
                                tfb.Shift(shift_theta)]))
    nvp_nd = hk.without_apply_rng(
        hk.transform(lambda theta, y: SmoothNPE()(y).log_prob(theta).squeeze()))
    nvp_sample_nd = hk.transform(lambda x : SmoothNPE()(x).sample(len(sample), seed=hk.next_rng_key()))
    return nvp_nd

    