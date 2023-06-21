import jax.numpy as jnp
import jax
import haiku as hk

import tensorflow_probability as tfp

tfp = tfp.experimental.substrates.jax
tfb = tfp.bijectors
tfd = tfp.distributions

from sbi_lens.normflow.bijectors.bijectors import MixtureAffineSigmoidBijector


class AffineCoupling(hk.Module):

  def __init__(self,
               y,
               *args,
               layers=[128, 128],
               activation=jax.nn.leaky_relu,
               **kwargs):
    """
    Args:
    y, conditioning variable
    layers, list of hidden layers
    activation, activation function for hidden layers
    """
    self.y = y
    self.layers = layers
    self.activation = activation
    super(AffineCoupling, self).__init__(*args, **kwargs)

  def __call__(self, x, output_units, **condition_kwargs):

    net = jnp.concatenate([x, self.y], axis=-1)
    for i, layer_size in enumerate(self.layers):
      net = self.activation(hk.Linear(layer_size, name='layer%d' % i)(net))

    shifter = tfb.Shift(hk.Linear(output_units)(net))
    scaler = tfb.Scale(
        jnp.clip(jnp.exp(hk.Linear(output_units)(net)), 1e-2, 1e2))
    return tfb.Chain([shifter, scaler])


class AffineSigmoidCoupling(hk.Module):
  """This is the coupling layer used in the Flow."""

  def __init__(self,
               y,
               *args,
               layers=[128, 128],
               n_components=32,
               activation=jax.nn.silu,
               **kwargs):
    """
    Args:
    y, conditioning variable
    layers, list of hidden layers
    n_components, number of mixture components
    activation, activation function for hidden layers
    """
    self.y = y
    self.layers = layers
    self.n_components = n_components
    self.activation = activation
    super(AffineSigmoidCoupling, self).__init__(*args, **kwargs)

  def __call__(self, x, output_units, **condition_kwargs):

    net = jnp.concatenate([x, self.y], axis=-1)
    for i, layer_size in enumerate(self.layers):
      net = self.activation(hk.Linear(layer_size, name='layer%d' % i)(net))

    log_a_bound = 4
    min_density_lower_bound = 1e-4
    n_components = self.n_components

    log_a = jax.nn.tanh(
        hk.Linear(output_units * n_components, name='l3')(net)) * log_a_bound
    b = hk.Linear(output_units * n_components, name='l4')(net)
    c = min_density_lower_bound + jax.nn.sigmoid(
        hk.Linear(output_units * n_components, name='l5')
        (net)) * (1 - min_density_lower_bound)
    p = hk.Linear(output_units * n_components, name='l6')(net)

    log_a = log_a.reshape(-1, output_units, n_components)
    b = b.reshape(-1, output_units, n_components)
    c = c.reshape(-1, output_units, n_components)
    p = p.reshape(-1, output_units, n_components)
    p = jax.nn.softmax(p)

    return MixtureAffineSigmoidBijector(jnp.exp(log_a), b, c, p)


class ConditionalRealNVP(hk.Module):
  """A normalizing flow based on RealNVP using specified bijector functions."""

  def __init__(self,
               d,
               *args,
               n_layers=3,
               bijector_fn=AffineSigmoidCoupling,
               **kwargs):
    """
    Args:
    d, dimensionality of the input
    n_layers, number of layers
    coupling_layer, list of coupling layers
    """
    self.d = d
    self.n_layer = n_layers
    self.bijector_fn = bijector_fn
    super(ConditionalRealNVP, self).__init__(*args, **kwargs)

  def __call__(self, y):
    chain = tfb.Chain([
        tfb.Permute(jnp.arange(self.d)[::-1])(tfb.RealNVP(
            self.d // 2, bijector_fn=self.bijector_fn(y, name='b%d' % i)))
        for i in range(self.n_layer)
    ])

    nvp = tfd.TransformedDistribution(
      tfd.MultivariateNormalDiag(
        0.5 * jnp.ones(self.d),
        0.05 * jnp.ones(self.d)),
      bijector=chain
    )

    return nvp
