import pytest
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp
from numpy.testing import assert_allclose, assert_equal
from sbi_lens.normflow.bijectors import MixtureAffineSigmoidBijector

xfail = pytest.mark.xfail
tfp = tfp.substrates.jax
tfd = tfp.distributions

_test_params_MASB = [{
    'a': 2.,
    'b': 0.25,
    'c': 0.1,
    'nb_dimension': 2,
    'nb_component': 4
}, {
    'a': 0.01,
    'b': 0.75,
    'c': 0.1,
    'nb_dimension': 3,
    'nb_component': 5
}, {
    'a': 1.,
    'b': 0.05,
    'c': 0.99,
    'nb_dimension': 4,
    'nb_component': 2
}]


def test_bijectorMixtureAffineSigmoidBijector_inverse():
  """
    Testing that the inverse transform of MixtureAffineSigmoidBijector
    is indeed the inverse of forward.
    """

  batch_size = 100

  for params in _test_params_MASB:
    x = jax.random.uniform(jax.random.PRNGKey(0),
                           shape=[batch_size, params['nb_dimension']])

    # shape for components of the transform
    shape = [batch_size, params['nb_dimension'], params['nb_component']]
    bij = MixtureAffineSigmoidBijector(
        jnp.ones(shape) * params['a'],
        jnp.ones(shape) * params['b'],
        jnp.ones(shape) * params['c'],
        jax.nn.softmax(jax.random.uniform(jax.random.PRNGKey(1), shape=shape)))

  assert_allclose(bij.inverse(bij.forward(x) * 1.), x, rtol=2e-2, atol=1e-5)


def test_bijectorMixtureAffineSigmoidBijector_fldj_dim():
  """
    Testing forward_log_det_jacobian dimension of MixtureAffineSigmoidBijector
    """
  batch_size = 100

  for params in _test_params_MASB:
    x = jax.random.uniform(jax.random.PRNGKey(0),
                           shape=[batch_size, params['nb_dimension']])

    # shape for components of the transform
    shape = [batch_size, params['nb_dimension'], params['nb_component']]

    bij = MixtureAffineSigmoidBijector(
        jnp.ones(shape) * params['a'],
        jnp.ones(shape) * params['b'],
        jnp.ones(shape) * params['c'],
        jax.nn.softmax(jax.random.uniform(jax.random.PRNGKey(1), shape=shape)))

  assert_equal(
      bij.forward_log_det_jacobian(x * 1).shape,
      (batch_size, params['nb_dimension']),
      'forward_log_det_jacobian output dim is not (batch_size,nb_dimension)')
