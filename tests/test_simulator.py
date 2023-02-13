import jax
from numpyro.handlers import seed, condition
import numpy as np

import tensorflow_probability as tfp

tfp = tfp.substrates.jax
tfd = tfp.distributions

from sbi_lens.simulator.LogNormal_field import lensingLogNormal
import lenstools as lt
import astropy.units as u
from numpy.testing import assert_allclose

from functools import partial

jax.config.update("jax_enable_x64", True)


def test_LogNormalmodel():
    #Create our fiducial observations
    model = partial(lensingLogNormal,
                    N=64,
                    map_size=5,
                    gal_per_arcmin2=10,
                    sigma_e=0.26,
                    model_type='lognormal',
                    with_noise=True)
    fiducial_model = condition(model, {'omega_c': 0.3, 'sigma_8': 0.8})
    sample_map_fiducial = seed(fiducial_model, jax.random.PRNGKey(42))
    m_data = sample_map_fiducial()
    kmap_lt = lt.ConvergenceMap(m_data, 5 * u.deg)
    l_edges = np.arange(100.0, 2000.0, 50.0)
    Pl2 = kmap_lt.powerSpectrum(l_edges)[1]
    # Check against precomputed value with the same seed
    Pl2_prec = np.array([
        2.18986080e-09, 3.17677659e-09, 2.90191771e-09, 1.17660875e-09,
        3.82220752e-10, 1.20896672e-09, 1.05876556e-09, 8.19649374e-10,
        1.01941782e-09, 1.05519042e-09, 1.12039812e-09, 8.78809269e-10,
        6.81467157e-10, 7.26929807e-10, 1.19080587e-09, 6.36538321e-10,
        6.06747461e-10, 4.61243165e-10, 7.90651522e-10, 6.69439969e-10,
        8.28188277e-10, 7.14728737e-10, 7.46595429e-10, 9.61709632e-10,
        6.99437023e-10, 5.90386930e-10, 5.75511855e-10, 6.78740962e-10,
        6.62436825e-10, 5.71815616e-10, 6.91627736e-10, 6.51595612e-10,
        6.58286216e-10, 7.26794876e-10, 6.97780612e-10, 7.41781716e-10,
        6.46185090e-10
    ])

    assert_allclose(Pl2, Pl2_prec, atol=1e-5)
