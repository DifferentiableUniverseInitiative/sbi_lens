from functools import partial

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose
from numpyro.handlers import condition, seed, trace
from sbi_lens.config import config_lsst_y_10
from sbi_lens.simulator.LogNormal_field import lensingLogNormal
from sbi_lens.simulator.utils import compute_power_spectrum


def test_LogNormalmodel():
    N = config_lsst_y_10.N
    map_size = config_lsst_y_10.map_size
    sigma_e = config_lsst_y_10.sigma_e
    gals_per_arcmin2 = config_lsst_y_10.gals_per_arcmin2
    nbins = config_lsst_y_10.nbins
    a = config_lsst_y_10.a
    b = config_lsst_y_10.b
    z0 = config_lsst_y_10.z0

    params_name = config_lsst_y_10.params_name

    # define model LSST Y 10
    model = partial(
        lensingLogNormal,
        N=N,
        map_size=map_size,
        gal_per_arcmin2=gals_per_arcmin2,
        sigma_e=sigma_e,
        nbins=nbins,
        a=a,
        b=b,
        z0=z0,
        model_type="lognormal",
        lognormal_shifts="LSSTY10",
        with_noise=False,
    )

    @jax.vmap
    @jax.jit
    def get_batch(key):
        def get_maps(k, theta):
            model_see_cond = condition(
                seed(model, k),
                {
                    "omega_c": theta[0],
                    "omega_b": theta[1],
                    "sigma_8": theta[2],
                    "h_0": theta[3],
                    "n_s": theta[4],
                    "w_0": theta[5],
                },
            )
            obs = trace(model_see_cond).get_trace()["y"]["value"]
            return obs

        model_trace = trace(seed(model, key)).get_trace()
        theta = jnp.stack([model_trace[name]["value"] for name in params_name], axis=-1)

        obs = (lambda key: jax.vmap(get_maps, in_axes=(0, None))(key, theta))(
            jax.random.split(key, 10)
        )
        return obs, theta

    N_sample = 5
    m_data, cosmo_params = get_batch(jax.random.split(jax.random.PRNGKey(14), N_sample))

    for q in range(N_sample):
        (
            cl_the,
            cl_exp,
            _,
        ) = compute_power_spectrum(
            map_size,
            sigma_e,
            a,
            b,
            z0,
            gals_per_arcmin2,
            cosmo_params[q],
            m_data[q],
            with_noise=False,
        )
        assert_allclose(cl_exp, cl_the, atol=1e-8)
