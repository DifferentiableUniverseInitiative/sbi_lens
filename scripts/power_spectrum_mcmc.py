import logging

import jax
import jax.numpy as jnp
import numpyro
from jax.lib import xla_bridge

from sbi_lens.simulator.config import config_lsst_y_10
from sbi_lens.simulator.utils import ForwardModelPowerSpectrum

print(xla_bridge.get_backend().platform)


logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())
"unset XLA_FLAGS"


N = config_lsst_y_10.N
map_size = config_lsst_y_10.map_size
gals_per_arcmin2 = config_lsst_y_10.gals_per_arcmin2
sigma_e = config_lsst_y_10.sigma_e

m_data = jnp.load(
    f"/linkhome/rech/genkqu01/ufa23yn/sbi_lens/sbi_lens/data/m_data__{N}N_{map_size}ms_{gals_per_arcmin2}gpa_{sigma_e}se.npy"
)

model_ps = ForwardModelPowerSpectrum(config_lsst_y_10)

posterior, diagnostic = model_ps.run_mcmc(
    m_data=m_data,
    fixed_covariance=True,
    num_results=12_000,
    num_warmup=200,
    num_chains=16,
    chain_method="vectorized",
    max_tree_depth=5,
    step_size=1e-2,
    init_strat=numpyro.infer.init_to_median,
    key=jax.random.PRNGKey(3),
)


print("diagnostic: ", diagnostic)


jnp.save(
    "posterior_power_spectrum__{}N_{}ms_{}gpa_{}se.npy".format(
        N, map_size, gals_per_arcmin2, sigma_e
    ),
    posterior,
)
