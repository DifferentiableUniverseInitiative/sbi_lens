import argparse
import logging
from functools import partial

import jax
import jax.numpy as jnp
import numpyro
from jax.lib import xla_bridge
from numpyro.handlers import condition, seed, trace

from sbi_lens.simulator import lensingLogNormal
from sbi_lens.simulator.config import config_lsst_y_10
from sbi_lens.simulator.utils import ForwardModelMassMap

print(xla_bridge.get_backend().platform)


logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())
"unset XLA_FLAGS"


######### SCRIPT ARGS ##########
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=4)
parser.add_argument("--filename", type=str, default="res")
args = parser.parse_args()


######### RANDOM SEED ##########
key = jax.random.PRNGKey(3)
subkey = jax.random.split(key, 200)
key_par = subkey[args.seed]


######### HACK TO INIT MCMC ON FID ##########
# condition numpyro model on fid
model = partial(lensingLogNormal, model_type="lognormal", with_noise=True)

fiducial_model = condition(
    model,
    {
        "omega_c": 0.2664,
        "omega_b": 0.0492,
        "sigma_8": 0.831,
        "h_0": 0.6727,
        "n_s": 0.9645,
        "w_0": -1.0,
    },
)

# same seed we used to generate our fixed fid map
model_trace = trace(seed(fiducial_model, jax.random.PRNGKey(42))).get_trace()

# hence we obtain the exact same fid map
m_data = model_trace["y"]["value"]

# the fid values to init the mcmc
init_values = {
    k: model_trace[k]["value"]
    for k in ["z", "omega_c", "sigma_8", "omega_b", "h_0", "n_s", "w_0"]
}


######### RUN FULL-FIELD MCMC ##########
N = config_lsst_y_10.N
map_size = config_lsst_y_10.map_size
gals_per_arcmin2 = config_lsst_y_10.gals_per_arcmin2
sigma_e = config_lsst_y_10.sigma_e


model_field = ForwardModelMassMap(
    config_lsst_y_10,
    model_type="lognormal",
    lognormal_shifts="LSSTY10",
    with_noise=True,
)

posterior, diagnostic = model_field.run_mcmc(
    m_data=m_data,
    num_results=1000,
    num_warmup=200,
    max_tree_depth=6,
    step_size=1e-2,
    num_chains=1,
    nb_loop=5,
    init_strat=numpyro.infer.init_to_value(values=init_values),
    chain_method="vectorized",
    key=key_par,
)


######### SAVE CHAINS ##########
print("diagnostic: ", diagnostic)


jnp.save(
    "chains/posterior_full_field_job_{}_{}N_{}ms_{}gpa_{}se.npy".format(
        args.filename, N, map_size, gals_per_arcmin2, sigma_e
    ),
    posterior,
)

jnp.save(
    "diagnostic/diagnostic_job_{}_{}N_{}ms_{}gpa_{}se.npy".format(
        args.filename, N, map_size, gals_per_arcmin2, sigma_e
    ),
    diagnostic,
)
