# sbi_lens

# Installation

```sh
pip install git+https://github.com/DifferentiableUniverseInitiative/sbi_lens.git
```

# Usage

Imports packages.

``` python
from functools import partial
import jax
from numpyro.handlers import seed, condition
from sbi_lens.simulator import lensingLogNormal
from sbi_lens.simulator.utils import (
    get_reference_sample_posterior_full_field,
    get_reference_sample_posterior_power_spectrum
)
```

First, we create our fiducials. For this, we define our [lensing model](https://github.com/DifferentiableUniverseInitiative/sbi_lens/blob/main/sbi_lens/simulator/LogNormal_field.py), condition it on our true parameters $\Omega_c$ and $\sigma_8$ and simulate a mass map. Then, we run MCMCs to get reference posteriors from both full field inference and power spectrum one.

``` python
# define lensing model
model = partial(lensingLogNormal,
                N=128,
                map_size=5,
                gal_per_arcmin2=30,
                sigma_e=0.2,
                model_type='lognormal')

# condition the model on a given set of parameters
fiducial_model = condition(model, {'omega_c': 0.3, 'sigma_8': 0.8})

# sample a mass map
sample_map_fiducial = seed(fiducial_model, jax.random.PRNGKey(42))
m_data = sample_map_fiducial()

# run MCMCs
samples_ps = get_reference_sample_posterior_power_spectrum(
    run_mcmc=True,
    gals_per_arcmin2=30,
    sigma_e=0.2,
    m_data=m_data,
    num_results=10000,
    key=jax.random.PRNGKey(0)
)
samples_ff = get_reference_sample_posterior_full_field(
    run_mcmc=True,
    N=128,
    map_size=5,
    gals_per_arcmin2=30,
    sigma_e=0.2,
    model=model,
    m_data=m_data,
    num_results=10000,
    key=jax.random.PRNGKey(0)
)
```
<p align=center>
    <img src="img/doc_observation.png" style="width:350px;">
    <img src="img/doc_contour.png" style="width:300px;">
</p>

Or we can directly load existing ones.

``` python
# load reference posteriors, observation m_data, and true parameters
samples_ps, m_data, truth = get_reference_sample_posterior_power_spectrum(
    run_mcmc=False,
    N=128,
    map_size=5,
    gals_per_arcmin2=30,
    sigma_e=0.2,
)
samples_ff, _, _ = get_reference_sample_posterior_full_field(
    run_mcmc=False,
    N=128,
    map_size=5,
    gals_per_arcmin2=30,
    sigma_e=0.2,
)
```

# Contributors

# Licence

MIT
