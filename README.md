# sbi_lens

# Installation 

```sh
pip install git+https://github.com/DifferentiableUniverseInitiative/sbi_lens.git
```

# Usage

Imports packages

``` python 
%pylab inline
import sbi_lens
from chainconsumer import ChainConsumer
from functools import partial 
```

First, we create our fiducials. For this, we can create our observation from our true parameters and run MCMCs to get
reference posteriors from both full field inference or power spectrum one.

``` python 
from sbi_lens.simulator.utils import (
    get_reference_sample_posterior_full_field, 
    get_reference_sample_posterior_power_spectrum
)
from sbi_lens.simulator import lensingLogNormal

# define lensing model
model = partial(lensingLogNormal,  
                N=128, 
                map_size=5,
                gal_per_arcmin2=30, 
                sigma_e=0.2, 
                model_type='lognormal')

# condition the model on a given set of parameters which is 
# considered as 'true parameters'
fiducial_model = condition(model, {'omega_c': 0.3, 'sigma_8': 0.8})

# sample a mass map considered as our observation
sample_map_fiducial = seed(fiducial_model, jax.random.PRNGKey(42))
m_data = sample_map_fiducial()

# run MCMCs
samples_ps = get_reference_sample_posterior_power_spectrum(
    run_mcmc=True,
    N=128,
    map_size=5,
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

Or directly load existing ones.

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
