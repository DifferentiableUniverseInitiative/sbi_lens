<h1 align='center'>sbi_lens</h1>

<div align="center">
    
[![CI Test](https://github.com/DifferentiableUniverseInitiative/sbi_lens/workflows/Python%20package/badge.svg)]() [![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/DifferentiableUniverseInitiative/sbi_lens/blob/main/LICENSE) [![All Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg?)](https://github.com/DifferentiableUniverseInitiative/sbi_lens/graphs/contributors)

</div>

<hr><hr>

<h2 align='center'>JAX-based log-normal lensing simulation package.</h2>

**sbi_lens** provides a diferentiable log-normal mass map simulator with 5 tomographic redshift bins and 6 cosmological parameters to infer ($\Omega_c, \Omega_b, \sigma_8, n_s, w_0, h_0$). The shift parameter is computed with [CosMomentum](https://github.com/OliverFHD/CosMomentum) and depends on $\Omega_c, \sigma_8, w_0$.

Note: only LSST year 10 implemented for the moment.

# Installation

```sh
pip install git+https://github.com/DifferentiableUniverseInitiative/sbi_lens.git
```
# Quick example

``` python

# load lsst year 10 settings
from sbi_lens.config import config_lsst_y_10

N                = config_lsst_y_10.N
map_size         = config_lsst_y_10.map_size
sigma_e          = config_lsst_y_10.sigma_e
gals_per_arcmin2 = config_lsst_y_10.gals_per_arcmin2
nbins            = config_lsst_y_10.nbins
a                = config_lsst_y_10.a
b                = config_lsst_y_10.b
z0               = config_lsst_y_10.z0


# define lsst year 10 log normal model
from sbi_lens.simulator.LogNormal_field import lensingLogNormal

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
    model_type='lognormal',
    lognormal_shifts='LSSTY10',
    with_noise=False,
)

# simulate one mass map
from sbi_lens.simulator.utils import get_samples_and_scores

(log_prob, samples), gradients = get_samples_and_scores(
  model,
  PRNGKey(0),
  batch_size=1,
  with_noise=False
)
map_example = samples['y']
```

``` python
for i in range(5):
  subplot(1,5, i+1)
  imshow(map_example[0][...,i], cmap='cividis')
  title('Bin %d'%(i+1))
  axis('off')
```
<p align=center>
    <img src="img/convergence_map.png" style="width:1000px;">
</p>

Check out a full example here: [![colab link](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pSjhrOJbVi80RQlsVz2oXhVAtxwBhSbn?usp=sharing)


# Contributors

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):


<table>
  <tr>
    <td align="center"><a href="https://aboucaud.github.io"><img src="https://avatars0.githubusercontent.com/u/3065310?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Alexandre Boucaud</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/sbi_lens/commits?author=aboucaud" title="Code">💻</a></td>
    <td align="center"><a href="http://flanusse.net"><img src="https://avatars0.githubusercontent.com/u/861591?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Francois Lanusse</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/sbi_lens/commits?author=EiffL" title="Code">💻</a></td>
    <td align="center"><a href="https://www.cosmostat.org/people/denise-lanzieri"><img src="https://avatars.githubusercontent.com/u/72620117?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Denise Lanzieri</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/sbi_lens/commits?author=dlanzieri" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/Justinezgh"><img src="https://avatars.githubusercontent.com/u/72011736?v=4" width="100px;" alt=""/><br /><sub><b>Justine Zeghal</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/sbi_lens/commits?author=Justinezgh" title="Code">💻</a></td>
  </tr>
</table>
