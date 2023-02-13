from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpyro
import numpyro.distributions as dist
from jax.scipy.ndimage import map_coordinates

__all__ = ['lensingLogNormal']

SOURCE_FILE = Path(__file__)
SOURCE_DIR = SOURCE_FILE.parent
ROOT_DIR = SOURCE_DIR.parent.resolve()
DATA_DIR = ROOT_DIR / "data"

lognormal_params = np.loadtxt(DATA_DIR / "lognormal_shift.csv",
                              skiprows=1,
                              delimiter=',').reshape([8, 8, 3])


@jax.jit
def shift_fn(omega_m, sigma_8):
    """
  Compute the interpolated shift parameters of the lognormal distribution as function of cosmology
  -----------
  omega_m: float
  The total matter density fraction.

  sigma_8: float
  Variance of matter density perturbations at an 8 Mpc/h scale.

  Returns
  -------
  lambda_shift: float
  Interpolated shift parameters

  """
    omega_m = jnp.atleast_1d(omega_m)
    sigma_8 = jnp.atleast_1d(sigma_8)
    lambda_shift = map_coordinates(lognormal_params[:, :, 2],
                                   jnp.stack([(omega_m - 0.2) / 0.2 * 8 - 0.5,
                                              (sigma_8 - 0.6) / 0.4 * 8 - 0.5],
                                             axis=0).reshape([2, -1]),
                                   order=1,
                                   mode='nearest').squeeze()

    return lambda_shift


def make_power_map(pk_fn, N, map_size, zero_freq_val=0.0):
    """
  Calculate fourier-space Gaussian fields generated with a given power spectrum.
  -----------
  pk_fn:
   Given power spectrum

  N: Int
   Number of grid-points on a side or number of wavenumbers to use.

  map_size: Int
  The total angular size area is given by map_size x map_size

  zero_freq_val: float
   The zero point to shift the vector

  Returns
  -------
  power_map: Jax.DeviceArray (N,N)
  Image

  """
    k = 2 * jnp.pi * jnp.fft.fftfreq(N, d=map_size / N)
    kcoords = jnp.meshgrid(k, k)
    k = jnp.sqrt(kcoords[0]**2 + kcoords[1]**2)
    ps_map = pk_fn(k)
    ps_map = ps_map.at[0, 0].set(zero_freq_val)
    power_map = ps_map * (N / map_size)**2
    return power_map


def make_lognormal_power_map(power_map, shift, zero_freq_val=0.0):
    """
  Calculate Log-Normal lensing fields with given Gaussian fields.
  -----------
  power_map: Jax.DeviceArray
  Fourier-space Gaussian fields generated with a given power spectrum

  shift: Float
  The shift parameter of the lognormal distribution with a given cosmology

  zero_freq_val: float
  The zero point to shift the vector

  Returns
  -------
  power_spectrum_for_lognorm: Jax.DeviceArray
  Log-Normal lensing fields
  """
    power_spectrum_for_lognorm = jnp.fft.ifft2(power_map).real
    power_spectrum_for_lognorm = jnp.log(1 +
                                         power_spectrum_for_lognorm / shift**2)
    power_spectrum_for_lognorm = jnp.abs(
        jnp.fft.fft2(power_spectrum_for_lognorm))
    power_spectrum_for_lognorm = power_spectrum_for_lognorm.at[0, 0].set(0.)
    return power_spectrum_for_lognorm


def lensingLogNormal(N=128,
                     map_size=5,
                     gal_per_arcmin2=10,
                     sigma_e=0.26,
                     model_type='lognormal',
                     with_noise=True):
    """
  Calculate Log-Normal lensing convergence map.
  -----------
  N: int
  Number of pixels on the map.

  map_size: int
  The total angular size area is given by map_size x map_size

  gal_per_arcmin2: int
  Number of galaxies per arcmin


  sigma_e : float
  Dispersion of the ellipticity distribution

  model_type: string
  Physcal model adopted for the simulations

  with_noise : boolean
  If True Gaussian noise will be added to the lensing map

  Returns
  -------
  x: Jax.DeviceArray (N,N)
  Lensing convergence map
  """

    pix_area = (map_size * 60 / N)**2
    map_size = map_size / 180 * jnp.pi
    omega_c = numpyro.sample('omega_c', dist.Normal(0.3, 0.05))
    sigma_8 = numpyro.sample('sigma_8', dist.Normal(0.8, 0.05))
    cosmo = jc.Planck15(Omega_c=omega_c, sigma8=sigma_8)
    pz = jc.redshift.smail_nz(0.5, 2., 1.0)
    tracer = jc.probes.WeakLensing([pz])
    ell_tab = jnp.logspace(0, 4.5, 128)
    cell_tab = jc.angular_cl.angular_cl(cosmo, ell_tab, [tracer])[0]
    P = lambda k: jc.scipy.interpolate.interp(k.flatten(), ell_tab, cell_tab
                                              ).reshape(k.shape)
    z = numpyro.sample(
        'z',
        dist.MultivariateNormal(loc=jnp.zeros((N, N)),
                                precision_matrix=jnp.eye(N)))

    power_map = make_power_map(P, N, map_size)
    if model_type == 'lognormal':
        shift = shift_fn(cosmo.Omega_m, sigma_8)
        power_map = make_lognormal_power_map(power_map, shift)
    field = jnp.fft.ifft2(jnp.fft.fft2(z) * jnp.sqrt(power_map)).real
    if model_type == 'lognormal':
        field = shift * (jnp.exp(field - jnp.var(field) / 2) - 1)

    if with_noise == True:
        x = numpyro.sample(
            'y',
            dist.Independent(
                dist.Normal(field,
                            sigma_e / jnp.sqrt(gal_per_arcmin2 * pix_area)),
                2))
    else:
        x = numpyro.deterministic('y', field)

    return x
