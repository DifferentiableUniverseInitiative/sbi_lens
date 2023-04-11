from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpyro
import numpyro.distributions as dist
from jax.scipy.ndimage import map_coordinates
from sbi_lens.simulator.redshift import subdivide

__all__ = ['lensingLogNormal']

SOURCE_FILE = Path(__file__)
SOURCE_DIR = SOURCE_FILE.parent
ROOT_DIR = SOURCE_DIR.parent.resolve()
DATA_DIR = ROOT_DIR / "data"

@jax.jit
def shift_fn(params, theta):
  """
  Compute the interpolated shift parameters of the lognormal distribution as function of cosmology
  -----------

  params: Jax.numpy.ndarray
  The interpolation table of the shift parameters, with dimensions [n_omega_m, n_sigma_8, n_w, n_bins, 4] 

  theta: jax.numpy.ndarray
  Nd array of parameters [omega_m, sigma_8, w]

  Returns
  -------
  lambda_shift: Jax.DeviceArray
  Interpolated shift parameters for each redshift bin
  """
  ntheta = len(params.shape[:-2])
  nbins = params.shape[-2]
  shift = [
      map_coordinates(params[..., i, -1],
                      jnp.stack([(theta[j] - params[...,j].min()) / (params[...,j].max() - params[...,j].min()) * params.shape[j] - 0.5
                                  for j in range(ntheta)],
                                axis=0).reshape([ntheta, -1]),
                      order=1,
                      mode='nearest').squeeze() for i in range(params.shape[-2])
  ]
  return jnp.stack(shift)


def fill_shift_array(shifts):
  """
  Calculate the product of the shift parameters between auto and cross redshift bins
  -----------
  shifts: Jax.DeviceArray
  Value of the shift parameter of the lognormal distribution with a given cosmology for each redshift bin

  Returns
  -------
  shift_array: Jax.DeviceArray
   Array containing the proudct 𝜆i*𝜆j with i and j ith and jth redshift bin
  """
  idx = jnp.mask_indices(len(shifts), jnp.triu)
  shift_array = jnp.outer(shifts, shifts)[idx]
  return shift_array


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


def make_lognormal_power_map(power_map, pshift, zero_freq_val=0.0):
  """
  Calculate Log-Normal lensing fields from a given Gaussian fields.
  -----------
  power_map: Jax.DeviceArray
  Fourier-space Gaussian fields generated with a given power spectrum

  pshift: Jax.DeviceArray
  The product of the shift parameters of the lognormal distribution with a given cosmology, between the auto and cross redshift bin

  zero_freq_val: float
  The zero point to shift the vector

  Returns
  -------
  power_spectrum_for_lognorm: Jax.DeviceArray
  Log-Normal lensing fields
  """
  power_spectrum_for_lognorm = jnp.fft.ifft2(power_map).real
  power_spectrum_for_lognorm = jnp.log(1 + power_spectrum_for_lognorm / pshift)
  power_spectrum_for_lognorm = jnp.abs(
      jnp.fft.fft2(power_spectrum_for_lognorm))
  power_spectrum_for_lognorm = power_spectrum_for_lognorm.at[0, 0].set(0.)
  return power_spectrum_for_lognorm


def lensingLogNormal(
    N=256,
    map_size=10,
    gal_per_arcmin2=27,
    sigma_e=0.26,
    nbins=5,
    a=2,
    b=0.68,
    z0=0.11,
    model_type='lognormal',
    lognormal_shifts='LSSTY10',
    with_noise=False,
):
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

  nbins : Int
  Number of redshift bins (with equal galaxy number)

  a, b, zo: float
  Parameters defining the redshift distribution

  model_type: string
  Physical model adopted for the simulations

  with_noise : boolean
  If True Gaussian noise will be added to the lensing map

  Returns
  -------
  x: Jax.DeviceArray (N,N)
  Lensing convergence map
  """
  # Field parameters
  pix_area = (map_size * 60 / N)**2
  map_size = map_size / 180 * jnp.pi
  
  # Cosmological parameters
  omega_c = numpyro.sample('omega_c', dist.TruncatedNormal(0.2664, 0.2, low=0))
  omega_b = numpyro.sample('omega_b', dist.Normal(0.0492, 0.006))
  sigma_8 = numpyro.sample('sigma_8', dist.Normal(0.831, 0.14))
  h_0 = numpyro.sample('h_0', dist.Normal(0.6727, 0.063))
  n_s = numpyro.sample('n_s', dist.Normal(0.9645, 0.08))
  w_0 = numpyro.sample('w_0', dist.Normal(-1.0, 0.9))

  cosmo = jc.Planck15(Omega_c=omega_c,
                      Omega_b=omega_b,
                      h=h_0,
                      n_s=n_s,
                      sigma8=sigma_8,
                      w0=w_0)

  # Load shift parameters corresponding to the model
  if model_type == 'lognormal':
    if lognormal_shifts == 'LSSTY10':
      lognormal_shifts_params = np.load(DATA_DIR / 'lognormal_shifts_LSSTY10_om_s8_w_bin.npy')
    else:
      raise NotImplementedError
    shift = shift_fn(lognormal_shifts_params, jnp.array([cosmo.Omega_m, cosmo.sigma_8, cosmo.w0]))
    shift_array = fill_shift_array(shift)

  # Define redshift distribution
  nz = jc.redshift.smail_nz(a, b, z0, gals_per_arcmin2=gal_per_arcmin2)
  nz_bins = subdivide(nz, nbins=nbins)
  tracer = jc.probes.WeakLensing(nz_bins, sigma_e=sigma_e)

  # Calculate power spectrum
  ell_tab = 2 * jnp.pi * abs(jnp.fft.fftfreq(2 * N, d=map_size / (2 * N)))
  cell_tab = jc.angular_cl.angular_cl(cosmo, ell_tab, [tracer])
  power = []
  for cl, l_shift in zip(cell_tab, shift_array):
    P = lambda k: jc.scipy.interpolate.interp(k.flatten(), ell_tab, cl
                                              ).reshape(k.shape)
    power_map = make_power_map(P, N, map_size)
    if model_type == 'lognormal':
      power_map = make_lognormal_power_map(power_map, l_shift)
    power.append(power_map)
  power = jnp.stack(power, axis=-1)

  @jax.vmap
  def fill_cov_mat(m):
    idx = np.triu_indices(nbins)
    cov_mat = jnp.zeros((nbins, nbins)).at[idx].set(m).T.at[idx].set(m)
    return cov_mat

  cov_mat = fill_cov_mat(power.reshape(-1, len(cell_tab)))
  l, A = jnp.linalg.eigh(cov_mat)
  L = jax.vmap(
      lambda M, v: M.dot(jnp.diag(jnp.sqrt(jnp.clip(v, a_min=0))).dot(M.T)))(A,
                                                                             l)
  L = L.reshape([N, N, nbins, nbins])
  L = L.at[0, 0].set(jnp.zeros((nbins, nbins)))
  L = L.transpose([2, 3, 0, 1])

  z = numpyro.sample(
      'z',
      dist.MultivariateNormal(loc=jnp.zeros((nbins, N, N)),
                              precision_matrix=jnp.eye(N)))
  field = (jnp.fft.fft2(z) * L)
  field = (jnp.fft.ifft2(jnp.sum(field, axis=1)).real)
  if model_type == 'lognormal':
    field = jnp.einsum(
        'i, ijk -> ijk', shift,
        jnp.exp(field - jnp.var(field, axis=(1, 2), keepdims=True) / 2) - 1)
  
  # Transposing the field to put the redshift axis last
  field = jnp.transpose(field, [1,2,0])

  # Adding noise
  if with_noise == True:
    x = numpyro.sample(
        'y',
        dist.MultivariateNormal(loc=field, 
                                covariance_matrix=jnp.diag(sigma_e**2 / (jnp.array([b.gal_per_arcmin2 for b in nz_bins]) * pix_area)))
        )
  else:
    x = numpyro.deterministic('y', field)

  return x
