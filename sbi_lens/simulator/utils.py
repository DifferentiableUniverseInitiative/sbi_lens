from pathlib import Path
import numpy as np
import numpyro
from numpyro import sample
from numpyro.handlers import seed, trace, condition, reparam
from numpyro.infer.reparam import LocScaleReparam, TransformReparam
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import lenstools as lt
import astropy.units as u
import tensorflow_probability as tfp
import itertools
from sbi_lens.simulator.redshift import subdivide
from functools import partial

tfp = tfp.substrates.jax
tfd = tfp.distributions

SOURCE_FILE = Path(__file__)
SOURCE_DIR = SOURCE_FILE.parent
ROOT_DIR = SOURCE_DIR.parent.resolve()
DATA_DIR = ROOT_DIR / "data"


def get_samples_and_scores(
    model,
    key,
    batch_size=64,
    score_type='density',
    thetas=None,
    with_noise=True,
):
  """ Handling function sampling and computing the score from the model.

    Parameters
    ----------
    model : numpyro model
    key : PRNG Key
    batch_size : int, optional
        size of the batch to sample, by default 64
    score_type : str, optional
        'density' for nabla_theta log p(theta | y, z) or
        'conditional' for nabla_theta log p(y | z, theta), by default 'density'
    thetas : Array (batch_size, 2), optional
        thetas used to sample simulations or
        'None' sample thetas from the model, by default None
    with_noise : bool, optional
        add noise in simulations, by default True
        note: if no noise the score is only nabla_theta log p(theta, z)
        and log_prob log p(theta, z)

    Returns
    -------
    Array
        (log_prob, sample), score
    """

  params_name = ['omega_c', 'omega_b', 'sigma_8', 'h_0', 'n_s', 'w_0']

  def log_prob_fn(theta, key):
    cond_model = seed(model, key)
    cond_model = condition(
      cond_model,
      {'omega_c': theta[0],
       'omega_b': theta[1],
       'sigma_8': theta[2],
       'h_0': theta[3],
       'n_s': theta[4],
       'w_0': theta[5],
       }
    )
    model_trace = trace(cond_model).get_trace()
    sample = {
      'theta': jnp.stack(
        [model_trace[name]['value'] for name in params_name],
        axis=-1
      ),
      'y': model_trace['y']['value']
    }

    logp = 0
    if score_type == 'density':
      for name in params_name:
        logp += model_trace[name]['fn'].log_prob(model_trace[name]['value'])

    if with_noise:
      logp += model_trace['y']['fn'].log_prob(
          jax.lax.stop_gradient(model_trace['y']['value'])).sum()
    logp += model_trace['z']['fn'].log_prob(model_trace['z']['value']).sum()

    return logp, sample

  # Split the key by batch
  keys = jax.random.split(key, batch_size)

  # Sample theta from the model
  if thetas is None:
    @jax.vmap
    def get_params(key):
      model_trace = trace(seed(model, key)).get_trace()
      thetas = jnp.stack(
        [model_trace[name]['value'] for name in params_name],
        axis=-1
      )
      return thetas

    thetas = get_params(keys)

  return jax.vmap(jax.value_and_grad(log_prob_fn, has_aux=True))(thetas, keys)


def get_reference_sample_posterior_power_spectrum(
    run_mcmc=False,
    N=256,
    map_size=10,
    gals_per_arcmin2=27,
    sigma_e=0.26,
    nbins=5,
    a=2,
    b=0.68,
    z0=0.11,
    m_data=None,
    num_results=None,
    num_warmup=None,
    key=None,
):
  """ Posterior p(theta|x=m_data) from power spectrum analysis.
      Note: pre samples chains correspond to the following fiducial parameters:
      (omega_c, omega_b, sigma_8, h_0, n_s, w_0)
       = (0.2664, 0.0492, 0.831, 0.6727, 0.9645, -1.0)

    Parameters
    ----------
    run_mcmc : bool, optional
        if True the MCMC will be run,
        if False pre sampled chains are returned according to
        gals_per_arcmin2, sigma_e, N, map_size,
        by default False
    N : int, optional
        Number of pixels on the map., by default 128
    map_size : int, optional
        The total angular size area is given by map_size x map_size,
        by default 5
    gals_per_arcmin2 : int
        Number of galaxies per arcmin, by default 30
    sigma_e : float
        Dispersion of the ellipticity distribution, by default 0.2
    n_bins:Int
        Number of redshift bins
    m_data : Array (N,N)
        Lensing convergence map (only needed if run_mcmc=True), by default None
        if run_mcmc=True m_data can not be None
    num_results : int
        Number of samples (only needed if run_mcmc=True), by default None
        if run_mcmc=True num_results can not be None
    num_warmup : int
        Number of warmup steps (only needed if run_mcmc=True), by default None
        if run_mcmc=True num_warmup can not be None
    key : PRNG key
        only needed if run_mcmc=True, by default None
        if run_mcmc=True key can not be None

    Returns
    -------
    Array (num_results,2)
        MCMC chains corresponding to p(theta|x=m_data)
    """

  if run_mcmc:

    nz = jc.redshift.smail_nz(a, b, z0, gals_per_arcmin2=gals_per_arcmin2)
    nz_bins = subdivide(nz, nbins=nbins, zphot_sigma=0.05)

    l_edges = np.arange(100.0, 5000.0, 50.0)
    l2 = lt.ConvergenceMap(
      m_data[..., 0],
      map_size * u.deg
    ).powerSpectrum(l_edges)[0]
    pl_array = []
    for i, j in itertools.combinations_with_replacement(range(nbins), 2):
        pi = lt.ConvergenceMap(m_data[..., i], angle=map_size * u.deg).cross(
            lt.ConvergenceMap(m_data[..., j], angle=map_size * u.deg),
            l_edges=l_edges)[1]
        pl_array.append(pi)

    # Let's define the observations
    ell = l2
    cl_obs = np.stack(pl_array)

    def lensingPS(
      N=N,
      map_size=map_size,
      sigma_e=sigma_e,
    ):
        # Field parameters
        f_sky = map_size**2 / 41_253

        # Cosmological parameters
        omega_c = sample('omega_c', dist.TruncatedNormal(0.2664, 0.2, low=0))
        omega_b = sample('omega_b', dist.Normal(0.0492, 0.006))
        sigma_8 = sample('sigma_8', dist.Normal(0.831, 0.14))
        h_0 = sample('h_0', dist.Normal(0.6727, 0.063))
        n_s = sample('n_s', dist.Normal(0.9645, 0.08))
        w_0 = sample('w_0', dist.TruncatedNormal(-1.0, 0.9, low=-2.0, high=-0.3))

        cosmo = jc.Planck15(
            Omega_c=omega_c,
            Omega_b=omega_b,
            h=h_0,
            n_s=n_s,
            sigma8=sigma_8,
            w0=w_0
        )

        tracer = jc.probes.WeakLensing(nz_bins, sigma_e=sigma_e)

        # Calculate power spectrum
        cl_noise = jc.angular_cl.noise_cl(ell, [tracer]).flatten()
        cl, C = jc.angular_cl.gaussian_cl_covariance_and_mean(
            cosmo,
            ell,
            [tracer],
            f_sky=f_sky,
            sparse=True
        )

        # Compute precision matrix
        P = jc.sparse.to_dense(jc.sparse.inv(jax.lax.stop_gradient(C)))
        C = jc.sparse.to_dense(C)

        cl = sample(
            'cl',
            dist.MultivariateNormal(
                cl+cl_noise,
                precision_matrix=P,
                covariance_matrix=C
            )
        )

        return cl

    model_lensingPS = partial(
        lensingPS,
        N=N,
        map_size=map_size,
        sigma_e=sigma_e
    )

    # Now we condition the model on obervations
    observed_model = condition(model_lensingPS, {'cl': cl_obs.flatten()})

    def config(x):
        if type(x['fn']) is dist.TransformedDistribution:
            return TransformReparam()
        elif (type(x['fn']) is dist.Normal or type(x['fn']) is dist.TruncatedNormal)  and ('decentered' not in x['name']):
            return LocScaleReparam(centered=0)
        else:
            return None

    observed_model_reparam = reparam(observed_model, config=config)

    nuts_kernel = numpyro.infer.NUTS(
        observed_model_reparam,
        step_size=1e-2,
        init_strategy=numpyro.infer.init_to_median,
        max_tree_depth=4,
        dense_mass=True
    )

    mcmc = numpyro.infer.MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_results,
        num_chains=16,
        chain_method='vectorized',
        progress_bar=True
    )

    mcmc.run(key)
    samples = mcmc.get_samples()
    samples = jnp.stack([
        samples['omega_c'],
        samples['omega_b'],
        samples['sigma_8'],
        samples['h_0'],
        samples['n_s'],
        samples['w_0'],
    ], axis=-1)

    return samples

  else:
    SOURCE_FILE = Path(__file__)
    SOURCE_DIR = SOURCE_FILE.parent
    ROOT_DIR = SOURCE_DIR.parent.resolve()
    DATA_DIR = ROOT_DIR / "data"

    theta = np.load(DATA_DIR / "posterior_power_spectrum__"
                    "{}N_{}ms_{}gpa_{}se.npy".format(
                        N, map_size, gals_per_arcmin2, sigma_e))

    m_data = np.load(DATA_DIR / "m_data__"
                     "{}N_{}ms_{}gpa_{}se.npy".format(
                         N, map_size, gals_per_arcmin2, sigma_e))

    return theta, m_data


def get_reference_sample_posterior_full_field(
    run_mcmc=False,
    N=256,
    map_size=10,
    gals_per_arcmin2=27,
    sigma_e=0.26,
    model=None,
    m_data=None,
    num_results=None,
    num_warmup=None,
    key=None,
):
  """ Full field posterior p(theta|x=m_data).
    Note: pre samples chains correspond to the following fiducial parameters:
    (omega_c, omega_b, sigma_8, h_0, n_s, w_0)
    = (0.2664, 0.0492, 0.831, 0.6727, 0.9645, -1.0)

    Parameters
    ----------
    run_mcmc : bool, optional
        if True the MCMC will be run,
        if False pre sampled chains are returned according to
        gals_per_arcmin2, sigma_e, N, map_size,
        by default False
    N : int, optional
        Number of pixels on the map., by default 128
    map_size : int, optional
        The total angular size area is given by map_size x map_size,
        by default 5
    gals_per_arcmin2 : int
        Number of galaxies per arcmin, by default 30
    sigma_e : float
        Dispersion of the ellipticity distribution, by default 0.2
    model : numpyro model
        only needed if run_mcmc=True, by default None
        if run_mcmc=True model can not be None
    m_data : Array (N,N)
        Lensing convergence map (only needed if run_mcmc=True), by default None
        if run_mcmc=True m_data can not be None
    num_results : int
        Number of samples (only needed if run_mcmc=True), by default None
        if run_mcmc=True num_results can not be None
    num_warmup : int
        Number of warmup steps (only needed if run_mcmc=True), by default None
        if run_mcmc=True num_warmup can not be None
    key : PRNG key
        only needed if run_mcmc=True, by default None
        if run_mcmc=True key can not be None

    Returns
    -------
    Array (num_results,2)
        MCMC chains corresponding to p(theta|x=m_data)
    """

  if run_mcmc:

    def config(x):
        if type(x['fn']) is dist.TransformedDistribution:
            return TransformReparam()
        elif (type(x['fn']) is dist.Normal or type(x['fn']) is dist.TruncatedNormal) and ('decentered' not in x['name']):
            return LocScaleReparam(centered=0)
        else:
            return None

    observed_model = condition(model, {'y': m_data})
    observed_model_reparam = reparam(observed_model, config=config)
    nuts_kernel = numpyro.infer.NUTS(
        observed_model_reparam,
        init_strategy=numpyro.infer.init_to_median,
        max_tree_depth=6,
        step_size=0.02)
    mcmc = numpyro.infer.MCMC(nuts_kernel,
                              num_warmup=num_warmup,
                              num_samples=num_results,
                              progress_bar=True)

    mcmc.run(key)
    samples = mcmc.get_samples()

    return jnp.stack([
        samples['omega_c'],
        samples['omega_b'],
        samples['sigma_8'],
        samples['h_0'],
        samples['n_s'],
        samples['w_0'],
    ],
                     axis=-1)

  else:
    SOURCE_FILE = Path(__file__)
    SOURCE_DIR = SOURCE_FILE.parent
    ROOT_DIR = SOURCE_DIR.parent.resolve()
    DATA_DIR = ROOT_DIR / "data"

    theta = np.load(DATA_DIR / "posterior_full_field__"
                    "{}N_{}ms_{}gpa_{}se.npy".format(
                        N, map_size, gals_per_arcmin2, sigma_e))

    m_data = np.load(DATA_DIR / "m_data__"
                     "{}N_{}ms_{}gpa_{}se.npy".format(
                         N, map_size, gals_per_arcmin2, sigma_e))

    return theta, m_data
