from pathlib import Path
import numpy as np
import numpyro
from numpyro.handlers import seed, trace, condition, reparam
from numpyro.infer.reparam import LocScaleReparam
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import lenstools as lt
import astropy.units as u
import tensorflow_probability as tfp
import itertools
from sbi_lens.simulator.redshift import subdivide

tfp = tfp.substrates.jax
tfd = tfp.distributions

SOURCE_FILE = Path(__file__)
SOURCE_DIR = SOURCE_FILE.parent
ROOT_DIR = SOURCE_DIR.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
a = 2
b = 0.68
z0 = 0.11
nbins = 5
#nz = jc.redshift.smail_nz(a, b, z0, gals_per_arcmin2=27)
#nz_bins = subdivide(nz, nbins=nbins)
#tracer = jc.probes.WeakLensing(nz_bins, sigma_e=0.26)


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

  def log_prob_fn(theta, key):
    cond_model = condition(model, {'omega_c': theta[0], 'sigma_8': theta[1]})
    cond_model = seed(cond_model, key)
    model_trace = trace(cond_model).get_trace()
    sample = {
        'theta':
        jnp.stack(
            [model_trace['omega_c']['value'], model_trace['sigma_8']['value']],
            axis=-1),
        'y':
        model_trace['y']['value']
    }

    if score_type == 'density':
      logp = model_trace['omega_c']['fn'].log_prob(
          model_trace['omega_c']['value'])
      logp += model_trace['sigma_8']['fn'].log_prob(
          model_trace['sigma_8']['value'])
    elif score_type == 'conditional':
      logp = 0

    if with_noise:
      logp += model_trace['y']['fn'].log_prob(
          jax.lax.stop_gradient(model_trace['y']['value'])).sum()
    logp += model_trace['z']['fn'].log_prob(model_trace['z']['value']).sum()

    return logp, sample

  # Split the key by batch
  keys = jax.random.split(key, batch_size)

  # Sample theta from the model
  if thetas is None:
    omega_c = jax.vmap(
        lambda k: trace(seed(model, k)).get_trace()['omega_c']['value'])(keys)
    sigma_8 = jax.vmap(
        lambda k: trace(seed(model, k)).get_trace()['sigma_8']['value'])(keys)
    thetas = jnp.stack([omega_c, sigma_8], axis=-1)
  res = jax.vmap(jax.value_and_grad(log_prob_fn, has_aux=True))(thetas, keys)

  return res


def get_reference_sample_posterior_power_spectrum(
    Omega_c=[0.2664, 0.2],
    Omega_b=[0.0492, 0.006],
    sigma8=[0.831, 0.14],
    h0=[0.6727, 0.063],
    ns=[0.9645, 0.08],
    w0=[-1.0, 0.8],
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
    key=None,
):
  """ Posterior p(theta|x=m_data) from power spectrum analysis.

    Parameters
    ----------
    Omega_c: [float, float]
        Fiducial and prior value of the cold matter density fraction.
    Omega_b: [float, float]
        Fiducial and prior value of the  baryonic matter density fraction.
    sigma8: [float, float]
        Fiducial and prior value of the variance of matter density perturbations at an 8 Mpc/h scale.
    h: [float, float]
      Fiducial and prior value of the Hubble constant divided by 100 km/s/Mpc; unitless.
    ns:[float, float]
    Fiducial and prior value of the primordial scalar perturbation spectral
        index.
    w0:[float, float]
        Fiducial and prior value of the first order term of dark energy equation.

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
    key : PRNG key
        only needed if run_mcmc=True, by default None
        if run_mcmc=True key can not be None

    Returns
    -------
    Array (num_results,2)
        MCMC chains corresponding to p(theta|x=m_data)
    """

  if run_mcmc:
    cosmo = jc.Planck15(Omega_c=Omega_c[0],
                        Omega_b=Omega_b[0],
                        sigma8=sigma8[0],
                        h=h0[0],
                        n_s=ns[0],
                        w0=w0[0])
    nz = jc.redshift.smail_nz(a, b, z0, gals_per_arcmin2=gals_per_arcmin2)
    nz_bins = subdivide(nz, nbins=nbins)
    tracer = jc.probes.WeakLensing(nz_bins, sigma_e=sigma_e)
    f_sky = map_size**2 / 41_253
    l_edges = np.arange(100.0, 5000.0, 50.0)
    l2 = lt.ConvergenceMap(m_data[0],
                           map_size * u.deg).powerSpectrum(l_edges)[0]
    pl_array = []
    for i, j in itertools.combinations_with_replacement(range(nbins), 2):
      pi = lt.ConvergenceMap(m_data[i], angle=map_size * u.deg).cross(
          lt.ConvergenceMap(m_data[j], angle=map_size * u.deg),
          l_edges=l_edges)[1]
      pl_array.append(pi)
    Pl2 = np.stack(pl_array)
    cell_noise = jc.angular_cl.noise_cl(l2, [tracer])
    _, C = jc.angular_cl.gaussian_cl_covariance_and_mean(cosmo,
                                                         l2, [tracer],
                                                         f_sky=f_sky,
                                                         sparse=True)

    #@jax.jit
    @jax.vmap
    def log_prob_fn(params):
      cosmo = jc.Planck15(
          Omega_c=params[0] * Omega_c[1] + Omega_c[0],
          Omega_b=params[1] * Omega_b[1] + Omega_b[0],
          sigma8=params[2] * sigma8[1] + sigma8[0],
          h=params[3] * h0[1] + h0[0],
          n_s=params[4] * ns[1] + ns[0],
          w0=params[5] * w0[1] + w0[0],
      )
      cell = jc.angular_cl.angular_cl(cosmo, l2, [tracer])
      prior = tfd.MultivariateNormalDiag(loc=jnp.zeros(6),
                                         scale_identity_multiplier=1.)
      likelihood_log_prob = jc.likelihood.gaussian_log_likelihood(
          Pl2 - cell_noise, cell, C, include_logdet=False)
      logp = prior.log_prob(params) + likelihood_log_prob
      return logp

    # Initialize the HMC transition kernel.
    num_burnin_steps = int(1e2)
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=log_prob_fn,
                                       num_leapfrog_steps=3,
                                       step_size=0.07),
        num_adaptation_steps=int(num_burnin_steps * 0.8))

    # Run the chain (with burn-in).
    nb_parallel = 10
    num_results = num_results // nb_parallel
    samples, is_accepted = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=np.random.randn(nb_parallel, 6),
        kernel=adaptive_hmc,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
        seed=key)

    theta = samples[is_accepted] * jnp.array(
        [Omega_c, Omega_b, sigma8, h0, ns, w0])[:, 0] + jnp.array(
            [Omega_c, Omega_b, sigma8, h0, ns, w0])[:, 1]
    return theta

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

    truth = jnp.array([Omega_c, Omega_b, sigma8, h0, ns, w0])[:, 0]
    return theta, m_data, truth


def get_reference_sample_posterior_full_field(
    Omega_c=0.2664,
    Omega_b=0.0492,
    sigma8=0.83,
    h0=0.6727,
    ns=0.9645,
    w0=-1.0,
    run_mcmc=False,
    N=256,
    map_size=10,
    gals_per_arcmin2=27,
    sigma_e=0.26,
    model=None,
    m_data=None,
    num_results=None,
    key=None,
):
  """ Full field posterior p(theta|x=m_data).

    Parameters
    ----------
    Omega_c: float
        Fiducial value of the cold matter density fraction.
    sigma8: float
        Fiducial value of the variance of matter density perturbations at an 8 Mpc/h scale.
    Omega_c: float
        Fiducial value of the cold matter density fraction.
    Omega_b: float
        Fiducial value of the  baryonic matter density fraction.
    sigma8: float
        Fiducial value of the variance of matter density perturbations at an 8 Mpc/h scale.
    h: float
      Fiducial value of the Hubble constant divided by 100 km/s/Mpc; unitless.
    ns: float
    Fiducial value of the primordial scalar perturbation spectral
        index.
    w0:[float, float]
        Fiducial value of the first order term of dark energy equation.
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
      if x['name'] == 'omega_c' and ('decentered' not in x['name']):
        return LocScaleReparam(centered=0)
      elif x['name'] == 'omega_b' and ('decentered' not in x['name']):
        return LocScaleReparam(centered=0)
      elif x['name'] == 'sigma_8' and ('decentered' not in x['name']):
        return LocScaleReparam(centered=0)
      elif x['name'] == 'h_0' and ('decentered' not in x['name']):
        return LocScaleReparam(centered=0)
      elif x['name'] == 'n_s' and ('decentered' not in x['name']):
        return LocScaleReparam(centered=0)
      elif x['name'] == 'w_0' and ('decentered' not in x['name']):
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
                              num_warmup=100,
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

    truth = jnp.array([Omega_c, Omega_b, sigma8, h0, ns, w0])
    return theta, m_data, truth
