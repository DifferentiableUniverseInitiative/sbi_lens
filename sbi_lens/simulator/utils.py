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
tfp = tfp.substrates.jax
tfd = tfp.distributions


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
        cond_model = condition(model, {'omega_c': theta[0], 
                                       'sigma_8': theta[1]})
        cond_model = seed(cond_model, key)
        model_trace = trace(cond_model).get_trace()
        sample = {'theta': jnp.stack(
          [model_trace['omega_c']['value'], model_trace['sigma_8']['value']],
          axis=-1
          ),
                  'y': model_trace['y']['value']}

        if score_type == 'density':
            logp = model_trace['omega_c']['fn'].log_prob(
              model_trace['omega_c']['value']
            )
            logp += model_trace['sigma_8']['fn'].log_prob(
              model_trace['sigma_8']['value']
            )
        elif score_type == 'conditional':
            logp = 0

        if with_noise:
            logp += model_trace['y']['fn'].log_prob(
              jax.lax.stop_gradient(model_trace['y']['value'])
            ).sum()
        logp += model_trace['z']['fn'].log_prob(
          model_trace['z']['value']
        ).sum()

        return logp, sample

    # Split the key by batch
    keys = jax.random.split(key, batch_size)

    # Sample theta from the model
    if thetas is None:
        thetas = jax.vmap(
          lambda k: trace(seed(model, k)).get_trace()['theta']['value']
        )(keys)
    res = jax.vmap(
      jax.value_and_grad(log_prob_fn, has_aux=True)
    )(thetas, keys)

    return res


def get_reference_sample_posterior_power_spectrum(
  m_data,
  gals_per_arcmin2,
  sigma_e,
  num_results,
  key,
  run_mcmc=True,
  N=None,
  map_size=None,
):

    """ Posterior p(theta|x=m_data) from power spectrum analysis.

    Parameters
    ----------
    m_data : Array (N,N)
        Lensing convergence map.
    gals_per_arcmin2 : int
        Number of galaxies per arcmin
    sigma_e : float
        Dispersion of the ellipticity distribution
    num_results : int
        Number of samples
    key : PRNG key
    run_mcmc : bool, optional
        if True the MCMC will be run.
        if False pre samples chains are returned according to
        gals_per_arcmin2, sigma_e, N, map_size,
        by default True
    N : int, optional
        Number of pixels on the map., by default None
    map_size : int, optional
        The total angular size area is given by map_size x map_size,
        by default None

    Returns
    -------
    Array (num_results,2)
        MCMC chains corresponding to p(theta|x=m_data)
    """

    if run_mcmc:

        cosmo = jc.Planck15(Omega_c=0.3, sigma8=0.8)
        pz = jc.redshift.smail_nz(
          0.5,
          2.,
          1.0,
          gals_per_arcmin2=gals_per_arcmin2
        )
        tracer = jc.probes.WeakLensing([pz], sigma_e=sigma_e)
        f_sky = 5**2/41_253
        kmap_lt = lt.ConvergenceMap(m_data, 5 * u.deg)
        l_edges = np.arange(100.0, 5000.0, 50.0)
        l2, Pl2 = kmap_lt.powerSpectrum(l_edges)
        cell_noise = jc.angular_cl.noise_cl(l2, [tracer])[0]
        _, C = jc.angular_cl.gaussian_cl_covariance_and_mean(
          cosmo,
          l2,
          [tracer],
          f_sky=f_sky)

        @jax.jit
        @jax.vmap
        def log_prob_fn(params):
            cosmo = jc.Planck15(Omega_c=params[0] * 0.05 + 0.3,
                                sigma8=params[1] * 0.05 + 0.8)
            cell = jc.angular_cl.angular_cl(cosmo, l2, [tracer])[0]
            prior = tfd.MultivariateNormalDiag(
              loc=jnp.zeros(2),
              scale_identity_multiplier=1.
            )
            likelihood = tfd.MultivariateNormalDiag(
              cell,
              scale_diag=jnp.sqrt(np.diag(C))
            )
            logp = prior.log_prob(params) + likelihood.log_prob(Pl2-cell_noise)
            return logp

        # Initialize the HMC transition kernel.
        num_burnin_steps = int(1e2)
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=log_prob_fn,
                num_leapfrog_steps=3,
                step_size=0.07),
            num_adaptation_steps=int(num_burnin_steps * 0.8))

        # Run the chain (with burn-in).
        nb_parallel = 10
        num_results = num_results // nb_parallel
        samples, is_accepted = tfp.mcmc.sample_chain(
              num_results=num_results,
              num_burnin_steps=num_burnin_steps,
              current_state=np.random.randn(nb_parallel, 2),
              kernel=adaptive_hmc,
              trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
              seed=key)

        theta = samples[is_accepted] * 0.05 + jnp.array([0.3, 0.8])
        inds = np.random.randint(0, int(num_results), len(theta))
        theta = theta[inds]

        return theta

    else:
        SOURCE_FILE = Path(__file__)
        SOURCE_DIR = SOURCE_FILE.parent
        ROOT_DIR = SOURCE_DIR.parent.resolve()
        DATA_DIR = ROOT_DIR / "data"

        theta = np.load(
          DATA_DIR / "posterior_power_spectrum__"
          "{}N_{}ms_{}gpa_{}se".format(N, map_size, gals_per_arcmin2, sigma_e)
        )

        return theta, m_data


def get_reference_sample_posterior_full_field(
  model,
  m_data,
  num_results,
  key,
  run_mcmc=True,
  N=None,
  map_size=None,
  sigma_e=None,
  gals_per_arcmin2=None
):

    """ Full field posterior p(theta|x=m_data).

    Parameters
    ----------
    model : numpyro model
    m_data : Array (N,N)
        Lensing convergence map.
    num_results : int
        Number of samples
    key : PRNG key
    run_mcmc : bool, optional
        if True the MCMC will be run.
        if False pre samples chains are returned according to
        gals_per_arcmin2, sigma_e, N, map_size,
        by default True
    N : int, optional
        Number of pixels on the map., by default None
    map_size : int, optional
        The total angular size area is given by map_size x map_size,
        by default None
    sigma_e : float, optional
        Dispersion of the ellipticity distribution, by default None
    gals_per_arcmin2 : int, optional
        Number of galaxies per arcmin, by default None

    Returns
    -------
    Array (num_results,2)
        MCMC chains corresponding to p(theta|x=m_data)
    """

    if run_mcmc:

        def config(x):
            if x['name'] == 'omega_c' and ('decentered' not in x['name']):
                return LocScaleReparam(centered=0)
            elif x['name'] == 'sigma_8' and ('decentered' not in x['name']):
                return LocScaleReparam(centered=0)
            else:
                return None

        observed_model = condition(model, {'y': m_data})
        observed_model_reparam = reparam(observed_model, config=config)
        nuts_kernel = numpyro.infer.NUTS(
          observed_model_reparam,
          init_strategy=numpyro.infer.init_to_median,
          max_tree_depth=6,
          step_size=0.02
        )
        mcmc = numpyro.infer.MCMC(
          nuts_kernel,
          num_warmup=100,
          num_samples=num_results,
          progress_bar=True
        )
        mcmc.run(key)
        samples = mcmc.get_samples()

        return jnp.stack([samples['omega_c'], samples['sigma_8']], axis=-1)

    else:
        SOURCE_FILE = Path(__file__)
        SOURCE_DIR = SOURCE_FILE.parent
        ROOT_DIR = SOURCE_DIR.parent.resolve()
        DATA_DIR = ROOT_DIR / "data"

        theta = np.load(
          DATA_DIR / "posterior_full_field__"
          "{}N_{}ms_{}gpa_{}se".format(N, map_size, gals_per_arcmin2, sigma_e)
        )

        return theta, m_data
