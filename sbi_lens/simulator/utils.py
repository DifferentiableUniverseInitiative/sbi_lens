import itertools
from functools import partial
from pathlib import Path

import astropy.units as u
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
import numpyro
import numpyro.distributions as dist
import tensorflow_probability as tfp
from lenstools import ConvergenceMap
from numpyro import sample
from numpyro.handlers import condition, reparam, seed, trace
from numpyro.infer.reparam import LocScaleReparam, TransformReparam

from sbi_lens.simulator.redshift import subdivide

tfp = tfp.substrates.jax
tfd = tfp.distributions

np.complex = complex
np.float = float

SOURCE_FILE = Path(__file__)
SOURCE_DIR = SOURCE_FILE.parent
ROOT_DIR = SOURCE_DIR.parent.resolve()
DATA_DIR = ROOT_DIR / "data"


def get_samples_and_scores(
    model,
    key,
    batch_size=64,
    score_type="density",
    thetas=None,
    with_noise=True,
):
    """Handling function sampling and computing the score from the model.

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

    params_name = ["omega_c", "omega_b", "sigma_8", "h_0", "n_s", "w_0"]

    def log_prob_fn(theta, key):
        cond_model = seed(model, key)
        cond_model = condition(
            cond_model,
            {
                "omega_c": theta[0],
                "omega_b": theta[1],
                "sigma_8": theta[2],
                "h_0": theta[3],
                "n_s": theta[4],
                "w_0": theta[5],
            },
        )
        model_trace = trace(cond_model).get_trace()
        sample = {
            "theta": jnp.stack(
                [model_trace[name]["value"] for name in params_name], axis=-1
            ),
            "y": model_trace["y"]["value"],
        }

        if score_type == "density":
            logp = 0
            for name in params_name:
                logp += model_trace[name]["fn"].log_prob(model_trace[name]["value"])
        elif score_type == "conditional":
            logp = 0

        if with_noise:
            logp += (
                model_trace["y"]["fn"]
                .log_prob(jax.lax.stop_gradient(model_trace["y"]["value"]))
                .sum()
            )
        logp += model_trace["z"]["fn"].log_prob(model_trace["z"]["value"]).sum()

        return logp, sample

    # Split the key by batch
    keys = jax.random.split(key, batch_size)

    # Sample theta from the model
    if thetas is None:

        @jax.vmap
        def get_params(key):
            model_trace = trace(seed(model, key)).get_trace()
            thetas = jnp.stack(
                [model_trace[name]["value"] for name in params_name], axis=-1
            )
            return thetas

        thetas = get_params(keys)

    return jax.vmap(jax.value_and_grad(log_prob_fn, has_aux=True))(thetas, keys)


def _lensingPS(map_size, sigma_e, a, b, z0, gals_per_arcmin2, ell, nbins):
    # Field parameters
    f_sky = map_size**2 / 41_253
    nz = jc.redshift.smail_nz(a, b, z0, gals_per_arcmin2=gals_per_arcmin2)
    nz_bins = subdivide(nz, nbins=nbins, zphot_sigma=0.05)

    # Cosmological parameters
    omega_c = sample("omega_c", dist.TruncatedNormal(0.2664, 0.2, low=0))
    omega_b = sample("omega_b", dist.Normal(0.0492, 0.006))
    sigma_8 = sample("sigma_8", dist.Normal(0.831, 0.14))
    h_0 = sample("h_0", dist.Normal(0.6727, 0.063))
    n_s = sample("n_s", dist.Normal(0.9645, 0.08))
    w_0 = sample("w_0", dist.TruncatedNormal(-1.0, 0.9, low=-2.0, high=-0.3))

    cosmo = jc.Planck15(
        Omega_c=omega_c, Omega_b=omega_b, h=h_0, n_s=n_s, sigma8=sigma_8, w0=w_0
    )

    tracer = jc.probes.WeakLensing(nz_bins, sigma_e=sigma_e)

    # Calculate power spectrum
    cl_noise = jc.angular_cl.noise_cl(ell, [tracer]).flatten()
    cl, C = jc.angular_cl.gaussian_cl_covariance_and_mean(
        cosmo, ell, [tracer], f_sky=f_sky, sparse=True
    )

    # Compute precision matrix
    P = jc.sparse.to_dense(jc.sparse.inv(jax.lax.stop_gradient(C)))
    C = jc.sparse.to_dense(C)

    cl = sample(
        "cl",
        dist.MultivariateNormal(cl + cl_noise, precision_matrix=P, covariance_matrix=C),
    )

    return cl


def compute_power_spectrum_theory(
    nbins,
    sigma_e,
    a,
    b,
    z0,
    gals_per_arcmin2,
    cosmo_params,
    ell,
    with_noise=True,
):
    """Compute theoric power spectrum given given cosmological
    parameters, redshift distribution and multipole bin edges

    Parameters
    ----------
    n_bins: int
        Number of redshift bins
    sigma_e : float
        Dispersion of the ellipticity distribution
    a : float
        Parameter defining the redshift distribution
    b : float
        Parameter defining the redshift distribution
    z0 : float
        Parameter defining the redshift distribution
    gals_per_arcmin2 : int
        Number of galaxies per arcmin
    cosmo_params : Array (6)
        cosmological parameters in the following order:
        (omega_c, omega_b, sigma_8, h_0, n_s, w_0)
    ell : Array
        Multipole bin edges
    with_noise : bool, optional
        True if there is noise in the mass_map, by default True
    Returns
    -------
        Theoric power spectrum
    """

    omega_c, omega_b, sigma_8, h_0, n_s, w_0 = cosmo_params

    # power spectrum from theory
    cosmo = jc.Planck15(
        Omega_c=omega_c,
        Omega_b=omega_b,
        h=h_0,
        n_s=n_s,
        sigma8=sigma_8,
        w0=w_0,
    )

    nz = jc.redshift.smail_nz(a, b, z0, gals_per_arcmin2=gals_per_arcmin2)
    nz_bins = subdivide(nz, nbins=nbins, zphot_sigma=0.05)
    tracer = jc.probes.WeakLensing(nz_bins, sigma_e=sigma_e)

    cell_theory = jc.angular_cl.angular_cl(cosmo, ell, [tracer])
    cell_noise = jc.angular_cl.noise_cl(ell, [tracer])

    if with_noise:
        Cl_theo = cell_theory + cell_noise
    else:
        Cl_theo = cell_theory

    return Cl_theo


def compute_power_spectrum_mass_map(nbins, map_size, mass_map):
    """Compute the power spectrum of the convergence map

    Parameters
    ----------
    n_bins: int
        Number of redshift bins
    map_size : int
        The total angular size area is given by map_size x map_size
    mass_map : Array (N,N, nbins)
        Lensing convergence maps

    Returns
    -------
        Power spectrum and ell
    """

    l_edges_kmap = np.arange(100.0, 5000.0, 50.0)

    ell = ConvergenceMap(mass_map[:, :, 0], angle=map_size * u.deg).cross(
        ConvergenceMap(mass_map[:, :, 0], angle=map_size * u.deg),
        l_edges=l_edges_kmap,
    )[0]

    # power spectrum of the map
    ps = []

    for i, j in itertools.combinations_with_replacement(range(nbins), 2):
        ps_ij = ConvergenceMap(mass_map[:, :, i], angle=map_size * u.deg).cross(
            ConvergenceMap(mass_map[:, :, j], angle=map_size * u.deg),
            l_edges=l_edges_kmap,
        )[1]

        ps.append(ps_ij)

    return np.array(ps), ell


def gaussian_log_likelihood(
    cosmo_params, mass_map, nbins, map_size, sigma_e, a, b, z0, gals_per_arcmin2
):
    """Compute the gaussian likelihood log probrobability

    Parameters
    ----------
    cosmo_params : Array
        cosmological parameters in the following order:
        (omega_c, omega_b, sigma_8, h_0, n_s, w_0)
    mass_map : Array (N,N, nbins)
        Lensing convergence maps
    n_bins: int
        Number of redshift bins
    map_size : int
        The total angular size area is given by map_size x map_size
    sigma_e : float
        Dispersion of the ellipticity distribution
    a : float
        Parameter defining the redshift distribution
    b : float
        Parameter defining the redshift distribution
    z0 : float
        Parameter defining the redshift distribution
    gals_per_arcmin2 : int
        Number of galaxies per arcmin
    Returns
    -------
    log p(mass_map | cosmo_params)
    """

    pl_array, ell = compute_power_spectrum_mass_map(nbins, map_size, mass_map)

    cl_obs = np.stack(pl_array)

    model_lensingPS = partial(
        _lensingPS,
        map_size=map_size,
        sigma_e=sigma_e,
        a=a,
        b=b,
        z0=z0,
        gals_per_arcmin2=gals_per_arcmin2,
        ell=ell,
        nbins=5,
    )

    # Now we condition the model on obervations
    cond_model = condition(
        model_lensingPS,
        {
            "cl": cl_obs.flatten(),
            "omega_c": cosmo_params[0],
            "omega_b": cosmo_params[1],
            "sigma_8": cosmo_params[2],
            "h_0": cosmo_params[3],
            "n_s": cosmo_params[4],
            "w_0": cosmo_params[5],
        },
    )

    model_trace = trace(cond_model).get_trace()
    log_prob = model_trace["cl"]["fn"].log_prob(model_trace["cl"]["value"])

    return log_prob


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
    num_results=500,
    num_warmup=200,
    num_chains=1,
    chain_method="parallel",
    max_tree_depth=6,
    step_size=1e-2,
    init_strat=numpyro.infer.init_to_value,
    key=None,
):
    """Posterior p(theta|x=m_data) from power spectrum analysis.
      Note: pre samples chains correspond to the following fiducial parameters:
      (omega_c, omega_b, sigma_8, h_0, n_s, w_0)
       = (0.2664, 0.0492, 0.831, 0.6727, 0.9645, -1.0)

    Parameters
    ----------
    run_mcmc : bool, optional
        if True the MCMC (No U-Turn Sampler) will be run,
        if False pre sampled chains are returned according to
        gals_per_arcmin2, sigma_e, N, map_size,
        by default False
    N : int, optional
        Number of pixels on the map., by default 256
    map_size : int, optional
        The total angular size area is given by map_size x map_size,
        by default 10
    gals_per_arcmin2 : int
        Number of galaxies per arcmin, by default 27
    sigma_e : float
        Dispersion of the ellipticity distribution, by default 0.26
    n_bins: int
        Number of redshift bins, by defautlt 5
    a : float
        Parameter defining the redshift distribution, by defautlt 2
    b : float
        Parameter defining the redshift distribution, by defautlt 0.68
    z0 : float
        Parameter defining the redshift distribution, , by defautlt 0.11
    m_data : Array (N,N)
        Lensing convergence map, by default None
        if run_mcmc=True m_data can not be None
    num_results : int
        Number of samples, by default 500
    num_warmup : int
        Number of warmup steps, by default 200
    num_chains : int
        Number of MCMC chains to run, by default 1
    chain_method : str
        'parallel', 'sequential', 'vectorized', by default 'parallel'
    max_tree_depth : int
        Max depth of the binary tree created during the doubling scheme
        of NUTS sampler, by default 6
    step_size : float
        Size of a single step, by default 1e-2
    init_strat : callable
        Sampler initialization Strategies.
        See https://num.pyro.ai/en/stable/utilities.html#init-strategy
    key : PRNG key
        Only needed if run_mcmc=True, by default None

    Returns
    -------
    Array (num_results,2)
        MCMC chains corresponding to p(theta|x=m_data)
    """

    if run_mcmc:
        pl_array, ell = compute_power_spectrum_mass_map(nbins, map_size, m_data)

        cl_obs = np.stack(pl_array)

        model_lensingPS = partial(
            _lensingPS,
            map_size=map_size,
            sigma_e=sigma_e,
            a=a,
            b=b,
            z0=z0,
            gals_per_arcmin2=gals_per_arcmin2,
            ell=ell,
            nbins=nbins,
        )

        # Now we condition the model on obervations
        observed_model = condition(model_lensingPS, {"cl": cl_obs.flatten()})

        def config(x):
            if type(x["fn"]) is dist.TransformedDistribution:
                return TransformReparam()
            elif (
                type(x["fn"]) is dist.Normal or type(x["fn"]) is dist.TruncatedNormal
            ) and ("decentered" not in x["name"]):
                return LocScaleReparam(centered=0)
            else:
                return None

        observed_model_reparam = reparam(observed_model, config=config)
        nuts_kernel = numpyro.infer.NUTS(
            model=observed_model_reparam,
            init_strategy=init_strat,
            max_tree_depth=max_tree_depth,
            step_size=step_size,
        )
        mcmc = numpyro.infer.MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_results,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=True,
        )

        mcmc.run(key)
        samples = mcmc.get_samples()
        samples = jnp.stack(
            [
                samples["omega_c"],
                samples["omega_b"],
                samples["sigma_8"],
                samples["h_0"],
                samples["n_s"],
                samples["w_0"],
            ],
            axis=-1,
        )

        return samples

    else:
        SOURCE_FILE = Path(__file__)
        SOURCE_DIR = SOURCE_FILE.parent
        ROOT_DIR = SOURCE_DIR.parent.resolve()
        DATA_DIR = ROOT_DIR / "data"

        theta = np.load(
            DATA_DIR / "posterior_power_spectrum__"
            "{}N_{}ms_{}gpa_{}se.npy".format(N, map_size, gals_per_arcmin2, sigma_e)
        )

        m_data = np.load(
            DATA_DIR / "m_data__"
            "{}N_{}ms_{}gpa_{}se.npy".format(N, map_size, gals_per_arcmin2, sigma_e)
        )

        return theta, m_data


def get_reference_sample_posterior_full_field(
    run_mcmc=False,
    N=256,
    map_size=10,
    gals_per_arcmin2=27,
    sigma_e=0.26,
    model=None,
    m_data=None,
    num_results=500,
    num_warmup=200,
    nb_loop=1,
    num_chains=1,
    chain_method="parallel",
    max_tree_depth=6,
    step_size=1e-2,
    init_strat=numpyro.infer.init_to_value,
    key=None,
):
    """Full field posterior p(theta|x=m_data).
    Note: pre samples chains correspond to the following fiducial parameters:
    (omega_c, omega_b, sigma_8, h_0, n_s, w_0)
    = (0.2664, 0.0492, 0.831, 0.6727, 0.9645, -1.0)

    Parameters
    ----------
    run_mcmc : bool, optional
        if True the MCMC (No U-Turn Sampler) will be run,
        if False pre sampled chains are returned according to
        gals_per_arcmin2, sigma_e, N, map_size,
        by default False
    N : int, optional
        Number of pixels on the map., by default 256
    map_size : int, optional
        The total angular size area is given by map_size x map_size,
        by default 10
    gals_per_arcmin2 : int
        Number of galaxies per arcmin, by default 27
    sigma_e : float
        Dispersion of the ellipticity distribution, by default 0.26
    model : numpyro model
        only needed if run_mcmc=True, by default None
        if run_mcmc=True model can not be None
    m_data : Array (N,N)
        Lensing convergence map, by default None
        if run_mcmc=True m_data can not be None
    num_results : int
        Number of samples, by default 500
    num_warmup : int
        Number of warmup steps, by default 200
    nb_loop : int
        Sequentially draw num_results samples
        (ex nb_loop=2 and num_results=100, the number of samples you
        get at the end is 200), by default 1
    num_chains : int
        Number of MCMC chains to run, by default 1
    chain_method : str
        'parallel', 'sequential', 'vectorized', by default 'parallel'
    max_tree_depth : int
        Max depth of the binary tree created during the doubling scheme
        of NUTS sampler, by default 6
    step_size : float
        Size of a single step, by default 1e-2
    init_strat : callable
        Sampler initialization Strategies.
        See https://num.pyro.ai/en/stable/utilities.html#init-strategy
    key : PRNG key
        Only needed if run_mcmc=True, by default None

    Returns
    -------
    Array (num_results,2)
        MCMC chains corresponding to p(theta|x=m_data)
    """

    if run_mcmc:

        def config(x):
            if type(x["fn"]) is dist.TransformedDistribution:
                return TransformReparam()
            elif (
                type(x["fn"]) is dist.Normal or type(x["fn"]) is dist.TruncatedNormal
            ) and ("decentered" not in x["name"]):
                return LocScaleReparam(centered=0)
            else:
                return None

        observed_model = condition(model, {"y": m_data})
        observed_model_reparam = reparam(observed_model, config=config)

        nuts_kernel = numpyro.infer.NUTS(
            model=observed_model_reparam,
            init_strategy=init_strat,
            max_tree_depth=max_tree_depth,
            step_size=step_size,
        )
        mcmc = numpyro.infer.MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_results,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=True,
        )

        samples_ff_store = []
        mcmc.run(key)
        samples_ = mcmc.get_samples()
        mcmc.post_warmup_state = mcmc.last_state

        # save only sample of interest
        samples_ = jnp.stack(
            [
                samples_["omega_c"],
                samples_["omega_b"],
                samples_["sigma_8"],
                samples_["h_0"],
                samples_["n_s"],
                samples_["w_0"],
            ],
            axis=-1,
        )
        samples_ff_store.append(samples_)

        for i in range(1, nb_loop):
            mcmc.run(mcmc.post_warmup_state.rng_key)
            samples_ = mcmc.get_samples()
            mcmc.post_warmup_state = mcmc.last_state

            # save only sample of interest
            samples_ = jnp.stack(
                [
                    samples_["omega_c"],
                    samples_["omega_b"],
                    samples_["sigma_8"],
                    samples_["h_0"],
                    samples_["n_s"],
                    samples_["w_0"],
                ],
                axis=-1,
            )
            samples_ff_store.append(samples_)
        return jnp.array(samples_ff_store).reshape([-1, 6])

    else:
        SOURCE_FILE = Path(__file__)
        SOURCE_DIR = SOURCE_FILE.parent
        ROOT_DIR = SOURCE_DIR.parent.resolve()
        DATA_DIR = ROOT_DIR / "data"

        theta = np.load(
            DATA_DIR / "posterior_full_field__"
            "{}N_{}ms_{}gpa_{}se.npy".format(N, map_size, gals_per_arcmin2, sigma_e)
        )

        m_data = np.load(
            DATA_DIR / "m_data__"
            "{}N_{}ms_{}gpa_{}se.npy".format(N, map_size, gals_per_arcmin2, sigma_e)
        )

        return theta, m_data
