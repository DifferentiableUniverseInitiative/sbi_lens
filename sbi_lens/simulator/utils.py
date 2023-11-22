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

from sbi_lens.simulator.LogNormal_field import lensingLogNormal

tfp = tfp.substrates.jax
tfd = tfp.distributions

class ForwardModelMassMap:
    def __init__(self, config, model_type, lognormal_shifts, with_noise):
        self.config = config

        self.simulator_model = partial(
            lensingLogNormal,
            N=self.config.N,
            map_size=self.config.map_size,
            gal_per_arcmin2=self.config.gals_per_arcmin2,
            sigma_e=self.config.sigma_e,
            nbins=self.config.nbins,
            a=self.config.a,
            b=self.config.b,
            z0=self.config.z0,
            model_type=model_type,
            lognormal_shifts=lognormal_shifts,
            with_noise=with_noise,
        )

    def get_samples_and_scores(
        self,
        key,
        batch_size=64,
        score_type="density",
        thetas=None,
        with_noise=True,
    ):
        """Handling function sampling and computing the score from the model.

        Parameters
        ----------
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

        params_name = self.config.params_name

        def log_prob_fn(theta, key):
            cond_model = seed(self.simulator_model, key)
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
                model_trace = trace(seed(self.simulator_model, key)).get_trace()
                thetas = jnp.stack(
                    [model_trace[name]["value"] for name in params_name], axis=-1
                )
                return thetas

            thetas = get_params(keys)

        return jax.vmap(jax.value_and_grad(log_prob_fn, has_aux=True))(thetas, keys)

    def get_reference_posterior(self):
        SOURCE_FILE = Path(__file__)
        SOURCE_DIR = SOURCE_FILE.parent
        ROOT_DIR = SOURCE_DIR.parent.resolve()
        DATA_DIR = ROOT_DIR / "data"

        theta = np.load(
            DATA_DIR / "posterior_full_field__"
            "{}N_{}ms_{}gpa_{}se.npy".format(
                self.config.N,
                self.config.map_size,
                self.config.gals_per_arcmin2,
                self.config.sigma_e,
            )
        )

        m_data = np.load(
            DATA_DIR / "m_data__"
            "{}N_{}ms_{}gpa_{}se.npy".format(
                self.config.N,
                self.config.map_size,
                self.config.gals_per_arcmin2,
                self.config.sigma_e,
            )
        )

        return theta, m_data, self.config.truth

    def run_mcmc(
        self,
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

        Parameters
        ----------
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

        def config(x):
            if type(x["fn"]) is dist.TransformedDistribution:
                return TransformReparam()
            elif (
                type(x["fn"]) is dist.Normal or type(x["fn"]) is dist.TruncatedNormal
            ) and ("decentered" not in x["name"]):
                return LocScaleReparam(centered=0)
            else:
                return None

        observed_model = condition(self.simulator_model, {"y": m_data})
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
        mcmc.run(key, extra_fields=("num_steps",))
        samples_ = mcmc.get_samples()
        nb_of_log_prob_evaluation = mcmc.get_extra_fields()["num_steps"].sum()
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
            mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=("num_steps",))
            samples_ = mcmc.get_samples()
            nb_of_log_prob_evaluation += mcmc.get_extra_fields()["num_steps"].sum()
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

        return jnp.array(samples_ff_store).reshape([-1, 6]), nb_of_log_prob_evaluation


class UtilsFunctions:
    def __init__(self, config):
        self.config = config

    def compute_power_spectrum_theory(
        self,
        cosmo_params,
        ell,
        with_noise=True,
    ):
        """Compute theoric power spectrum given given cosmological
        parameters, redshift distribution and multipole bin edges

        Parameters
        ----------
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

        cell_theory = jc.angular_cl.angular_cl(cosmo, ell, [self.config.tracer])
        cell_noise = jc.angular_cl.noise_cl(ell, [self.config.tracer])

        if with_noise:
            Cl_theo = cell_theory, cell_noise
        else:
            Cl_theo = cell_theory

        return Cl_theo

    def get_covariance_matrix(self, cosmo_params_fiducial, ell):
        cl_signal, cl_noise = self.compute_power_spectrum_theory(
            cosmo_params=cosmo_params_fiducial, ell=ell, with_noise=True
        )

        covariance = jc.angular_cl.gaussian_cl_covariance(
            ell,
            [self.config.tracer],
            cl_signal,
            cl_noise,
            f_sky=self.config.f_sky,
            sparse=True,
        )

        return covariance

    def compute_power_spectrum_mass_map(self, mass_map):
        """Compute the power spectrum of the convergence map

        Parameters
        ----------
        mass_map : Array (N,N, nbins)
            Lensing convergence maps

        Returns
        -------
            Power spectrum and ell
        """

        l_edges_kmap = np.arange(100.0, 5000.0, 50.0)

        ell = ConvergenceMap(
            mass_map[:, :, 0], angle=self.config.map_size * u.deg
        ).cross(
            ConvergenceMap(mass_map[:, :, 0], angle=self.config.map_size * u.deg),
            l_edges=l_edges_kmap,
        )[
            0
        ]

        # power spectrum of the map
        ps = []

        for i, j in itertools.combinations_with_replacement(
            range(self.config.nbins), 2
        ):
            ps_ij = ConvergenceMap(
                mass_map[:, :, i], angle=self.config.map_size * u.deg
            ).cross(
                ConvergenceMap(mass_map[:, :, j], angle=self.config.map_size * u.deg),
                l_edges=l_edges_kmap,
            )[
                1
            ]

            ps.append(ps_ij)

        return np.array(ps), ell


class ForwardModelPowerSpectrum:
    def __init__(self, config):
        self.config = config
        self.utils_fun = UtilsFunctions(self.config)

    def analytic_model_power_spectrum(self, ell, covariance):
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

        # Calculate power spectrum
        cl_noise = jc.angular_cl.noise_cl(ell, [self.config.tracer]).flatten()
        cl, C = jc.angular_cl.gaussian_cl_covariance_and_mean(
            cosmo, ell, [self.config.tracer], f_sky=self.config.f_sky, sparse=True
        )

        if covariance is not None:
            C = covariance

        # Compute precision matrix
        P = jc.sparse.to_dense(jc.sparse.inv(C))
        C = jc.sparse.to_dense(C)

        cl = sample(
            "cl",
            dist.MultivariateNormal(
                cl + cl_noise, precision_matrix=P, covariance_matrix=C
            ),
        )

        return cl

    def get_reference_posterior(self):
        SOURCE_FILE = Path(__file__)
        SOURCE_DIR = SOURCE_FILE.parent
        ROOT_DIR = SOURCE_DIR.parent.resolve()
        DATA_DIR = ROOT_DIR / "data"

        theta = np.load(
            DATA_DIR / "posterior_power_spectrum__"
            "{}N_{}ms_{}gpa_{}se.npy".format(
                self.config.N,
                self.config.map_size,
                self.config.gals_per_arcmin2,
                self.config.sigma_e,
            )
        )

        m_data = np.load(
            DATA_DIR / "m_data__"
            "{}N_{}ms_{}gpa_{}se.npy".format(
                self.config.N,
                self.config.map_size,
                self.config.gals_per_arcmin2,
                self.config.sigma_e,
            )
        )

        return theta, m_data, self.config.truth

    def run_mcmc(
        self,
        m_data=None,
        fixed_covariance=False,
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

        Parameters
        ----------
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

        pl_array, ell = self.utils_fun.compute_power_spectrum_mass_map(m_data)

        if fixed_covariance:
            covariance = self.utils_fun.get_covariance_matrix(self.config.truth, ell)
        else:
            covariance = None

        cl_obs = np.stack(pl_array)

        model_lensingPS = partial(
            self.analytic_model_power_spectrum, ell=ell, covariance=covariance
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

        mcmc.run(key, extra_fields=("num_steps",))
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

        nb_of_log_prob_evaluation = mcmc.get_extra_fields()["num_steps"].sum()

        return samples, nb_of_log_prob_evaluation
