from functools import partial
from pathlib import Path

import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow_datasets.core.utils import gcs_utils

from sbi_lens.simulator.LogNormal_field import lensingLogNormal
from sbi_lens.simulator.utils import get_samples_and_scores

tfp = tfp.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors

# disable internet connection
gcs_utils.gcs_dataset_info_files = lambda *args, **kwargs: None
gcs_utils.is_dataset_on_gcs = lambda *args, **kwargs: False

_CITATION = """
"""

_DESCRIPTION = """
"""


class LensingLogNormalDatasetConfig(tfds.core.BuilderConfig):
    def __init__(
        self,
        *,
        N,
        map_size,
        gal_per_arcmin2,
        sigma_e,
        nbins,
        a,
        b,
        z0,
        model_type,
        lognormal_shifts,
        proposal,
        score_type,
        with_noise,
        **kwargs,
    ):
        v1 = tfds.core.Version("0.0.1")
        super().__init__(
            description=("Log Normal lensing simulations."), version=v1, **kwargs
        )
        self.N = N
        self.map_size = map_size
        self.gal_per_arcmin2 = gal_per_arcmin2
        self.sigma_e = sigma_e
        self.nbins = nbins
        self.a = a
        self.b = b
        self.z0 = z0
        self.model_type = model_type
        self.lognormal_shifts = lognormal_shifts
        self.proposal = proposal
        self.score_type = score_type
        self.with_noise = with_noise


class LensingLogNormalDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for LensingLogNormalDataset dataset."""

    VERSION = tfds.core.Version("0.0.1")
    RELEASE_NOTES = {
        "0.0.1": "Initial release.",
    }
    BUILDER_CONFIGS = [
        LensingLogNormalDatasetConfig(
            name="year_10_with_noise_score_density",
            N=256,
            map_size=10,
            gal_per_arcmin2=27,
            sigma_e=0.26,
            nbins=5,
            a=2,
            b=0.68,
            z0=0.11,
            model_type="lognormal",
            lognormal_shifts="LSSTY10",
            proposal=False,
            score_type="density",
            with_noise=True,
        ),
        LensingLogNormalDatasetConfig(
            name="year_10_without_noise_score_density",
            N=256,
            map_size=10,
            gal_per_arcmin2=27,
            sigma_e=0.26,
            nbins=5,
            a=2,
            b=0.68,
            z0=0.11,
            model_type="lognormal",
            lognormal_shifts="LSSTY10",
            proposal=False,
            score_type="density",
            with_noise=False,
        ),
        LensingLogNormalDatasetConfig(
            name="year_10_with_noise_score_density_proposal",
            N=256,
            map_size=10,
            gal_per_arcmin2=27,
            sigma_e=0.26,
            nbins=5,
            a=2,
            b=0.68,
            z0=0.11,
            model_type="lognormal",
            lognormal_shifts="LSSTY10",
            proposal=True,
            score_type="density",
            with_noise=True,
        ),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "simulation": tfds.features.Tensor(
                        shape=[
                            self.builder_config.N,
                            self.builder_config.N,
                            self.builder_config.nbins,
                        ],
                        dtype=tf.float32,
                    ),
                    "theta": tfds.features.Tensor(shape=[6], dtype=tf.float32),
                    "score": tfds.features.Tensor(shape=[6], dtype=tf.float32),
                }
            ),
            supervised_keys=None,
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN, gen_kwargs={"size": 100000}
            ),
        ]

    def _generate_examples(self, size):
        """Yields examples."""
        bs = 20
        if self.builder_config.proposal is True:
            SOURCE_FILE = Path(__file__)
            SOURCE_DIR = SOURCE_FILE.parent
            ROOT_DIR = SOURCE_DIR.parent.resolve()
            DATA_DIR = ROOT_DIR / "data"

            thetas = np.load(
                DATA_DIR / "posterior_power_spectrum__"
                "{}N_{}ms_{}gpa_{}se.npy".format(
                    self.builder_config.N,
                    self.builder_config.map_size,
                    self.builder_config.gal_per_arcmin2,
                    self.builder_config.sigma_e,
                )
            )
            # 'thinning'
            nb_sample_min_to_keep = 100_000
            inds = jax.random.randint(
                jax.random.PRNGKey(42), (nb_sample_min_to_keep,), 0, thetas.shape[0]
            )
            thetas = thetas[inds]

            size_thetas = len(thetas)
            thetas = thetas.reshape([-1, bs, 6])
        else:
            size_thetas = size
            thetas = np.array([None]).repeat(size // bs)

        if size > size_thetas:
            size = size_thetas

        model = partial(
            lensingLogNormal,
            self.builder_config.N,
            self.builder_config.map_size,
            self.builder_config.gal_per_arcmin2,
            self.builder_config.sigma_e,
            self.builder_config.nbins,
            self.builder_config.a,
            self.builder_config.b,
            self.builder_config.z0,
            self.builder_config.model_type,
            self.builder_config.lognormal_shifts,
            self.builder_config.with_noise,
        )

        @jax.jit
        def get_batch(key, thetas):
            (_, samples), scores = get_samples_and_scores(
                model=model,
                key=key,
                batch_size=bs,
                score_type=self.builder_config.score_type,
                thetas=thetas,
                with_noise=self.builder_config.with_noise,
            )

            return samples["y"], samples["theta"], scores

        master_key = jax.random.PRNGKey(2948570986789)

        for i in range(size // bs):
            key, master_key = jax.random.split(master_key)
            simu, theta, score = get_batch(key, thetas[i])

            for j in range(bs):
                yield f"{i}-{j}", {
                    "simulation": simu[j],
                    "theta": theta[j],
                    "score": score[j],
                }
