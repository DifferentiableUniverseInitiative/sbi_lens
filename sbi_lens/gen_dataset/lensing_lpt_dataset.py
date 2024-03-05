from functools import partial

import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow_datasets.core.utils import gcs_utils

from sbi_lens.config import config_lsst_y_10
from sbi_lens.simulator.Lpt_field import lensingLpt
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


class LensingLPTDatasetConfig(tfds.core.BuilderConfig):
    def __init__(
        self,
        *,
        N,
        map_size,
        box_size,
        box_shape,
        gal_per_arcmin2,
        sigma_e,
        nbins,
        a,
        b,
        z0,
        score_type,
        with_noise,
        **kwargs,
    ):
        v1 = tfds.core.Version("0.0.1")
        super().__init__(description=("LPT lensing simulations."), version=v1, **kwargs)
        self.N = N
        self.map_size = map_size
        self.box_size = box_size
        self.box_shape = box_shape
        self.gal_per_arcmin2 = gal_per_arcmin2
        self.sigma_e = sigma_e
        self.nbins = nbins
        self.a = a
        self.b = b
        self.z0 = z0
        self.score_type = score_type
        self.with_noise = with_noise


class LensingLPTDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for LensingLPTDataset dataset."""

    VERSION = tfds.core.Version("0.0.1")
    RELEASE_NOTES = {
        "0.0.1": "Initial release.",
    }
    BUILDER_CONFIGS = [
        LensingLPTDatasetConfig(
            name="year_10_with_noise_score_density",
            N=60,
            map_size=5,
            box_size=[400.0, 400.0, 4000.0],
            box_shape=[300, 300, 128],
            gal_per_arcmin2=config_lsst_y_10.gals_per_arcmin2,
            sigma_e=config_lsst_y_10.sigma_e,
            nbins=config_lsst_y_10.nbins,
            a=config_lsst_y_10.a,
            b=config_lsst_y_10.b,
            z0=config_lsst_y_10.z0,
            score_type="density",
            with_noise=True,
        ),
        LensingLPTDatasetConfig(
            name="year_10_without_noise_score_density",
            N=60,
            map_size=5,
            box_size=[400.0, 400.0, 4000.0],
            box_shape=[300, 300, 128],
            gal_per_arcmin2=config_lsst_y_10.gals_per_arcmin2,
            sigma_e=config_lsst_y_10.sigma_e,
            nbins=config_lsst_y_10.nbins,
            a=config_lsst_y_10.a,
            b=config_lsst_y_10.b,
            z0=config_lsst_y_10.z0,
            score_type="density",
            with_noise=False,
        ),
        LensingLPTDatasetConfig(
            name="year_10_with_noise_score_conditional",
            N=60,
            map_size=5,
            box_size=[400.0, 400.0, 4000.0],
            box_shape=[300, 300, 128],
            gal_per_arcmin2=config_lsst_y_10.gals_per_arcmin2,
            sigma_e=config_lsst_y_10.sigma_e,
            nbins=config_lsst_y_10.nbins,
            a=config_lsst_y_10.a,
            b=config_lsst_y_10.b,
            z0=config_lsst_y_10.z0,
            score_type="conditional",
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
        bs = 5

        model = partial(
            lensingLpt,
            self.builder_config.N,
            self.builder_config.map_size,
            self.builder_config.box_size,
            self.builder_config.box_shape,
            self.builder_config.gal_per_arcmin2,
            self.builder_config.sigma_e,
            self.builder_config.nbins,
            self.builder_config.a,
            self.builder_config.b,
            self.builder_config.z0,
            self.builder_config.with_noise,
        )

        @jax.jit
        def get_batch(key):
            (_, samples), scores = get_samples_and_scores(
                model=model,
                key=key,
                batch_size=bs,
                score_type=self.builder_config.score_type,
                thetas=None,
                with_noise=self.builder_config.with_noise,
            )

            return samples["y"], samples["theta"], scores

        master_key = jax.random.PRNGKey(2948570986789)

        for i in range(size // bs):
            key, master_key = jax.random.split(master_key)
            simu, theta, score = get_batch(key)

            for j in range(bs):
                yield f"{i}-{j}", {
                    "simulation": simu[j],
                    "theta": theta[j],
                    "score": score[j],
                }
