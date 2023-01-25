import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_probability as tfp

tfp = tfp.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors
from tensorflow_datasets.core.utils import gcs_utils
import jax
from functools import partial
from sbi_lens.simulator import LogNormal_field
from sbi_lens.gen_dataset.utils import get_samples_and_scores

# disable internet connection
gcs_utils.gcs_dataset_info_files = lambda *args, **kwargs: None
gcs_utils.is_dataset_on_gcs = lambda *args, **kwargs: False

_CITATION = """
"""

_DESCRIPTION = """
"""


class LensingLogNormalDatasetConfig(tfds.core.BuilderConfig):

    def __init__(self, *, N, map_size, gal_per_arcmin2, sigma_e, model_type,
                 proposal, score_type, with_noise, **kwargs):
        v1 = tfds.core.Version("0.0.1")
        super(LensingLogNormalDatasetConfig,
              self).__init__(description=("Log Normal lensing simulations."),
                             version=v1,
                             **kwargs)
        self.N = N
        self.map_size = map_size
        self.gal_per_arcmin2 = gal_per_arcmin2
        self.sigma_e = sigma_e
        self.model_type = model_type
        self.proposal = proposal
        self.score_type = score_type
        self.with_noise = with_noise


class LensingLogNormalDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for LensingLogNormalDataset dataset."""

    VERSION = tfds.core.Version('0.0.1')
    RELEASE_NOTES = {
        '0.0.1': 'Initial release.',
    }
    BUILDER_CONFIGS = [
        LensingLogNormalDatasetConfig(
            name="toy_model_without_noise_score_density",
            N=128,
            map_size=5,
            gal_per_arcmin2=30,
            sigma_e=0.2,
            model_type='lognormal',
            proposal=False,
            score_type='density',
            with_noise=False),
        LensingLogNormalDatasetConfig(
            name="toy_model_with_noise_score_density",
            N=128,
            map_size=5,
            gal_per_arcmin2=30,
            sigma_e=0.2,
            model_type='lognormal',
            proposal=False,
            score_type='density',
            with_noise=True),
        LensingLogNormalDatasetConfig(name="year_1_with_noise_score_density",
                                      N=128,
                                      map_size=5,
                                      gal_per_arcmin2=10,
                                      sigma_e=0.26,
                                      model_type='lognormal',
                                      proposal=True,
                                      score_type='density',
                                      with_noise=True),
        LensingLogNormalDatasetConfig(name="year_10_with_noise_score_density",
                                      N=128,
                                      map_size=5,
                                      gal_per_arcmin2=27,
                                      sigma_e=0.26,
                                      model_type='lognormal',
                                      proposal=True,
                                      score_type='density',
                                      with_noise=True),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'simulation':
                tfds.features.Tensor(
                    shape=[self.builder_config.N, self.builder_config.N],
                    dtype=tf.float32),
                'theta':
                tfds.features.Tensor(shape=[2], dtype=tf.float32),
                'score':
                tfds.features.Tensor(shape=[2], dtype=tf.float32),
            }),
            supervised_keys=None,
            homepage='https://dataset-homepage/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        return [
            tfds.core.SplitGenerator(name=tfds.Split.TRAIN,
                                     gen_kwargs={'size': 100000}),
        ]

    def _generate_examples(self, size):
        """Yields examples."""

        model = partial(LogNormal_field, self.builder_config.N,
                        self.builder_config.map_size,
                        self.builder_config.gal_per_arcmin2,
                        self.builder_config.sigma_e,
                        self.builder_config.model_type,
                        self.builder_config.with_noise)

        @jax.jit
        def get_batch(key, thetas):
            (_, samples), scores = get_samples_and_scores(
                model=model,
                key=key,
                batch_size=bs,
                score_type=self.builder_config.score_type,
                thetas=thetas,
                with_noise=self.builder_config.with_noise)

            return samples['y'], samples['theta'], scores

        master_key = jax.random.PRNGKey(2948570986789)

        bs = 50
        for i in range(size // bs):
            key, master_key = jax.random.split(master_key)
            simu, theta, score = get_batch(key, None)

            for j in range(bs):
                yield '{}-{}'.format(i, j), {
                    'simulation': simu[j],
                    'theta': theta[j],
                    'score': score[j]
                }
