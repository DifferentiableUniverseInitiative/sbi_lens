import logging

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):

    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())

from tqdm import tqdm
from chainconsumer import ChainConsumer
from pathlib import Path

import jax
import pickle
import jax.numpy as jnp
from jax.lib import xla_bridge

print(xla_bridge.get_backend().platform)

import tensorflow_probability as tfp

tfp = tfp.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors
import tensorflow_datasets as tfds
import tensorflow as tf
import haiku as hk
import optax
from absl import app
import numpy as np
from absl import flags

from sbi_lens.training.training_config import compressor_conf, estimator_conf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

flags.DEFINE_string(
    "model_name",
    "LensingLogNormalDataset/toy_model_without_noise_score_density",
    "Configuration for the Lensing LogNormal dataset")
flags.DEFINE_string("data_dir",
                    '/linkhome/rech/genmfd01/ulm75uc/tensorflow_datasets',
                    "Input data folder")
flags.DEFINE_float("map_size", 5., "Size of the lensing field in degrees")
flags.DEFINE_integer("N", 128, "Number of pixels on the map.")
flags.DEFINE_integer("gal_per_arcmin2", 30, "Number of galaxies per arcmin")
flags.DEFINE_float("sigma_e", 0.26,
                   "Dispersion of the ellipticity distribution")
flags.DEFINE_boolean(
    "Augmentation_with_noise", False,
    "if True, gaussian noise will be added during the augmentation stage")

flags.DEFINE_integer("batch_size_com", 128,
                     " Number of training examples in one forward pass")
flags.DEFINE_integer("total_steps_com", 2, "Number of iteration")

flags.DEFINE_integer("score_weight", 0, "Score weight")
flags.DEFINE_integer("total_steps_est", 5, "Number of iteration")

FLAGS = flags.FLAGS

SOURCE_FILE = Path(__file__)
SOURCE_DIR = SOURCE_FILE.parent
ROOT_DIR = SOURCE_DIR.parent.resolve()
DATA_DIR = ROOT_DIR / "data"


def augmentation_with_noise(example):
    pix_area = (FLAGS.map_size * 60 / FLAGS.N)**2
    x = tf.expand_dims(example['simulation'], axis=-1) + tf.random.normal(
        shape=(FLAGS.N, FLAGS.N, 1),
        stddev=FLAGS.sigma_e / tf.math.sqrt(FLAGS.gal_per_arcmin2 * pix_area))
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return {
        'simulation': x,
        'theta': example['theta'],
        'score': example['score']
    }


def data_preprocessing(Augmentation_with_noise):
    ds = tfds.load(FLAGS.model_name, split='train', data_dir=FLAGS.data_dir)
    ds = ds.repeat()
    ds = ds.shuffle(1000)
    if Augmentation_with_noise:
        ds = ds.map(augmentation_with_noise)
    ds = ds.batch(128)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    ds_train = iter(tfds.as_numpy(ds))
    return ds_train


def train_compressor(ds_train):
    nf, compressor, parameters_compressor, opt_state_resnet = compressor_conf()

    def loss_vmim(params, mu, batch, state_resnet):
        y, opt_state_resnet = compressor.apply(
            params, state_resnet, None, batch.reshape([-1, 128, 128, 1]))
        log_prob = jax.vmap(lambda theta, x: nf.apply(
            params, theta.reshape([1, 2]), x.reshape([1, 2])).squeeze())(mu, y)
        return -jnp.mean(log_prob), opt_state_resnet

    @jax.jit
    def update_compressor_with_vmim(params, opt_state, mu, batch,
                                    state_resnet):
        """Single SGD update step."""
        (loss, opt_state_resnet), grads = jax.value_and_grad(
            loss_vmim, has_aux=True)(params, mu, batch, state_resnet)
        updates, new_opt_state = optimizer_c.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state, opt_state_resnet

    lr_scheduler = optax.piecewise_constant_schedule(
        init_value=0.001,
        boundaries_and_scales={
            int(FLAGS.total_steps_com * 0.1): 0.7,
            int(FLAGS.total_steps_com * 0.2): 0.7,
            int(FLAGS.total_steps_com * 0.3): 0.7,
            int(FLAGS.total_steps_com * 0.4): 0.7,
            int(FLAGS.total_steps_com * 0.5): 0.7,
            int(FLAGS.total_steps_com * 0.6): 0.7,
            int(FLAGS.total_steps_com * 0.7): 0.7,
            int(FLAGS.total_steps_com * 0.8): 0.7,
            int(FLAGS.total_steps_com * 0.9): 0.7
        })
    optimizer_c = optax.adam(learning_rate=lr_scheduler)
    opt_state_c = optimizer_c.init(parameters_compressor)
    batch_loss = []
    for batch in tqdm(range(FLAGS.total_steps_com)):
        sample = next(ds_train)
        l, parameters_compressor, opt_state_c, opt_state_resnet = update_compressor_with_vmim(
            parameters_compressor, opt_state_c, sample['theta'],
            sample['simulation'], opt_state_resnet)
        if batch % 100 == 0:
            batch_loss.append(l)
    return compressor, parameters_compressor, opt_state_resnet


def train_estimator(ds_train, sample_full_field, scale_theta, shift_theta):

    compressor, parameters_compressor, opt_state_resnet = train_compressor(
        ds_train)
    nvp_nd, nvp_sample_nd = estimator_conf(scale_theta, shift_theta,
                                           sample_full_field)
    rng_seq = hk.PRNGSequence(1989)
    params_nd = nvp_nd.init(next(rng_seq), 0.5 * jnp.ones([1, 2]),
                            0.5 * jnp.ones([1, 2]))

    def loss_fn(params, parameters_compressor, opt_state_resnet, weight, mu,
                batch, score):

        y, _ = compressor.apply(parameters_compressor, opt_state_resnet, None,
                                batch.reshape([-1, 128, 128, 1]))

        log_prob, out = jax.vmap(
            jax.value_and_grad(lambda theta, x: nvp_nd.apply(
                params, theta.reshape([1, 2]), x.reshape([1, 2])).squeeze()))(
                    mu, y)

        return -jnp.mean(log_prob) + weight * jnp.mean(
            jnp.sum((out - score)**2, axis=1))

    @jax.jit
    def update(params, parameters_compressor, opt_state, opt_state_resnet,
               weight, mu, batch, score):
        """Single SGD update step."""
        loss, grads = jax.value_and_grad(loss_fn)(params,
                                                  parameters_compressor,
                                                  opt_state_resnet, weight, mu,
                                                  batch, score)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return loss, new_params, new_opt_state

    # training
    lr_scheduler = optax.piecewise_constant_schedule(
        init_value=0.001,
        boundaries_and_scales={
            int(FLAGS.total_steps_est * 0.2): 0.7,
            int(FLAGS.total_steps_est * 0.4): 0.7,
            int(FLAGS.total_steps_est * 0.6): 0.7,
            int(FLAGS.total_steps_est * 0.8): 0.7
        })
    optimizer = optax.adam(learning_rate=lr_scheduler)
    opt_state = optimizer.init(params_nd)
    batch_loss = []
    for batch in tqdm(range(FLAGS.total_steps_est)):
        sample = next(ds_train)
        l, params_nd, opt_state = update(params_nd, parameters_compressor,
                                         opt_state, opt_state_resnet,
                                         FLAGS.score_weight, sample['theta'],
                                         sample['simulation'], sample['score'])
        if batch % 100 == 0:
            batch_loss.append(l)

    return compressor, parameters_compressor, opt_state_resnet, params_nd


def main(_):
    # plot resultst
    ds_train = data_preprocessing(
        Augmentation_with_noise=FLAGS.Augmentation_with_noise)
    scale_theta = jnp.array([0.8183354, 0.8473379])
    shift_theta = jnp.array([-0.53523827, -0.50171137])
    m_data = jnp.load(DATA_DIR / 'm_data_lensing.npy')
    sample_power_spectrum = np.load(DATA_DIR /
                                    'sample_power_spectrum_toy_model.npy')
    sample_full_field = np.load(DATA_DIR / 'sample_full_field.npy')
    compressor, parameters_compressor, opt_state_resnet, params_nd = train_estimator(
        ds_train, sample_full_field, scale_theta, shift_theta)
    nvp_nd, nvp_sample_nd = estimator_conf(scale_theta, shift_theta,
                                           sample_power_spectrum)
    rng_seq = hk.PRNGSequence(1989)
    y, _ = compressor.apply(parameters_compressor, opt_state_resnet, None,
                            m_data.reshape([1, 128, 128, 1]))
    sample_nd = nvp_sample_nd.apply(params_nd,
                                    rng=next(rng_seq),
                                    x=y *
                                    jnp.ones([len(sample_power_spectrum), 2]))
    c = ChainConsumer()
    c.add_chain(sample_power_spectrum,
                parameters=["$\Omega_c$", "$\sigma_8$"],
                name='Power Spectrum')
    c.add_chain(sample_full_field,
                parameters=["$\Omega_c$", "$\sigma_8$"],
                name='Full Field')
    c.add_chain(sample_nd, parameters=["$\Omega_c$", "$\sigma_8$"], name='SBI')

    fig = c.plotter.plot(figsize="column", truth=[0.3, 0.8])
    with open(DATA_DIR / "params_nd_compressor.pkl", "wb") as fp:
        pickle.dump(parameters_compressor, fp)

    with open(DATA_DIR / "opt_state_resnet.pkl", "wb") as fp:
        pickle.dump(opt_state_resnet, fp)
    fig.savefig(DATA_DIR / 'image.pdf')


if __name__ == "__main__":
    app.run(main)
