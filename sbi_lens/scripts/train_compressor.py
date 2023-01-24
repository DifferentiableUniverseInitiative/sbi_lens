
#!/usr/bin/env python
# coding: utf-8

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
from haiku._src.nets.resnet import ResNet18
from absl import app
import numpy as np
from absl import flags

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

flags.DEFINE_string("model_name",
                    "LensingLogNormalDataset/toy_model_without_noise",
                    "Configuration for the Lensing LogNormal dataset")
flags.DEFINE_string("data_dir", 'tensorflow_dataset', "Input data folder")
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
flags.DEFINE_integer("total_steps_com", 200000, "Number of iteration")

flags.DEFINE_integer("score_weight", 0, "Score weight")
flags.DEFINE_integer("total_steps_est", 50000, "Number of iteration")

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

def data_preprocessing(Augmentation_with_noise=False):
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
    
    nf=compressor_conf()
    params_nf = nf.init(jax.random.PRNGKey(8), 0.5 * jnp.ones([1, 2]),
                    0.5 * jnp.ones([1, 2]))
    compressor = hk.transform_with_state(lambda x: ResNet18(2)
                                     (x, is_training=True))
    parameters_resnet, opt_state_resnet = compressor.init(
    jax.random.PRNGKey(873457568), 0.5 * jnp.ones([1, 128, 128, 1]))
    parameters_compressor = hk.data_structures.merge(parameters_resnet, params_nf)   
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
    return parameters_compressor, opt_state_resnet

def train_estimator(ds_train, parameters_compressor, opt_state_resnet, sample_full_field, scale_theta, shift_theta):   
    nvp_nd, nvp_sample_nd =estimator_conf(scale_theta, shift_theta)
    rng_seq = hk.PRNGSequence(1989)
    params_nd = nvp_nd.init(next(rng_seq), 0.5 * jnp.ones([1, 2]),
                            0.5 * jnp.ones([1, 2]))
    # training
    lr_scheduler = optax.piecewise_constant_schedule(
        init_value=0.001,
        boundaries_and_scales={
            int(FLAGS.FLAGS.total_steps_est * 0.2): 0.7,
            int(FLAGS.FLAGS.total_steps_est * 0.4): 0.7,
            int(FLAGS.FLAGS.total_steps_est * 0.6): 0.7,
            int(FLAGS.FLAGS.total_steps_est * 0.8): 0.7
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
            batch_loss.append()

    return lr_scheduler, params_nd


def main(_):
    # plot resultst
    ds_train=data_preprocessing()
    m_data = jnp.load(DATA_DIR /'m_data_lensing.npy')
    sample_power_spectrum = np.load(DATA_DIR /
                                    'sample_power_spectrum_toy_model.npy')
    sample_full_field = np.load(DATA_DIR / 'sample_full_field.npy')
    rng_seq = hk.PRNGSequence(1989)
    parameters_compressor, opt_state_resnet= train_compressor(ds_train)
    y, _ = compressor.apply(parameters_compressor, opt_state_resnet, None,
                            m_data.reshape([1, 128, 128, 1]))
    lr_scheduler, params_nd =train_estimator(ds_train, parameters_compressor, opt_state_resnet)
    sample_nd = nvp_sample_nd.apply(params_nd,
                                    rng=next(rng_seq),
                                    x=y *
                                    jnp.ones([len(sample_power_spectrum), 2]))

    optimizer = optax.adam(learning_rate=lr_scheduler)
    opt_state = optimizer.init(params_nd)
    c = ChainConsumer()
    c.add_chain(sample_power_spectrum,
                parameters=["$\Omega_c$", "$\sigma_8$"],
                name='Power Spectrum')
    c.add_chain(sample_full_field,
                parameters=["$\Omega_c$", "$\sigma_8$"],
                name='Full Field')
    c.add_chain(sample_nd, parameters=["$\Omega_c$", "$\sigma_8$"], name='SBI')

    fig = c.plotter.plot(figsize="column", truth=[0.3, 0.8])
    with open("data/params_nd_compressor.pkl", "wb") as fp:
        pickle.dump(parameters_compressor, fp)

    with open("data/opt_state_resnet.pkl", "wb") as fp:
        pickle.dump(opt_state_resnet, fp)

   
if __name__ == "__main__":
    app.run(main)