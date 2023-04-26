import jax.numpy as jnp
import tensorflow as tf
import jax_cosmo as jc
from sbi_lens.simulator.redshift import subdivide


def augmentation_flip(example):

  x = example['simulation']
  x = tf.image.random_flip_left_right(x)
  x = tf.image.random_flip_up_down(x)

  return {
    'simulation': x,
    'theta': example['theta'],
    'score': example['score']
  }

def augmentation_noise(
  example,
  N,
  map_size,
  sigma_e,
  gal_per_arcmin2,
  nbins,
  a,
  b,
  z0
  ):

  nz = jc.redshift.smail_nz(a, b, z0, gals_per_arcmin2=gal_per_arcmin2)
  nz_bins = subdivide(nz, nbins=nbins, zphot_sigma=0.05)

  pix_area = (map_size * 60 / N)**2

  x = example['simulation']
  x += tf.random.normal(
    shape=(N, N, nbins),
    stddev=sigma_e / tf.math.sqrt(
      jnp.array([b.gals_per_arcmin2 for b in nz_bins]) * pix_area
    )
  )

  return {
    'simulation': x,
    'theta': example['theta'],
    'score': example['score']
  }