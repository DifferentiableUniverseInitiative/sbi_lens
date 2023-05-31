import argparse
import pickle
from functools import partial
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from haiku._src.nets.resnet import ResNet18, ResNet34
from tqdm import tqdm

from sbi_lens.config import config_lsst_y_10
from sbi_lens.gen_dataset.utils import augmentation_flip, augmentation_noise
from sbi_lens.normflow.models import AffineCoupling, ConditionalRealNVP
from sbi_lens.normflow.train_model import TrainModel

tfp = tfp.experimental.substrates.jax
tfb = tfp.bijectors
tfd = tfp.distributions


# script arguments
parser = argparse.ArgumentParser()
parser.add_argument("--total_steps", type=int, default=50)
parser.add_argument("--resnet", type=str, default="resnet18")
parser.add_argument("--loss", type=str, default="train_compressor_vmim")

args = parser.parse_args()

SOURCE_FILE = Path(__file__)
SOURCE_DIR = SOURCE_FILE.parent
ROOT_DIR = SOURCE_DIR.parent.resolve()
DATA_DIR = ROOT_DIR / "sbi_lens/data"

print("######## CONFIG LSST Y 10 ########")

dim = 6

N = config_lsst_y_10.N
map_size = config_lsst_y_10.map_size
sigma_e = config_lsst_y_10.sigma_e
gals_per_arcmin2 = config_lsst_y_10.gals_per_arcmin2
nbins = config_lsst_y_10.nbins
a = config_lsst_y_10.a
b = config_lsst_y_10.b
z0 = config_lsst_y_10.z0

truth = config_lsst_y_10.truth

params_name = config_lsst_y_10.params_name

print("######## LOAD OBSERVATION ########")

# plot observed mass map
m_data = jnp.load(
    DATA_DIR / f"m_data__{N}N_{map_size}ms_{gals_per_arcmin2}gpa_{sigma_e}se.npy"
)

print("######## DATA AUGMENTATION ########")
tf.random.set_seed(1)


def augmentation(example):
    return augmentation_flip(
        augmentation_noise(
            example=example,
            N=N,
            map_size=map_size,
            sigma_e=sigma_e,
            gal_per_arcmin2=gals_per_arcmin2,
            nbins=nbins,
            a=a,
            b=b,
            z0=z0,
        )
    )


print("######## CREATE COMPRESSOR ########")

# nf
bijector_layers_compressor = [128] * 2

bijector_compressor = partial(
    AffineCoupling, layers=bijector_layers_compressor, activation=jax.nn.silu
)

NF_compressor = partial(ConditionalRealNVP, n_layers=4, bijector_fn=bijector_compressor)


class Flow_nd_Compressor(hk.Module):
    def __call__(self, y):
        nvp = NF_compressor(dim)(y)
        return nvp


nf = hk.without_apply_rng(
    hk.transform(lambda theta, y: Flow_nd_Compressor()(y).log_prob(theta).squeeze())
)

# compressor
if args.resnet == "resnet34":
    print("ResNet34")

    compressor = hk.transform_with_state(lambda y: ResNet34(dim)(y, is_training=True))

elif args.resnet == "resnet18":
    print("ResNet18")

    compressor = hk.transform_with_state(lambda y: ResNet18(dim)(y, is_training=True))

print("######## TRAIN ########")

# init compressor
parameters_resnet, opt_state_resnet = compressor.init(
    jax.random.PRNGKey(0), y=0.5 * jnp.ones([1, N, N, nbins])
)
# init nf
params_nf = nf.init(
    jax.random.PRNGKey(0), theta=0.5 * jnp.ones([1, dim]), y=0.5 * jnp.ones([1, dim])
)

if args.loss == "train_compressor_vmim":
    parameters_compressor = hk.data_structures.merge(parameters_resnet, params_nf)
elif args.loss == "train_compressor_mse":
    parameters_compressor = parameters_resnet


# define optimizer
total_steps = args.total_steps
lr_scheduler = optax.piecewise_constant_schedule(
    init_value=0.001,
    boundaries_and_scales={
        int(total_steps * 0.1): 0.7,
        int(total_steps * 0.2): 0.7,
        int(total_steps * 0.3): 0.7,
        int(total_steps * 0.4): 0.7,
        int(total_steps * 0.5): 0.7,
        int(total_steps * 0.6): 0.7,
        int(total_steps * 0.7): 0.7,
        int(total_steps * 0.8): 0.7,
        int(total_steps * 0.9): 0.7,
    },
)

optimizer_c = optax.adam(learning_rate=lr_scheduler)
opt_state_c = optimizer_c.init(parameters_compressor)

model_compressor = TrainModel(
    compressor=compressor,
    nf=nf,
    optimizer=optimizer_c,
    loss_name=args.loss,
    nb_pixels=N,
    nb_bins=nbins,
)

ds = tfds.load(
    "LensingLogNormalDataset/year_10_without_noise_score_density", split="train"
)

ds = ds.repeat()
ds = ds.shuffle(1000)
ds = ds.map(augmentation)
ds = ds.batch(128)
ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
ds_train = iter(tfds.as_numpy(ds))

update = jax.jit(model_compressor.update)

store_loss = []
for batch in tqdm(range(total_steps + 1)):
    ex = next(ds_train)
    if not jnp.isnan(ex["simulation"]).any():
        batch_loss, parameters_compressor, opt_state_c, opt_state_resnet = update(
            model_params=parameters_compressor,
            opt_state=opt_state_c,
            theta=ex["theta"],
            x=ex["simulation"],
            state_resnet=opt_state_resnet,
        )
        store_loss.append(batch_loss)

        if jnp.isnan(batch_loss):
            print("NaN Loss")
            break

# save params

if args.loss == "train_compressor_vmim":
    l_name = "vmim"
elif args.loss == "train_compressor_mse":
    l_name = "mse"

with open(
    DATA_DIR / f"params_compressor/params_nd_compressor_{l_name}.pkl", "wb"
) as fp:
    pickle.dump(parameters_compressor, fp)

with open(DATA_DIR / f"params_compressor/opt_state_resnet_{l_name}.pkl", "wb") as fp:
    pickle.dump(opt_state_resnet, fp)
