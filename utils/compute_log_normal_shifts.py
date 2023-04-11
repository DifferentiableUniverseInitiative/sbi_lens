import numpy as np
import os
from multiprocessing.pool import Pool
from absl import app
from absl import flags
import ctypes
import jax_cosmo as jc
from sbi_lens.simulator.redshift import subdivide

flags.DEFINE_float("s8_min", 0.3, "sigma8 min")
flags.DEFINE_float("s8_max", 1.3, "sigma8 max")
flags.DEFINE_float("om_min", 0.1, "omegam min")
flags.DEFINE_float("om_max", 0.9, "omegam max")
flags.DEFINE_float("w_min", -2, "w min")
flags.DEFINE_float("w_max", -.3, "w max")
flags.DEFINE_integer("nsteps", 10, "number of steps in parameters")
flags.DEFINE_float("ob", 0.0492, "omegab")
flags.DEFINE_float("h", 0.6727, "h")
flags.DEFINE_float("ns", 0.9645, "ns")
flags.DEFINE_float("wa", 0.0, "wa")

flags.DEFINE_float("pixel_size", 2.34, "pixel scale in arcmin")

# Parameters for redshift distribution
flags.DEFINE_float("smail_a", 2., "smail a parameter")
flags.DEFINE_float("smail_b", 0.68, "smail b parameter")
flags.DEFINE_float("smail_z0", 0.11, "smail z0 parameter")
flags.DEFINE_integer("nbins", 5, "Number of redshift bins")
flags.DEFINE_float("zphot_sigma", 0.05, "photoz error")

FLAGS = flags.FLAGS

def fn(params):
    # Importing the library and defining useful functions
    lib=ctypes.CDLL("./DSS.so")
    initialise_new_Universe = lib.initialise_new_Universe
    initialise_new_Universe.argtypes = [ ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    initialise_new_Universe.restype = None

    add_projected_galaxy_sample = lib.add_projected_galaxy_sample
    add_projected_galaxy_sample.argtypes = [         ctypes.c_int,         ctypes.c_char_p,                    ctypes.c_double,  ctypes.c_double,  ctypes.c_double, ctypes.c_double, ctypes.c_double]
    add_projected_galaxy_sample.restype = None

    return_lognormal_shift_for_individual_FLASK_bin = lib.return_lognormal_shift_for_individual_FLASK_bin
    return_lognormal_shift_for_individual_FLASK_bin.argtypes = [       ctypes.c_double,               ctypes.c_int,            ctypes.c_int]
    return_lognormal_shift_for_individual_FLASK_bin.restype = ctypes.c_double

    # Extracting parameters at which to compute the shift parameters
    omega_m, sigma8, w0, bin_id = params

    # initialising a new universe and its matter content
    a_initial = 0.000025
    a_final = 1.0
    density_sample_1 = 69.47036304452095/(np.pi*30.0**2) #not used when we compute the shift for galaxy sources
    b1_sample_1 = 1.8  #not used when we compute the shift for galaxy sources
    b2_sample_1 = 0.0  #not used when we compute the shift for galaxy sources
    a0 = 1.26          #not used when we compute the shift for galaxy sources
    a1 = 0.28          #not used when we compute the shift for galaxy sources

    # Read the filename
    nz_filename = ctypes.c_char_p(( 'outputs/shear_photoz_bin_%d.tab'%bin_id).encode('utf-8'))

    initialise_new_Universe(a_initial, a_final, omega_m, FLAGS.ob, 0.0, 1.0-omega_m, sigma8, FLAGS.ns, FLAGS.h, w0, FLAGS.wa)
    add_projected_galaxy_sample(0, nz_filename, density_sample_1, b1_sample_1, b2_sample_1, a0, a1)
    # Converting from pixel size to scale used in cosmomentum, following Boruah et al. 2021
    theta_in_arcmin = FLAGS.pixel_size / np.sqrt(np.pi)

    return omega_m, sigma8, w0, bin_id, return_lognormal_shift_for_individual_FLASK_bin(theta_in_arcmin, 0, 0)


def main(_):
    # Define the redshift distributions we are going to use, and save them to file for cosmomentum
    parent_nz = jc.redshift.smail_nz(FLAGS.smail_a,  FLAGS.smail_b,  FLAGS.smail_z0)
    nz_bins = subdivide(parent_nz, nbins=FLAGS.nbins, zphot_sigma=FLAGS.zphot_sigma)

    # Build the redshift distribution files
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    header = 'z_left p([z_left, z_right])'
    z = np.linspace(0, 5, 1024)
    for i in range(FLAGS.nbins):
        row_list = np.stack([z, nz_bins[i](z)],axis=1)
        np.savetxt('outputs/shear_photoz_bin_%d.tab'%i,  row_list,  header=header, delimiter='\t')

    # Generate the range of parameters we want to probe
    om = np.linspace(FLAGS.om_min, FLAGS.om_max, FLAGS.nsteps)
    s8 = np.linspace(FLAGS.s8_min, FLAGS.s8_max, FLAGS.nsteps)
    w = np.linspace(FLAGS.w_min, FLAGS.w_max, FLAGS.nsteps)
    bin_id = np.arange(FLAGS.nbins)

    # Create the grid of parameter values
    grid = np.stack(np.meshgrid(om, s8, w, bin_id), axis=-1).reshape([-1, 4])

    # Start multiprocessing the grid
    pool = Pool(maxtasksperchild=1)
    print('Starting the computation')
    results = pool.map(fn, grid, chunksize=1)
    print('Done')
    pool.close()

    # Saving the results
    results = np.array(results).reshape([FLAGS.nsteps,FLAGS.nsteps,FLAGS.nsteps,FLAGS.nbins, 5])

    np.save('outputs/lognormal_shifts_om_s8_w_bin.npy', results)

if __name__ == "__main__":
    app.run(main)
