import astropy.units as u
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jax_cosmo.constants as constants
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.ndimage import map_coordinates
from jax_cosmo.scipy.integrate import simps
from jaxpm.kernels import fftk
from jaxpm.painting import cic_paint, compensate_cic
from jaxpm.pm import growth_factor, pm_forces

from sbi_lens.simulator.redshift import subdivide

# code from François Lanusse


def linear_field(mesh_shape, box_size, pk, field):
    """
    Generate initial conditions.
    """
    kvec = fftk(mesh_shape)
    kmesh = (
        sum((kk / box_size[i] * mesh_shape[i]) ** 2 for i, kk in enumerate(kvec)) ** 0.5
    )
    pkmesh = (
        pk(kmesh)
        * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2])
        / (box_size[0] * box_size[1] * box_size[2])
    )

    field = jnp.fft.rfftn(field) * pkmesh**0.5
    field = jnp.fft.irfftn(field)
    return field


def lpt_lightcone(cosmo, initial_conditions, positions, a, mesh_shape):
    """
    Computes first order LPT displacement
    """
    initial_force = pm_forces(positions, delta=initial_conditions).reshape(
        mesh_shape + [3]
    )
    a = jnp.atleast_1d(a)
    dx = growth_factor(cosmo, a).reshape([1, 1, -1, 1]) * initial_force
    return dx.reshape([-1, 3])


def convergence_Born(cosmo, density_planes, r, a, dx, dz, coords, z_source):
    """
    Compute the Born convergence
    Args:
      cosmo: `Cosmology`, cosmology object.
      density_planes: list of dictionaries (r, a, density_plane, dx, dz),
        lens planes to use
      coords: a 3-D array of angular coordinates in radians of N points with
        shape [batch, N, 2].
      z_source: 1-D `Tensor` of source redshifts with shape [Nz] .
      name: `string`, name of the operation.
    Returns:
      `Tensor` of shape [batch_size, N, Nz], of convergence values.
    """
    # Compute constant prefactor:
    constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c) ** 2
    # Compute comoving distance of source galaxies
    r_s = jc.background.radial_comoving_distance(cosmo, 1 / (1 + z_source))

    convergence = 0
    n_planes = len(r)

    def scan_fn(carry, i):
        density_planes, a, r = carry

        p = density_planes[:, :, i]
        density_normalization = dz * r[i] / a[i]
        p = (p - p.mean()) * constant_factor * density_normalization

        # Interpolate at the density plane coordinates
        im = map_coordinates(p, coords * r[i] / dx - 0.5, order=1, mode="wrap")

        return carry, im * jnp.clip(1.0 - (r[i] / r_s), 0, 1000).reshape([-1, 1, 1])

    _, convergence = jax.lax.scan(scan_fn, (density_planes, a, r), jnp.arange(n_planes))

    return convergence.sum(axis=0)


def make_full_field_model(field_size, field_npix, box_shape, box_size):
    def forward_model(cosmo, nz_shear, initial_conditions):
        # Create a small function to generate the matter power spectrum
        k = jnp.logspace(-4, 1, 128)
        pk = jc.power.linear_matter_power(cosmo, k)

        def pk_fn(x):
            return jc.scipy.interpolate.interp(x.reshape([-1]), k, pk).reshape(x.shape)

        # Create initial conditions
        lin_field = linear_field(box_shape, box_size, pk_fn, initial_conditions)

        # Create particles
        particles = jnp.stack(
            jnp.meshgrid(*[jnp.arange(s) for s in box_shape]), axis=-1
        ).reshape([-1, 3])

        # Compute the scale factor that corresponds to each slice of the volume
        r = (jnp.arange(box_shape[-1]) + 0.5) * box_size[-1] / box_shape[-1]
        a = jc.background.a_of_chi(cosmo, r)

        cosmo = jc.Cosmology(
            Omega_c=cosmo.Omega_c,
            sigma8=cosmo.sigma8,
            Omega_b=cosmo.Omega_b,
            h=cosmo.h,
            n_s=cosmo.n_s,
            w0=cosmo.w0,
            Omega_k=0.0,
            wa=0.0,
        )

        # Initial displacement
        dx = lpt_lightcone(cosmo, lin_field, particles, a, box_shape)

        # Paint the particles on a new mesh
        lightcone = cic_paint(jnp.zeros(box_shape), particles + dx)
        # Apply de-cic filter to recover more signal on small scales
        lightcone = compensate_cic(lightcone)

        dx = box_size[0] / box_shape[0]
        dz = box_size[-1] / box_shape[-1]

        # Defining the coordinate grid for lensing map
        xgrid, ygrid = np.meshgrid(
            np.linspace(
                0, field_size, box_shape[0], endpoint=False
            ),  # range of X coordinates
            np.linspace(0, field_size, box_shape[1], endpoint=False),
        )  # range of Y coordinates
        coords = jnp.array((np.stack([xgrid, ygrid], axis=0) * u.deg).to(u.rad))

        # Generate convergence maps by integrating over nz and source planes
        convergence_maps = [
            simps(
                lambda z: nz(z).reshape([-1, 1, 1])
                * convergence_Born(cosmo, lightcone, r, a, dx, dz, coords, z),
                0.01,
                3.0,
                N=32,
            )
            for nz in nz_shear
        ]

        # Reshape the maps to desired resoluton
        convergence_maps = [
            kmap.reshape(
                [
                    field_npix,
                    box_shape[0] // field_npix,
                    field_npix,
                    box_shape[1] // field_npix,
                ]
            )
            .mean(axis=1)
            .mean(axis=-1)
            for kmap in convergence_maps
        ]

        return convergence_maps, lightcone

    return forward_model


# Define the probabilistic model
def lensingLpt(
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
    with_noise=True
):
    """
    This function defines the top-level forward model for our observations
    """

    pix_area = (map_size * 60 / N) ** 2

    # Sampling initial conditions
    initial_conditions = numpyro.sample(
        "z", dist.Normal(jnp.zeros(box_shape), jnp.ones(box_shape))
    )

    omega_c = numpyro.sample("omega_c", dist.TruncatedNormal(0.2664, 0.2, low=0))
    omega_b = numpyro.sample("omega_b", dist.Normal(0.0492, 0.006))
    sigma_8 = numpyro.sample("sigma_8", dist.Normal(0.831, 0.14))
    h_0 = numpyro.sample("h_0", dist.Normal(0.6727, 0.063))
    n_s = numpyro.sample("n_s", dist.Normal(0.9645, 0.08))
    w_0 = numpyro.sample("w_0", dist.TruncatedNormal(-1.0, 0.9, low=-2.0, high=-0.3))

    cosmo = jc.Cosmology(
        Omega_c=omega_c,
        Omega_b=omega_b,
        sigma8=sigma_8,
        h=h_0,
        n_s=n_s,
        w0=w_0,
        wa=0.0,
        Omega_k=0.0,
    )

    # Generate random convergence maps
    nz = jc.redshift.smail_nz(a, b, z0, gals_per_arcmin2=gal_per_arcmin2)
    nz_shear = subdivide(nz, nbins=nbins, zphot_sigma=0.05)

    lensing_model = jax.jit(
        make_full_field_model(
            field_size=map_size,
            field_npix=N,
            box_size=box_size,
            box_shape=box_shape,
        )
    )

    field, _ = lensing_model(cosmo, nz_shear, initial_conditions)
    field = jnp.transpose(jnp.array(field), [1, 2, 0])

    if with_noise is True:
        x = numpyro.sample(
            "y",
            dist.MultivariateNormal(
                loc=field,
                covariance_matrix=jnp.diag(
                    sigma_e**2
                    / (jnp.array([b.gals_per_arcmin2 for b in nz_shear]) * pix_area)
                ),
            ),
        )
    else:
        x = numpyro.deterministic("y", field)

    return x
