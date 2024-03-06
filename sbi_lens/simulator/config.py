import jax_cosmo as jc

from sbi_lens.simulator.redshift import subdivide


class Config:
    def __init__(
        self,
        N,
        map_size,
        sigma_e,
        gals_per_arcmin2,
        nbins,
        a,
        b,
        z0,
        omega_c,
        omega_b,
        sigma_8,
        h_0,
        n_s,
        w_0,
    ):
        self.nb_cosmo_params = 6
        self.N = N
        self.map_size = map_size
        self.sigma_e = sigma_e
        self.gals_per_arcmin2 = gals_per_arcmin2
        self.nbins = nbins
        self.a = a
        self.b = b
        self.z0 = z0
        self.f_sky = self.map_size**2 / 41_253
        self.nz = jc.redshift.smail_nz(
            self.a, self.b, self.z0, gals_per_arcmin2=self.gals_per_arcmin2
        )
        self.nz_bins = subdivide(self.nz, nbins=self.nbins, zphot_sigma=0.05)
        self.tracer = jc.probes.WeakLensing(self.nz_bins, sigma_e=self.sigma_e)

        self.params_name_latex = [
            "$\Omega_c$",
            "$\Omega_b$",
            "$\sigma_8$",
            "$h_0$",
            "$n_s$",
            "$w_0$",
        ]
        self.params_name = ["omega_c", "omega_b", "sigma_8", "h_0", "n_s", "w_0"]
        self.truth = [omega_c, omega_b, sigma_8, h_0, n_s, w_0]


config_lsst_y_10 = Config(
    N=256,
    map_size=10,
    sigma_e=0.26,
    gals_per_arcmin2=27,
    nbins=5,
    a=2,
    b=0.68,
    z0=0.11,
    omega_c=0.2664,
    omega_b=0.0492,
    sigma_8=0.831,
    h_0=0.6727,
    n_s=0.9645,
    w_0=-1.0,
)
