r"""The GAMA groups stacked in the KiDS-1000 shear: their measured profiles, their baryons and the band statistics.

The measured profiles are in groups-stack-profiles.json beside this file, which groups_stack.py writes from the
public KiDS-1000 shear catalogue and the GAMA G3C v10 groups; that script describes the measurement. The bins
L1 to L6 are bins of group r-band luminosity that hold the fractions of the groups in the luminosity bins of
Viola et al. (2015); the bins M1 to M5 are bins of the dynamical mass proxy.

Per bin the file holds the radii R (Mpc), the excess surface density esd (Msun/pc^2) and its covariance cov,
the number of jackknife patches npatch, the mean lens redshift z, the number of lenses n_lens and the kind of
bin, L or M. It also holds the values from which the baryons are built. M200m is the mass of a single-mass fit
of the halo model of Viola et al. (2015) to the profile: an NFW halo with c200m = 0.84 times the relation of
Duffy et al. (2008), 38 percent of the groups offset by a 3D Gaussian of width 0.79 r_s, a point mass of
2.06 Mc and the linear two-halo term. Mstar_tot is 2 Msun/Lsun times the bin's mean group r-band luminosity,
and Mc the same for the mean BCG luminosity, both in Msun.

The geometry is the one in which the profiles were measured, the Planck 2018 background of background.py: flat,
Omega_m = 0.3152, H0 = 67.36 km/s/Mpc.
"""
import json
import math

import numpy as np

from mond_law import G, MSUN, MPC
from background import OMEGA_M, OMEGA_B, H

OM_G, H_G = OMEGA_M, H                           # the geometry of the measurement
OB = OMEGA_B                                     # the baryon density of the same background
FB = OB / OM_G                                   # the cosmic baryon fraction, the cap on the gas
RHO_CRIT0_H2 = 2.77536627e11                     # rho_crit today in h^2 Msun/Mpc^3
F_C = 0.84                                       # the concentration factor of the halo-model fit
R_GRID = np.geomspace(1e-4, 200.0, 1400)         # Mpc, physical
GSI = G * MSUN / MPC ** 2                        # G in SI for masses in Msun and radii in Mpc
BANDS = {'0.1-0.3': (0.1, 0.3), '0.3-1': (0.3, 1.0), '1-3': (1.0, 3.0)}


def load_profiles(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)['bins']


# ----------------------------------------------------------------------------------------------
# the halo of a bin
# ----------------------------------------------------------------------------------------------
def rho_ref(z, kind, h=H_G, om=OM_G):
    """Critical ('c') or mean ('m') density at z, Msun/Mpc^3 physical, for a flat matter + Lambda background."""
    if kind == 'c':
        return RHO_CRIT0_H2 * h * h * (om * (1 + z) ** 3 + (1 - om))
    return RHO_CRIT0_H2 * h * h * om * (1.0 + z) ** 3


def m_nfw(x):
    x = np.asarray(x, float)
    return np.log1p(x) - x / (1.0 + x)


class NFW:
    """An NFW halo of mass M within the radius of mean density delta times the reference density."""

    def __init__(self, M, delta, kind, c, z, h=H_G, om=OM_G):
        self.z, self.h, self.om = z, h, om
        self.rD = (3.0 * M / (4.0 * math.pi * delta * rho_ref(z, kind, h, om))) ** (1.0 / 3.0)
        self.c = c
        self.rs = self.rD / c
        self.norm = M / float(m_nfw(c))

    def mass(self, r, rt=None):
        r = np.asarray(r, float)
        if rt is not None:
            r = np.minimum(r, rt)
        return self.norm * m_nfw(r / self.rs)

    def radius(self, delta, kind):
        from scipy.optimize import brentq
        rho = delta * rho_ref(self.z, kind, self.h, self.om)
        f = lambda r: float(self.mass(r)) - rho * 4.0 / 3.0 * math.pi * r ** 3
        r = brentq(f, 1e-3 * self.rD, 20.0 * self.rD)
        return r, float(self.mass(r))


def duffy_mean(M200m_h, z):
    """Duffy et al. (2008) c200m, full sample, M in h^-1 Msun."""
    return 10.14 * (M200m_h / 2e12) ** (-0.081) * (1.0 + z) ** (-1.01)


def halo_200m(M200m, z):
    """r200m (Mpc), c200m and r_s of the halo model's NFW: c200m is F_C times the relation of Duffy et al."""
    r200m = (3.0 * M200m / (4.0 * math.pi * 200.0 * rho_ref(z, 'm'))) ** (1.0 / 3.0)
    c = F_C * duffy_mean(M200m * H_G, z)
    return r200m, c, r200m / c


def bin_halo(M200m, z, Mc, Mstar_tot):
    """The radii and masses of a bin's fitted halo, with the bin's stellar masses."""
    r200m, c, _ = halo_200m(M200m, z)
    halo = NFW(M200m, 200.0, 'm', c, z)
    r500c, M500c = halo.radius(500.0, 'c')
    r200c, M200c = halo.radius(200.0, 'c')
    return dict(z=z, M500c=M500c, r500c=r500c, r200c=r200c, r200m=r200m, rs=r200m / c, Mc=Mc, Mstar_tot=Mstar_tot)


# ----------------------------------------------------------------------------------------------
# the baryons of a bin
# ----------------------------------------------------------------------------------------------
def lovisari_fgas(M500):
    """Lovisari et al. 2015, A&A 573, A118, Table 4, groups and all HIFLUGCS clusters:
    log(f_gas/0.1) = 0.16 log(M500/5e13) - 0.122 (h70 = 1)."""
    return 0.1 * 10 ** (0.16 * math.log10(M500 / 5e13) - 0.122)


def beta_mass_shape(r500):
    """Enclosed mass of the beta model rho ~ (1 + (r/rc)^2)^(-3 beta/2), rc = 0.1 r500, beta = 0.6, on R_GRID,
    normalised to 1 at r500 (untruncated)."""
    rc = 0.1 * r500
    rho = (1.0 + (R_GRID / rc) ** 2) ** (-0.9)
    lr = np.log(R_GRID)
    integrand = rho * 4.0 * math.pi * R_GRID ** 3
    M = np.concatenate([[integrand[0] / 3.0],
                        integrand[0] / 3.0 + np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(lr))])
    return M / np.interp(math.log(r500), lr, M)


def stars_simple(r, r500, c500, Mc, Mtot, rt):
    """A central Hernquist profile (a = 20 kpc (Mc/1e12)^0.5) plus satellites in an NFW of c500 that bring the
    total within r500 to Mtot, the satellites truncated at rt."""
    r = np.asarray(r, float)
    a = 0.02 * math.sqrt(Mc / 1e12)
    rr = np.minimum(r, rt)
    cen = Mc * r ** 2 / (r + a) ** 2
    rs = r500 / c500
    Msat = max(Mtot - Mc * r500 ** 2 / (r500 + a) ** 2, 0.0)
    sat = Msat * m_nfw(rr / rs) / float(m_nfw(c500))
    return cen + sat


def baryons(b, fgas=1.0):
    """The enclosed baryonic mass on R_GRID (Msun): the central galaxy, the satellites and the hot gas of
    Lovisari et al. (2015) at the fitted M500c times fgas, capped at the cosmic baryon fraction minus the stars,
    in a beta model; gas and satellites are truncated at r200c."""
    r500, M500 = b['r500c'], b['M500c']
    rt = b['r200c']
    c500 = r500 / b['rs']                               # the fitted halo's scale radius
    Mtot_star = max(b['Mstar_tot'], b['Mc'])
    stars = stars_simple(R_GRID, r500, c500, b['Mc'], Mtot_star, rt)
    f = lovisari_fgas(M500) * fgas
    f_cap = max(FB - float(np.interp(math.log(r500), np.log(R_GRID), stars)) / M500, 0.0)
    f_used = min(f, f_cap)
    shape = beta_mass_shape(r500)
    gas = f_used * M500 * np.where(R_GRID <= rt, shape, np.interp(math.log(rt), np.log(R_GRID), shape))
    return stars + gas


# ----------------------------------------------------------------------------------------------
# projection and band statistics
# ----------------------------------------------------------------------------------------------
_T, _WT = np.polynomial.legendre.leggauss(200)


def esd_from_mass(R, r_grid, M_grid):
    """Excess surface density Delta Sigma(R) (Msun/Mpc^2) of a spherical distribution with enclosed mass M(<r)
    tabulated on a log grid r_grid (Mpc). Uses r = R cosh t, in which a shell at r > R puts the fraction
    1 - sqrt(1 - R^2/r^2) = 1 - tanh t inside the cylinder:
      Sigma(R)   = (1/(2 pi R)) int_0^inf m'(R cosh t)/cosh t dt,
      Mcyl(<R)   = M(<R) + int_0^inf m'(R cosh t) (1 - tanh t) R sinh t dt."""
    R = np.atleast_1d(np.asarray(R, float))
    lr = np.log(r_grid)
    dMdlnr = np.gradient(M_grid, lr)
    mprime = lambda r: np.interp(np.log(r), lr, dMdlnr, left=0.0, right=0.0) / r
    tmax = np.arccosh(r_grid[-1] / R)
    out = np.empty(R.size)
    for i, Ri in enumerate(R):
        t = 0.5 * tmax[i] * (_T + 1.0)
        wt = 0.5 * tmax[i] * _WT
        rr = Ri * np.cosh(t)
        mp = mprime(rr)
        Sig = np.sum(wt * mp / np.cosh(t)) / (2.0 * math.pi * Ri)
        one_minus_tanh = 2.0 / (1.0 + np.exp(2.0 * t))        # 1 - tanh t without cancellation
        Mcyl = np.interp(math.log(Ri), lr, M_grid) + np.sum(wt * mp * one_minus_tanh * Ri * np.sinh(t))
        out[i] = Mcyl / (math.pi * Ri * Ri) - Sig
    return out


def hartlap(n, p):
    """The Hartlap et al. (2007) factor on the inverse of a covariance from n jackknife patches, p points."""
    return (n - p - 2.0) / (n - 1.0)


def band_stats(d, C, npatch, m, R):
    """Per band: model over data from an amplitude fit with the covariance (1/alpha), its error, and the
    chi-square of the model as computed."""
    out = {}
    for name, (lo, hi) in BANDS.items():
        sel = (R >= lo) & (R < hi)
        ds, ms = d[sel], m[sel]
        Cs = C[np.ix_(sel, sel)]
        Ci = np.linalg.inv(Cs) * hartlap(npatch, int(sel.sum()))
        a = float(ms @ Ci @ ds / (ms @ Ci @ ms))
        ae = float(1.0 / math.sqrt(ms @ Ci @ ms))
        r = ds - ms
        out[name] = dict(ratio=1.0 / a, ratio_err=ae / a ** 2, chi2=float(r @ Ci @ r), n=int(sel.sum()))
    return out
