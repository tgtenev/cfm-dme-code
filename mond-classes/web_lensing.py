r"""The lensing of the web: the matter power of a CAMB template, Limber integrals for the CMB lensing convergence
and for cosmic shear, and the Planck 2018, ACT DR6 and KiDS-1000 data with their likelihoods.

The template is a CAMB run (Lewis, Challinor and Lasenby 2000) at the Planck 2018 TT,TE,EE+lowE+lensing
parameters (Planck Collaboration 2020, A&A 641, A6, Table 2): H0 = 67.36, omega_b = 0.02237,
omega_c = 0.1200, tau = 0.0544, ln(1e10 A_s) = 3.044, n_s = 0.9649, one massive neutrino of 0.06 eV,
with the nonlinear power of HMcode-2020 (Mead et al. 2021); the parameters are those of background.py. The
distances and the lensing kernels use a flat background, E^2 = (1 - Omega_m) + Omega_m (1+z)^3 with no radiation,
at the Omega_m that the caller passes, which is background.OMEGA_M.

The data are public releases, read from the data folder:
  <data>/planck2018-lensing/  the Planck 2018 lensing bandpowers, CMB-marginalized, conservative range
      8 <= L <= 400 (github.com/CobayaSampler/planck_supp_data_and_covmats, lensing/2018);
  <data>/act-dr6/v1.2/        the ACT DR6 lensing likelihood, baseline 40 < L < 763, lensing only
      (Qu et al. 2024; Madhavacheril et al. 2024; LAMBDA);
  <data>/kids1000/xipm_k1000.fits  the KiDS-1000 cosmic-shear release (Asgari et al. 2021;
      kids.strw.leidenuniv.nl/DR4).
Distances are in Mpc/h and wavenumbers in h/Mpc inside the Limber integrals, unless stated.
"""
import math
import os

import numpy as np
import camb
from scipy import special

C_KMS = 299792.458
C_H0 = C_KMS / 100.0           # c/H0 in Mpc/h
MPC_M = 3.0856775814913673e22  # m
Z_STAR = 1089.92               # Planck 2018 z_* (A&A 641, A6, Table 2)
ARCMIN = math.pi / 180.0 / 60.0

from background import P18  # noqa: E402  (Planck 2018 TT,TE,EE+lowE+lensing, A&A 641, A6, Table 2)


# ----------------------------------------------------------------------------------------------
# the CAMB template
# ----------------------------------------------------------------------------------------------
def default_z_cmb():
    return np.unique(np.concatenate([np.linspace(0.0, 4.0, 81), np.geomspace(4.25, 1100.0, 60)]))


class Template:
    """A CAMB run of the standard model: P_NL(k, z) and P_lin(k, z) in h units."""

    def __init__(self, zs, kmax=100.0):
        As = math.exp(P18['ln10As']) * 1e-10
        pars = camb.set_params(H0=P18['H0'], ombh2=P18['ombh2'], omch2=P18['omch2'], mnu=P18['mnu'],
                               omk=0.0, tau=P18['tau'], As=As, ns=P18['ns'], WantCls=False,
                               lmax=4000, lens_potential_accuracy=0)
        pars.NonLinearModel.set_params(halofit_version='mead2020')
        pars.set_matter_power(redshifts=np.sort(np.asarray(zs))[::-1], kmax=kmax, nonlinear=True)
        res = camb.get_results(pars)
        self.PK = res.get_matter_power_interpolator(nonlinear=True, var1='delta_tot', var2='delta_tot',
                                                    hubble_units=True, k_hunit=True, extrap_kmax=1e4)
        self.PL = res.get_matter_power_interpolator(nonlinear=False, var1='delta_tot', var2='delta_tot',
                                                    hubble_units=True, k_hunit=True, extrap_kmax=1e4)
        self.h = pars.H0 / 100.0
        self.zmax = float(np.max(zs))

    def P(self, z, k, nonlinear=True):
        """P(k, z) for paired arrays, (Mpc/h)^3; k clipped into the interpolator's range."""
        PK = self.PK if nonlinear else self.PL
        z = np.asarray(z, float)
        k = np.asarray(k, float)
        shape = np.broadcast(z, k).shape
        zz = np.broadcast_to(z, shape).ravel()
        kk = np.broadcast_to(k, shape).ravel()
        kc = np.clip(kk, PK.kmin * 1.0001, 9.9e3)
        out = PK.P(np.minimum(zz, self.zmax), kc, grid=False)
        lo = kk < PK.kmin * 1.0001          # below kmin: P ~ k^n_s
        out[lo] *= (kk[lo] / kc[lo]) ** P18['ns']
        out[kk > 9.9e3] = 0.0
        return out.reshape(shape)


# ----------------------------------------------------------------------------------------------
# the background and the Limber engine
# ----------------------------------------------------------------------------------------------
class Geometry:
    """A flat background, E^2 = (1 - om) + om (1+z)^3, no radiation; distances in Mpc/h."""

    def __init__(self, om, zmax=1200.0, n=20000):
        self.om = om
        x = np.linspace(0.0, math.log1p(zmax), n)
        z = np.expm1(x)
        integrand = (1.0 + z) / self.E(z)            # dchi/dx = C_H0 (1+z)/E
        chi = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(x))])
        self._x, self._chi = x, C_H0 * chi
        self.chi_star = float(self.chi(Z_STAR))

    def E(self, z):
        return np.sqrt((1.0 - self.om) + self.om * (1.0 + np.asarray(z)) ** 3)

    def chi(self, z):
        return np.interp(np.log1p(np.asarray(z, float)), self._x, self._chi)

    def dchidz(self, z):
        return C_H0 / self.E(z)


def trapz_weights(x):
    w = np.zeros_like(x)
    d = np.diff(x)
    w[:-1] += 0.5 * d
    w[1:] += 0.5 * d
    return w


class LimberGrid:
    """A precomputed (ell, z) grid: k = (ell + 1/2)/chi(z)."""

    def __init__(self, geom, ells, z):
        self.geom = geom
        self.ells = np.asarray(ells, float)
        self.z = np.asarray(z, float)
        self.chi = geom.chi(self.z)
        self.k = (self.ells[:, None] + 0.5) / self.chi[None, :]
        self.wz = trapz_weights(self.z) * geom.dchidz(self.z)

    def power(self, tpl):
        """The template's nonlinear power on the grid."""
        zz = np.broadcast_to(self.z[None, :], self.k.shape)
        return tpl.P(zz, self.k, nonlinear=True)

    def cl(self, P, kern):
        """kern: [nz] or [nz, npairs] product of kernels divided by chi^2."""
        kern = np.asarray(kern)
        if kern.ndim == 1:
            return P @ (kern * self.wz)
        return P @ (kern * self.wz[:, None])


def cmb_lensing_z(n=1400):
    x = np.linspace(math.log1p(1e-4), math.log1p(Z_STAR * 0.99999), n)
    return np.expm1(x)


def cmb_kernel_sq(geom, z):
    """[W_kappa(chi)]^2 / chi^2, with W = 1.5 Om (H0/c)^2 chi (1+z) (chi* - chi)/chi*."""
    chi = geom.chi(z)
    W = 1.5 * geom.om / C_H0 ** 2 * chi * (1.0 + z) * (geom.chi_star - chi) / geom.chi_star
    return W * W / (chi * chi)


# ----------------------------------------------------------------------------------------------
# cosmic shear
# ----------------------------------------------------------------------------------------------
def nz_normalize(zmid, nz):
    nz = np.clip(np.asarray(nz, float), 0.0, None)
    width = np.gradient(zmid)
    return nz / np.sum(nz * width), width


def shear_q(geom, z, zmid, nz, sub=5):
    """Lensing efficiency q(chi) for a histogram n(z) at zmid with uniform sub-sampling."""
    n, width = nz_normalize(zmid, nz)
    off = (np.arange(sub) + 0.5) / sub - 0.5
    zs = (zmid[:, None] + off[None, :] * width[:, None]).ravel()
    ws = (np.repeat(n * width, sub) / sub)
    keep = zs > 0
    zs, ws = zs[keep], ws[keep]
    chis = geom.chi(zs)
    chi = geom.chi(z)
    frac = np.clip(1.0 - chi[:, None] / chis[None, :], 0.0, None)
    eff = frac @ ws
    return 1.5 * geom.om / C_H0 ** 2 * chi * (1.0 + z) * eff


class Hankel:
    """xi_+/-(theta) = int dl l/(2 pi) J_0/4(l theta) C_l on a log grid. The integrand is tapered by
    exp(-(l theta/150)^4), which removes the under-sampled oscillating tail; against a grid of 300,000
    points the transform agrees to 4e-5 at the KiDS-1000 angles."""

    def __init__(self, theta_arcmin, ell_nodes, lmin=1.0, lmax=3.0e5, n=12000, taper=150.0):
        self.theta = np.asarray(theta_arcmin, float) * ARCMIN
        self.lf = np.geomspace(lmin, lmax, n)
        dln = math.log(self.lf[1] / self.lf[0])
        x = self.lf[None, :] * self.theta[:, None]
        base = self.lf ** 2 / (2.0 * math.pi) * dln
        tap = np.exp(-(x / taper) ** 4)
        self.J0 = special.j0(x) * base[None, :] * tap
        self.J4 = special.jv(4, x) * base[None, :] * tap
        lnn = np.log(ell_nodes)
        t = np.interp(np.log(self.lf), lnn, np.arange(len(lnn)))
        self.i0 = np.clip(np.floor(t).astype(int), 0, len(lnn) - 2)
        self.w1 = t - self.i0

    def fine(self, cl_nodes):
        """Log-log linear interpolation of C_ell from the nodes to the fine grid; cl_nodes [nl, np]."""
        lc = np.log(np.clip(cl_nodes, 1e-300, None))
        v = (1.0 - self.w1)[:, None] * lc[self.i0] + self.w1[:, None] * lc[self.i0 + 1]
        return np.exp(v)

    def xi(self, cl_nodes):
        cf = self.fine(cl_nodes)
        return self.J0 @ cf, self.J4 @ cf


def load_kids1000(path):
    """xi_+/xi_- with covariance and n(z) from the KiDS-1000 2pt FITS file, with the survey's scale cuts:
    0.5 to 300 arcmin for xi_+ and 4 to 300 arcmin for xi_-."""
    from astropy.io import fits
    with fits.open(path) as f:
        xp = f['xip'].data
        xm = f['xim'].data
        cov_full = np.array(f['COVMAT'].data, float)
        nzt = f['nz_source'].data
        nb = len([c for c in nzt.columns.names if c.startswith('BIN')])
        zmid = np.array(nzt['Z_MID'], float)
        nzs = [np.array(nzt[f'BIN{i}'], float) for i in range(1, nb + 1)]
        h = f['COVMAT'].header
        starts = {h[f'NAME_{i}'].lower(): int(h[f'STRT_{i}']) for i in range(10) if f'NAME_{i}' in h}
        rows = []
        for tag, tab in (('xip', xp), ('xim', xm)):
            s0 = starts[tag]
            for j, r in enumerate(tab):
                rows.append((tag, int(r['BIN1']), int(r['BIN2']), float(r['ANG']), float(r['VALUE']), s0 + j))
    keep = []
    for (tag, b1, b2, ang, val, ic) in rows:
        pair = (min(b1, b2), max(b1, b2))
        ok = (0.5 <= ang <= 300.0) if tag == 'xip' else (4.0 <= ang <= 300.0)
        if ok:
            keep.append((tag, pair, ang, val, ic))
    idx = np.array([k[4] for k in keep])
    cov = cov_full[np.ix_(idx, idx)]
    pairs = sorted({k[1] for k in keep})
    thetas = np.array(sorted({k[2] for k in keep}))
    return dict(rows=keep, cov=cov, data=np.array([k[3] for k in keep]), zmid=zmid, nzs=nzs, pairs=pairs,
                thetas=thetas)


class ShearModel:
    """The theory xi_+/xi_- data vector of the survey on a fixed (ell, z) grid, intrinsic alignment off."""

    def __init__(self, sd, geom):
        self.ells = np.geomspace(1.0, 3.0e5, 150)
        self.z = np.linspace(1e-3, 6.0, 500)
        self.grid = LimberGrid(geom, self.ells, self.z)
        q = [shear_q(geom, self.z, sd['zmid'], sd['nzs'][i]) for i in range(len(sd['nzs']))]
        chi2 = self.grid.chi ** 2
        self.kGG = np.stack([q[a - 1] * q[b - 1] / chi2 for (a, b) in sd['pairs']], axis=1)
        self.hank = Hankel(sd['thetas'], self.ells)
        pindex = {p: i for i, p in enumerate(sd['pairs'])}
        tindex = {t: i for i, t in enumerate(sd['thetas'])}
        self.sel = [(0 if r[0] == 'xip' else 1, tindex[r[2]], pindex[r[1]]) for r in sd['rows']]

    def vector(self, P):
        cl = np.clip(self.grid.cl(P, self.kGG), 1e-300, None)
        xp, xm = self.hank.xi(cl)
        out = np.empty(len(self.sel))
        for i, (t, it, ip) in enumerate(self.sel):
            out[i] = xp[it, ip] if t == 0 else xm[it, ip]
        return out


# ----------------------------------------------------------------------------------------------
# CMB lensing likelihoods
# ----------------------------------------------------------------------------------------------
class PlanckLensing:
    """Planck 2018 lensing, conservative 8 <= L <= 400, CMB-marginalized, in the cobaya format of
    planck_supp_data_and_covmats/lensing/2018. The bandpowers are binned [L(L+1)]^2 C_L^pp/2pi with the
    linear N1 correction M (C - C_fid) of the CMB-marginalized likelihood."""
    stem = 'smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8'

    def __init__(self, ddir):
        bp = np.loadtxt(os.path.join(ddir, self.stem + '_bandpowers.dat'))
        self.data = bp[:, 4]
        self.cov = np.loadtxt(os.path.join(ddir, self.stem + '_CMBmarged_cov.dat'))
        self.cinv = np.linalg.inv(self.cov)
        self.fidcorr = np.loadtxt(os.path.join(
            ddir, self.stem + '_CMBmarged_lensing_fiducial_correction.dat'))[:, 1]
        self.W = np.zeros((9, 3001))
        self.M = np.zeros((9, 3001))
        for b in range(9):
            w = np.loadtxt(os.path.join(ddir, 'win', 'window%d.dat' % (b + 1)))
            self.W[b, w[:, 0].astype(int)] = w[:, 1]
            m = np.loadtxt(os.path.join(ddir, 'lensdelta', 'window%d.dat' % (b + 1)))
            self.M[b, m[:, 0].astype(int)] = m[:, 1]

    @staticmethod
    def pp_from_kk(ckk):
        return ckk * 2.0 / np.pi          # [L(L+1)]^2 C^pp / 2pi = (2/pi) C^kk

    def binned(self, ckk_full):
        """ckk_full: C_L^kk for L = 0..3000."""
        pp = self.pp_from_kk(ckk_full[:3001])
        return self.W @ pp + self.M @ pp - self.fidcorr

    def template(self, ckk_full):
        pp = self.pp_from_kk(ckk_full[:3001])
        return (self.W + self.M) @ pp

    def chi2(self, ckk_full):
        d = self.data - self.binned(ckk_full)
        return float(d @ self.cinv @ d)

    def amplitude(self, ckk_full):
        T = self.template(ckk_full)
        y = self.data + self.fidcorr
        F = T @ self.cinv @ T
        A = (T @ self.cinv @ y) / F
        return A, 1.0 / np.sqrt(F)


class ACTLensing:
    """ACT DR6 lensing bandpowers, baseline 40 < L < 763, lensing only (CMB-marginalized covariance, no
    likelihood corrections), as in the variant act_baseline of act_dr6_lenslike."""

    def __init__(self, ddir):
        start, end = 2, -6
        y = np.loadtxt(os.path.join(ddir, 'clkk_bandpowers_act.txt'))
        nb_tot = y.size
        B = np.loadtxt(os.path.join(ddir, 'binning_matrix_act.txt'))
        self.data = y[start:end].copy()
        self.B = B[start:end, :]
        fcov = np.loadtxt(os.path.join(ddir, 'covmat_act_cmbmarg.txt'))
        sel = np.s_[nb_tot + end:nb_tot]
        cov = np.delete(np.delete(fcov, sel, 0), sel, 1)
        sel = np.s_[:start]
        cov = np.delete(np.delete(cov, sel, 0), sel, 1)
        nsims = 796
        self.hartlap = (nsims - self.data.size - 2.0) / (nsims - 1.0)
        self.cinv = np.linalg.inv(cov) * self.hartlap
        self.nL = self.B.shape[1]

    def binned(self, ckk_full):
        return self.B @ ckk_full[:self.nL]

    def chi2(self, ckk_full):
        d = self.data - self.binned(ckk_full)
        return float(d @ self.cinv @ d)

    def amplitude(self, ckk_full):
        T = self.binned(ckk_full)
        F = T @ self.cinv @ T
        return (T @ self.cinv @ self.data) / F, 1.0 / np.sqrt(F)


# ----------------------------------------------------------------------------------------------
# Newtonian peculiar accelerations of a linear field (SI accelerations, Mpc and 1/Mpc)
# ----------------------------------------------------------------------------------------------
def accel_per_lnk(Omega, z, k, Plin, H0_kms):
    """Per-ln-k rms physical peculiar acceleration (m s^-2) of a component with density parameter Omega whose
    linear contrast has power Plin (Mpc^3) at comoving k (1/Mpc): g = (3/2) H0^2 Omega (1+z)^2 Delta(k)/k."""
    H0s = H0_kms / (MPC_M / 1.0e3)        # 1/s
    Delta = np.sqrt(k ** 3 * Plin / (2.0 * np.pi ** 2))
    return 1.5 * H0s ** 2 * Omega * (1.0 + z) ** 2 * Delta / k * MPC_M


def accel_rms(Omega, z, ks, Plin_row, H0_kms, R):
    """3D rms peculiar acceleration (m s^-2) of the linear field smoothed with a top-hat of comoving radius
    R (Mpc)."""
    H0s = H0_kms / (MPC_M / 1.0e3)
    x = ks * R
    W = 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3
    I = np.trapezoid(Plin_row * W ** 2 * ks, np.log(ks)) / (2.0 * np.pi ** 2)   # Mpc^2
    return 1.5 * H0s ** 2 * Omega * (1.0 + z) ** 2 * np.sqrt(I) * MPC_M
