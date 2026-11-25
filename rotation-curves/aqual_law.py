"""Shared pieces of the AQUAL refits: the two laws and the axisymmetric AQUAL solver with the law and the scale a0 as
parameters.

The solver is that of aqual-disk-solver.py in this folder (grid, fast Poisson inversion, galaxy models), loaded
unchanged as the module T16. LawSolver repeats its QUMOND step and its AQUAL curl-correction iteration line for line,
with nu, mu and a0 passed in instead of the module's own law (the empirical function at a0 = 1.2e-10 m/s^2), and with
an optional warm start.

The laws (codes used in the tables and options of the refits):
  T11, the empirical function of McGaugh, Lelli and Schombert: nu(y) = 1/(1 - exp(-sqrt(y))); mu(x) from its inverse,
      as in aqual-disk-solver.py (MuFromNu).
  T3, the boost law of Paper I, Eq. (8) (mu_boost in refit_core.py):
      mu(x) = E/(E + delta), E = exp(lambda x) - 1, lambda = ln(1 + delta);
      nu(y) = 1/mu(x) on y = x mu(x), tabulated on 20,001 nodes of x (no root finding: x is the table variable).
Units: accelerations in (km/s)^2/kpc, as in aqual-disk-solver.py; CONV converts to m/s^2.
"""
import importlib.util, math, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_t16():
    """Import the disk solver (aqual-disk-solver.py) without letting it read this script's command line (it reads
    --hgas, --drdiv, --dzdiv and --profile at import); its defaults give every deposited result."""
    spec = importlib.util.spec_from_file_location("t16solver", os.path.join(HERE, "aqual-disk-solver.py"))
    mod = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


T16 = _load_t16()            # the disk solver module, aqual-disk-solver.py
G = T16.G                    # kpc (km/s)^2 / Msun
CONV = T16.CONV              # (km/s)^2/kpc -> m/s^2
A0_T16 = T16.A0              # 1.2e-10 m/s^2 in solver units, the scale of aqual-disk-solver.py


def a0_units(a0_si):
    return a0_si / CONV


# ------------------------------------------------------------------ the laws
class LawT11:
    """The empirical function (T11). mu is the disk solver's own table (MuFromNu), so that LawSolver reproduces
    aqual-disk-solver.py at its scale."""
    code = "T11"

    def __init__(self):
        self.shape = None
        self._mu = T16.MuFromNu(T16.nu_mcg)

    def nu(self, y):
        return T16.nu_mcg(y)

    def mu(self, x):
        return self._mu(x)

    def label(self):
        return "T11"


class LawT3:
    """The boost law (T3) at free volume delta: mu(x) = E/(E + delta), E = expm1(lambda x), lambda = ln(1+delta).
    nu(y) is tabulated against log10 y on x nodes from 1e-12 to 1e8 (20,001 nodes, 0.001 dex); beyond the table,
    nu -> sqrt(delta/(lambda y)) (the deep limit, x mu(x) -> (lambda/delta) x^2) and nu -> 1."""
    code = "T3"

    def __init__(self, delta):
        self.shape = float(delta)
        self.delta = float(delta)
        self.lam = math.log1p(self.delta)
        lx = np.linspace(-12.0, 8.0, 20001)
        x = 10.0 ** lx
        m = self.mu(x)
        y = x * m
        self.ly = np.log10(y)
        self.lnu = np.log10(1.0 / m)
        if np.any(np.diff(self.ly) <= 0):
            raise ValueError("x mu(x) not increasing")

    def mu(self, x):
        x = np.maximum(np.asarray(x, float), 1e-300)
        u = self.lam * x
        E = np.expm1(np.minimum(u, 40.0))
        return np.where(u > 40.0, 1.0, E / (E + self.delta))

    def nu(self, y):
        y = np.maximum(np.asarray(y, float), 1e-300)
        ly = np.log10(y)
        v = 10.0 ** np.interp(ly, self.ly, self.lnu)
        v = np.where(ly < self.ly[0], np.sqrt(self.delta / (self.lam * y)), v)
        return np.where(ly > self.ly[-1], 1.0, v)

    def label(self):
        return f"T3 delta={self.delta:g}"


def make_law(code, shape=None):
    if code == "T11":
        return LawT11()
    if code == "T3":
        return LawT3(shape)
    raise ValueError(code)


# ------------------------------------------------------------------ the solver with the law and a0 as parameters
class LawSolver(T16.Solver):
    """The disk solver's Solver (Newton and projections unchanged) with QUMOND and AQUAL for a given law and a0
    (solver units). The iteration is that of aqual-disk-solver.py: h = mu(|g|/a0) g; remove the divergence error of h by a gradient;
    g <- curl-free part of nu(|h'|/a0) h'. Converged when the midplane radial field changes by less than tol."""

    def __init__(self, grid, law, a0):
        super().__init__(grid)
        self.law, self.a0 = law, a0

    def set_law(self, law, a0):
        self.law, self.a0 = law, a0

    def qumond(self, gNR, gNZ, maskR=None, maskZ=None):
        mR, mZ = T16.face_magnitudes(gNR, gNZ)
        nR, nZ = self.law.nu(mR / self.a0), self.law.nu(mZ / self.a0)
        gR, gZ, phi = self.project(nR * gNR, nZ * gNZ)
        return gR, gZ

    def aqual(self, gNR, gNZ, maskR=None, maskZ=None, tol=1e-8, itmax=300, start=None):
        if start is None:
            gR, gZ = self.qumond(gNR, gNZ)
        else:
            gR, gZ = start
        tNz, tNr = T16.face_tangentials(gNR, gNZ)
        nu, mu, a0 = self.law.nu, self.law.mu, self.a0
        for it in range(itmax):
            tgz, tgr = T16.face_tangentials(gR, gZ)
            muR, muZ = mu(np.sqrt(gR ** 2 + tgz ** 2) / a0), mu(np.sqrt(gZ ** 2 + tgr ** 2) / a0)
            hR, hZ = muR * gR, muZ * gZ
            hR[-1] = gNR[-1]; hZ[:, -1] = gNZ[:, -1]
            cR, cZ, chi = self.project(hR - gNR, hZ - gNZ)
            hR2 = hR - cR; hZ2 = hZ - cZ
            hR2[-1] = gNR[-1]; hZ2[:, -1] = gNZ[:, -1]
            tcz, tcr = T16.face_tangentials(cR, cZ)
            thz = muR * tgz - tcz; thr = muZ * tgr - tcr
            thz[-1] = tNz[-1]; thr[:, -1] = tNr[:, -1]
            nR = nu(np.sqrt(hR2 ** 2 + thz ** 2) / a0); nZ = nu(np.sqrt(hZ2 ** 2 + thr ** 2) / a0)
            gR2, gZ2, phi = self.project(nR * hR2, nZ * hZ2)
            dch = np.max(np.abs(gR2[:, 0] - gR[:, 0]) / np.maximum(np.abs(gR2[:, 0]), 1e-300))
            gR, gZ = gR2, gZ2
            if dch < tol:
                break
        self.aqual_iters = it + 1
        return gR, gZ

    def aqual_residual(self, gR, gZ, gNR, gNZ):
        mR, mZ = T16.face_magnitudes(gR, gZ)
        muR, muZ = self.law.mu(mR / self.a0), self.law.mu(mZ / self.a0)
        hR, hZ = muR * gR, muZ * gZ
        hR[-1] = gNR[-1]; hZ[:, -1] = gNZ[:, -1]
        r = T16.div_faces(self.g, hR - gNR, hZ - gNZ)
        return np.abs(r).sum() / np.abs(T16.div_faces(self.g, gNR, gNZ)).sum()
