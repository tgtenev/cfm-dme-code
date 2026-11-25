"""The exact AQUAL solutions in the disks: an axisymmetric solver of the AQUAL and QUMOND field equations on a stretched
(R, z) grid, its validation on the Kuzmin disk, and the departure of the midplane field from the algebraic relation at
the SPARC points.

WHAT IT COMPUTES
  Each SPARC galaxy with points in sample A (147 galaxies, 2,696 points; sample A is defined in aqual-disk-densities.py)
  gets a model of finite thickness built from the two files of aqual-disk-densities.py. Newton's equation, QUMOND and
  AQUAL are solved for it with the empirical function at a0 = 1.2e-10 m/s^2 and the 3.6 micron mass-to-light ratios at
  their prior centres, Ups_disk = 0.5 and Ups_bulge = 0.7, and the midplane radial fields are read at the data radii. The summary
  gives the departure of the full AQUAL solution from the algebraic relation g = nu(g_bar/a0) g_bar, by bins of
  y = g_bar/a0, and the same for QUMOND.
  The module also supplies the grid, the solver and the galaxy models to the refits (through aqual_law.py). The refits
  solve the same models, with the laws and scales they fit, for the 153 galaxies and 3,168 points of Paper I, Sect. 4.1.

RESULTS SUPPORTED
  Paper II, Sect. 6.1 (the disk departure): the solver and the galaxy models behind its numbers, which the refits
  compute on these models (the median departure of about 3 percent, the 0.06 dex rms, the 18 percent where the vertical
  field exceeds the radial one, and the fits of the full solution). This script's own summary shows the departure at
  a0 = 1.2e-10 m/s^2 and that QUMOND reproduces it. Paper I, Sect. 4.2: the fits of the full solution (Delta chi^2 =
  246 +/- 143 and 286 +/- 142, and the lead of MOND's simple function by 39.3 +/- 14.7) use the same solver and models.

INPUTS
  rotation-curves/aqual-disk-points.csv and rotation-curves/aqual-disk-gas-profiles.csv (from aqual-disk-densities.py)

OUTPUTS
  rotation-curves/aqual-disk-solver-validate-results.txt   part validate
  rotation-curves/aqual-disk-pred-0.csv, -pred-1.csv       part galaxies, one file per half of the galaxies
  rotation-curves/aqual-disk-solver-results.txt            part summary
  Columns of the prediction files, one row per point of the 147 galaxies: galaxy, R (kpc); selA (1 for sample A), y and
  res, as in the points file; rho_kq, the midplane density (Msun/pc^3); V2N_over_V2bar, the model's Newtonian V^2 over
  SPARC's V_bar^2; boostQ and boostA, log10 of the QUMOND and AQUAL midplane radial field over the model's Newtonian
  one; hz, the stellar scale height (kpc); itA, the number of AQUAL iterations.

COMMANDS, from the repository root, in this order
  python rotation-curves/aqual-disk-solver.py --part validate --out rotation-curves/aqual-disk-solver-validate-results.txt
  python rotation-curves/aqual-disk-solver.py --part galaxies --slice 0/2 --pred rotation-curves/aqual-disk-pred-0.csv
  python rotation-curves/aqual-disk-solver.py --part galaxies --slice 1/2 --pred rotation-curves/aqual-disk-pred-1.csv
  python rotation-curves/aqual-disk-solver.py --part summary --pred rotation-curves/aqual-disk-pred-0.csv,rotation-curves/aqual-disk-pred-1.csv --out rotation-curves/aqual-disk-solver-results.txt
  The two halves can run at the same time. --data DIR reads the two input files from DIR (default: rotation-curves).
  --only NAME1,NAME2 runs the listed galaxies only, as a quick test. Needs numpy.
  Run times: validate about 1 minute, each half of the galaxies about 1 minute, summary a few seconds.
  Options of the galaxy model, read at import (the defaults give every deposited result): --hgas H, the scale height
  of the gas layer in kpc (0.2); --drdiv N and --dzdiv N, the cell sizes dR0 = R_d/N and dz0 = h_min/N (8);
  --profile sech2, stellar layers in sech^2(z/z0) with z0 = 2 h_z instead of exponential.

METHOD
  Finite volumes on a cylindrical grid, z >= 0 with reflection symmetry. The cells are uniform over the disk and the
  layer (dR0 = min(R_d/8, R_last/100), refined toward the innermost data point; dz0 = min(h_z, h_gas)/8) and grow
  geometrically (factor 1.07) to an outer box of 300 times the last data radius. The potential sits at cell centres,
  the fields are face-normal components. The constant-coefficient Laplacian is inverted exactly by fast
  diagonalization (eigenvectors of the 1D z-operator, a tridiagonal solve in R per mode).
  Newton: lap(Phi_N) = 4 pi G rho, with Dirichlet values from the monopole and quadrupole at the box.
  QUMOND (Milgrom 2010, as Lughausen et al. 2013, Eqs. 1 and 3, state it): g = curl-free part of nu(|g_N|/a0) g_N.
  AQUAL (Bekenstein and Milgrom 1984): div[mu(|g|/a0) g] = div(g_N), solved by a curl-correction iteration,
    h = mu(|g|) g;  h' = h - grad(chi), lap(chi) = div(h - g_N);  g <- curl-free part of nu(|h'|) h',
  with the tangential parts taken so that the round trip is exact. At convergence the discrete AQUAL flux equation
  holds to about 1e-5 of the source.
  Interpolating function: the empirical function of McGaugh, Lelli and Schombert, nu(y) = 1/(1 - exp(-sqrt(y))), at
  a0 = 1.2e-10 m/s^2, and for AQUAL its inverse mu(x) = 1/nu(y(x)), x = y nu(y).
  Galaxy models: SPARC 3.6 micron disks at Ups_disk = 0.5 (log-linear between the data radii, exponential with R_d
  outside, cut at R_last + 8 R_d), vertically exponential with h_z = 1.12 R_d/8.06 (Kregel, van der Kruit and Freeman
  2005, Table 1); the gas of aqual-disk-gas-profiles.csv (deconvolved for a layer of 0.1 kpc) placed in an exponential
  layer of h = 0.2 kpc (Angus et al. 2015, Sect. 2.1.1); bulges as Hernquist spheres fitted to Ups_bulge R V_bul^2/G
  with Ups_bulge = 0.7.

VALIDATION (part validate)
  The Kuzmin disk, whose MOND field is exactly nu(|g_N|/a0) g_N in both formulations (Famaey and McGaugh 2012, Sect.
  6.1.2; Brada and Milgrom 1995). The sheet is spread over the first cell row, so the midplane field converges to the
  razor-thin value linearly in dz0, and above the sheet the comparison is direct. A thick exponential disk checks the
  Newtonian solver against the ring-potential quadrature of aqual-disk-densities.py, and the QUMOND and AQUAL boosts
  against a grid with half the cell size.
"""
import csv, math, os, sys, time
import numpy as np

T00 = time.time()
G = 4.30091e-6            # kpc (km/s)^2 / Msun
CONV = 3.24078e-14        # (km/s)^2/kpc -> m/s^2
A0 = 1.2e-10 / CONV       # (km/s)^2/kpc
Q_CEN, Q_SIG = 8.06, 0.137   # flattening h_R/h_z of edge-on disks and its scatter in dex (aqual-disk-densities.py)
H_GAS = float(sys.argv[sys.argv.index("--hgas") + 1]) if "--hgas" in sys.argv else 0.20   # gas layer scale height (kpc)
UPS_D, UPS_B = 0.5, 0.7
DR_DIV = float(sys.argv[sys.argv.index("--drdiv") + 1]) if "--drdiv" in sys.argv else 8.0    # dR0 = R_d/DR_DIV
DZ_DIV = float(sys.argv[sys.argv.index("--dzdiv") + 1]) if "--dzdiv" in sys.argv else 8.0    # dz0 = h_min/DZ_DIV

ARGS = sys.argv
def arg(name, default=None):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default
OUT = arg("--out")
PROFILE = arg("--profile", "exp")          # vertical profile of the stellar disk: exp (baseline) or sech2 (z0 = 2 h_z)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))   # default directory of the two input files
_lines = []
def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


# ================================================================== interpolating function
def nu_mcg(y):
    y = np.maximum(np.asarray(y, float), 1e-300)
    return 1.0 / (-np.expm1(-np.sqrt(y)))


class MuFromNu:
    """mu(x) = 1/nu(y) on x = y nu(y), tabulated against log10 x (20,001 nodes from 1e-12 to 1e8 in y); beyond the
    table mu = 1 (x large) or mu = x (x small, the deep limit of the empirical function)."""
    def __init__(self, nu, lo=-12.0, hi=8.0, n=20001):
        ly = np.linspace(lo, hi, n)
        yy = 10.0 ** ly
        self.lx = np.log10(yy * nu(yy)); self.mu = 1.0 / nu(yy); self.nu = nu
    def __call__(self, x):
        x = np.maximum(np.asarray(x, float), 1e-300)
        lx = np.log10(x)
        m = np.interp(lx, self.lx, self.mu, right=1.0)
        return np.where(lx < self.lx[0], x, m)


MU = MuFromNu(nu_mcg)


# ================================================================== grid and fast Poisson solver
def faces_stretched(d0, x1, xmax, ratio, dstart=None, ratio_in=1.04):
    """Faces from 0: cells growing from dstart by ratio_in up to d0 (if dstart < d0), uniform d0 up to x1, then
    growing by ratio up to xmax."""
    f = [0.0]
    if dstart is not None and dstart < d0:
        d = dstart
        while d < d0 and f[-1] < x1:
            f.append(f[-1] + d)
            d *= ratio_in
    while f[-1] < x1 - 1e-12:
        f.append(f[-1] + d0)
    d = d0
    while f[-1] < xmax:
        d *= ratio
        f.append(f[-1] + d)
    return np.array(f)


class Grid:
    def __init__(self, Rf, zf):
        self.Rf, self.zf = np.asarray(Rf, float), np.asarray(zf, float)
        self.Rc = 0.5 * (self.Rf[1:] + self.Rf[:-1]); self.zc = 0.5 * (self.zf[1:] + self.zf[:-1])
        self.NR, self.NZ = len(self.Rc), len(self.zc)
        self.dz = np.diff(self.zf); self.dR = np.diff(self.Rf)
        self.dA = 0.5 * (self.Rf[1:] ** 2 - self.Rf[:-1] ** 2)
        self.V = self.dA[:, None] * self.dz[None, :]
        self.AR = self.Rf[1:, None] * self.dz[None, :]
        self.AZ = np.repeat(self.dA[:, None], self.NZ, axis=1)
        self.dRc = np.diff(self.Rc); self.dzc = np.diff(self.zc)
        self.dRb = self.Rf[-1] - self.Rc[-1]; self.dzb = self.zf[-1] - self.zc[-1]

    def T1(self, x_f, x_c, area_face, dirichlet):
        n = len(x_c)
        a = area_face[1:n] / np.diff(x_c)
        diag = np.zeros(n); diag[:-1] -= a; diag[1:] -= a
        if dirichlet:
            diag[-1] -= area_face[n] / (x_f[n] - x_c[n - 1])
        return diag, a


class FastPoisson:
    def __init__(self, grid, dirichlet):
        g = grid
        self.g = g
        dz_d, dz_o = g.T1(g.zf, g.zc, np.ones(g.NZ + 1), dirichlet)
        Dz = g.dz
        Mz = np.diag(dz_d / Dz)
        off = dz_o / np.sqrt(Dz[:-1] * Dz[1:])
        Mz += np.diag(off, 1) + np.diag(off, -1)
        lam, Q = np.linalg.eigh(Mz)
        self.U = np.sqrt(Dz)[:, None] * Q
        self.Uinv = Q.T / np.sqrt(Dz)[None, :]
        dr_d, dr_o = g.T1(g.Rf, g.Rc, g.Rf, dirichlet)
        NR, NZ = g.NR, g.NZ
        diag = dr_d[:, None] + lam[None, :] * g.dA[:, None]
        self.sing = None
        if not dirichlet:
            self.sing = int(np.argmin(np.abs(lam)))
            diag = diag.copy(); diag[-1, self.sing] = 1.0
        self.off = dr_o
        cp = np.zeros((NR, NZ)); den = np.zeros((NR, NZ)); den[0] = diag[0]
        self.lastrow_off = np.full(NZ, dr_o[-1])
        if self.sing is not None:
            self.lastrow_off[self.sing] = 0.0
        for i in range(1, NR):
            o_row = self.lastrow_off if i == NR - 1 else dr_o[i - 1]
            cp[i - 1] = dr_o[i - 1] / den[i - 1]
            den[i] = diag[i] - o_row * cp[i - 1]
        self.cp, self.den = cp, den

    def solve(self, S):
        g = self.g
        rhs = (S / g.dz[None, :]) @ self.U
        if self.sing is not None:
            rhs[-1, self.sing] = 0.0
        NR = g.NR
        y = np.empty_like(rhs); y[0] = rhs[0] / self.den[0]
        for i in range(1, NR):
            o_row = self.lastrow_off if i == NR - 1 else self.off[i - 1]
            y[i] = (rhs[i] - o_row * y[i - 1]) / self.den[i]
        psi = np.empty_like(y); psi[-1] = y[-1]
        for i in range(NR - 2, -1, -1):
            psi[i] = y[i] - self.cp[i] * psi[i + 1]
        return psi @ self.Uinv


def grad_faces(g, phi, phib_R=None, phib_Z=None):
    FR = np.zeros((g.NR, g.NZ)); FZ = np.zeros((g.NR, g.NZ))
    FR[:-1] = (phi[1:] - phi[:-1]) / g.dRc[:, None]
    FZ[:, :-1] = (phi[:, 1:] - phi[:, :-1]) / g.dzc[None, :]
    if phib_R is not None:
        FR[-1] = (phib_R - phi[-1]) / g.dRb
    if phib_Z is not None:
        FZ[:, -1] = (phib_Z - phi[:, -1]) / g.dzb
    return FR, FZ


def div_faces(g, FR, FZ, interior_only=False):
    fr = g.AR * FR; fz = g.AZ * FZ
    if interior_only:
        fr = fr.copy(); fz = fz.copy(); fr[-1] = 0.0; fz[:, -1] = 0.0
    d = fr.copy(); d[1:] -= fr[:-1]; d += fz; d[:, 1:] -= fz[:, :-1]
    return d


def cell_vectors(FR, FZ):
    gR = FR.copy(); gR[1:] += FR[:-1]; gR *= 0.5
    gZ = FZ.copy(); gZ[:, 1:] += FZ[:, :-1]; gZ *= 0.5
    return gR, gZ


def face_tangentials(FR, FZ):
    gRc, gZc = cell_vectors(FR, FZ)
    tz = gZc.copy(); tz[:-1] = 0.5 * (gZc[:-1] + gZc[1:])
    tr = gRc.copy(); tr[:, :-1] = 0.5 * (gRc[:, :-1] + gRc[:, 1:])
    return tz, tr


def face_magnitudes(FR, FZ):
    tz, tr = face_tangentials(FR, FZ)
    return np.sqrt(FR ** 2 + tz ** 2), np.sqrt(FZ ** 2 + tr ** 2)


def multipole_bc(g, rho):
    R2, Z2 = np.meshgrid(g.Rc, g.zc, indexing='ij')
    dm = 4.0 * np.pi * rho * g.V
    M = dm.sum()
    r2 = R2 ** 2 + Z2 ** 2
    P2c = np.where(r2 > 0, (3 * Z2 ** 2 / np.maximum(r2, 1e-300) - 1) / 2, 0.0)
    Q = (dm * r2 * P2c).sum()
    def pot(R, z):
        r = np.sqrt(R * R + z * z); P2 = (3 * z * z / (r * r) - 1) / 2
        return -G * M / r - G * Q * P2 / r ** 3
    return pot(g.Rf[-1], g.zc), pot(g.Rc, g.zf[-1]), M, Q


class Solver:
    """Newton, QUMOND and AQUAL on a Grid. The optional face masks (boolean arrays shaped like the face fields) set
    mu = nu = 1 on the masked faces; no deposited result uses them."""
    def __init__(self, grid):
        self.g = grid
        self.PD = FastPoisson(grid, dirichlet=True)
        self.PN = FastPoisson(grid, dirichlet=False)

    def newton(self, rho):
        g = self.g
        phib_R, phib_Z, M, Q = multipole_bc(g, rho)
        S = 4 * np.pi * G * rho * g.V
        S[-1, :] -= g.AR[-1, :] * phib_R / g.dRb
        S[:, -1] -= g.AZ[:, -1] * phib_Z / g.dzb
        phi = self.PD.solve(S)
        FR, FZ = grad_faces(g, phi, phib_R, phib_Z)
        self.M, self.Q = M, Q
        return phi, FR, FZ

    def project(self, FR, FZ):
        phi = self.PN.solve(div_faces(self.g, FR, FZ, interior_only=True))
        GR, GZ = grad_faces(self.g, phi)
        GR[-1] = FR[-1]; GZ[:, -1] = FZ[:, -1]
        return GR, GZ, phi

    def qumond(self, gNR, gNZ, maskR=None, maskZ=None):
        mR, mZ = face_magnitudes(gNR, gNZ)
        nR, nZ = nu_mcg(mR / A0), nu_mcg(mZ / A0)
        if maskR is not None:
            nR = np.where(maskR, 1.0, nR); nZ = np.where(maskZ, 1.0, nZ)
        gR, gZ, phi = self.project(nR * gNR, nZ * gNZ)
        return gR, gZ

    def aqual(self, gNR, gNZ, maskR=None, maskZ=None, tol=1e-8, itmax=300):
        gR, gZ = self.qumond(gNR, gNZ, maskR, maskZ)
        tNz, tNr = face_tangentials(gNR, gNZ)
        for it in range(itmax):
            tgz, tgr = face_tangentials(gR, gZ)
            muR, muZ = MU(np.sqrt(gR ** 2 + tgz ** 2) / A0), MU(np.sqrt(gZ ** 2 + tgr ** 2) / A0)
            if maskR is not None:
                muR = np.where(maskR, 1.0, muR); muZ = np.where(maskZ, 1.0, muZ)
            hR, hZ = muR * gR, muZ * gZ
            hR[-1] = gNR[-1]; hZ[:, -1] = gNZ[:, -1]
            cR, cZ, chi = self.project(hR - gNR, hZ - gNZ)
            hR2 = hR - cR; hZ2 = hZ - cZ
            hR2[-1] = gNR[-1]; hZ2[:, -1] = gNZ[:, -1]
            tcz, tcr = face_tangentials(cR, cZ)
            thz = muR * tgz - tcz; thr = muZ * tgr - tcr
            thz[-1] = tNz[-1]; thr[:, -1] = tNr[:, -1]
            nR = nu_mcg(np.sqrt(hR2 ** 2 + thz ** 2) / A0); nZ = nu_mcg(np.sqrt(hZ2 ** 2 + thr ** 2) / A0)
            if maskR is not None:
                nR = np.where(maskR, 1.0, nR); nZ = np.where(maskZ, 1.0, nZ)
            gR2, gZ2, phi = self.project(nR * hR2, nZ * hZ2)
            dch = np.max(np.abs(gR2[:, 0] - gR[:, 0]) / np.maximum(np.abs(gR2[:, 0]), 1e-300))
            gR, gZ = gR2, gZ2
            if dch < tol:
                break
        self.aqual_iters = it + 1
        return gR, gZ

    def aqual_residual(self, gR, gZ, gNR, gNZ, maskR=None, maskZ=None):
        mR, mZ = face_magnitudes(gR, gZ)
        muR, muZ = MU(mR / A0), MU(mZ / A0)
        if maskR is not None:
            muR = np.where(maskR, 1.0, muR); muZ = np.where(maskZ, 1.0, muZ)
        hR, hZ = muR * gR, muZ * gZ
        hR[-1] = gNR[-1]; hZ[:, -1] = gNZ[:, -1]
        r = div_faces(self.g, hR - gNR, hZ - gNZ)
        return np.abs(r).sum() / np.abs(div_faces(self.g, gNR, gNZ)).sum()


# ================================================================== ring-potential quadrature (as in aqual-disk-densities.py)
_LT, _LW = np.polynomial.laguerre.laggauss(32)


def ellipke(m):
    m = np.asarray(m, float)
    a = np.ones_like(m); b = np.sqrt(np.maximum(1.0 - m, 0.0)); c = np.sqrt(m)
    s = 0.5 * c * c; p = 1.0
    for _ in range(40):
        an = 0.5 * (a + b); bn = np.sqrt(a * b); c = 0.5 * (a - b); p *= 2.0
        s = s + 0.5 * p * c * c; a, b = an, bn
        if np.max(np.abs(c)) < 1e-16:
            break
    K = np.pi / (2.0 * a)
    return K, K * (1.0 - s)


def ring_vsq(R, sigma_fun, amax, h, dmin_frac=1e-7, ratio=1.03):
    dmin = dmin_frac * R
    dl = dmin * ratio ** np.arange(int(np.ceil(np.log(R / dmin) / np.log(ratio))) + 2); dl = dl[dl < R]
    aL = np.unique(np.concatenate([[0.0], np.geomspace(1e-4 * R, 0.5 * R, 400), (R - dl)[::-1]]))
    dr = dmin * ratio ** np.arange(int(np.ceil(np.log((amax - R) / dmin) / np.log(ratio))) + 2); dr = dr[dr < amax - R]
    aR = np.concatenate([R + dr, [amax]])
    tot = 0.0
    for seg in (aL, aR):
        w = np.zeros(len(seg)); dseg = np.diff(seg); w[:-1] += 0.5 * dseg; w[1:] += 0.5 * dseg
        A = seg[:, None]; z = h * _LT[None, :]
        s2 = (R + A) ** 2 + z * z
        K, E = ellipke(np.minimum(4.0 * A * R / s2, 1.0 - 1e-16))
        F = ((K - (A * A - R * R + z * z) / ((A - R) ** 2 + z * z) * E) / np.sqrt(s2) * _LW[None, :]).sum(1)
        tot += np.sum(2.0 * G * seg * F * w * sigma_fun(seg))
    return tot


# ================================================================== part: validation
def kuzmin_grid(b, dzf, dRf=0.05, box=2000.0):
    return Grid(faces_stretched(dRf * b, 20 * b, box * b, 1.06), faces_stretched(dzf * b, 0.3 * b, box * b, 1.06))


def kuzmin_exact(M, b, R, z):
    s2 = R * R + (b + z) ** 2
    return G * M * R / s2 ** 1.5, G * M * (b + z) / s2 ** 1.5


def part_validate():
    out("The exact AQUAL solutions in the disks: validation of the axisymmetric solver")
    out("=" * 100)
    out("\n1. KUZMIN DISK. Exact MOND field nu(|g_N|/a0) g_N in QUMOND and AQUAL (Famaey and McGaugh 2012 Sect. 6.1.2).")
    out("   The sheet fills the first cell row (|z| < dz0). Errors are maxima of |g_R(num)/g_R(exact) - 1| over 0.1 b < R < 30 b.")
    out("   Columns: Newton at the midplane against the exact field of the same slab; at z = b against the razor-thin field;")
    out("   QUMOND and AQUAL at z = b against nu g_N (razor-thin); QUMOND and AQUAL at the midplane against the razor-thin")
    out("   value, which the slab model reaches linearly in dz0; the AQUAL flux residual and its iteration count.")
    xg, wg = np.polynomial.legendre.leggauss(64)
    for (M, b) in ((1e10, 2.0), (1e9, 1.0)):
        out(f"   M = {M:.0e} Msun, b = {b} kpc (GM/b^2 = {G*M/b**2/A0:.2f} a0):")
        for dzf in (0.02, 0.01, 0.005):
            g = kuzmin_grid(b, dzf)
            Rf = g.Rf
            rho = np.zeros((g.NR, g.NZ))
            Mann = M * b * (1 / np.sqrt(Rf[:-1] ** 2 + b * b) - 1 / np.sqrt(Rf[1:] ** 2 + b * b))
            rho[:, 0] = Mann / (4 * np.pi * g.V[:, 0])
            S = Solver(g)
            phi, FR, FZ = S.newton(rho)
            qR, qZ = S.qumond(FR, FZ)
            aR, aZ = S.aqual(FR, FZ, tol=1e-10)
            res = S.aqual_residual(aR, aZ, FR, FZ)
            R = g.Rf[1:]; sel = (R > 0.1 * b) & (R < 30 * b)
            dz0 = dzf * b
            gslab = sum(0.5 * w * kuzmin_exact(M, b, R[sel], np.abs(g.zc[0] - dz0 * x))[0] for x, w in zip(xg, wg))
            eN = np.max(np.abs(FR[sel, 0] / gslab - 1))
            j = int(np.argmin(np.abs(g.zc - b)))
            gRz, gzz = kuzmin_exact(M, b, R[sel], g.zc[j])
            nz = nu_mcg(np.hypot(gRz, gzz) / A0)
            eNz = np.max(np.abs(FR[sel, j] / gRz - 1))
            eQz = np.max(np.abs(qR[sel, j] / (nz * gRz) - 1)); eAz = np.max(np.abs(aR[sel, j] / (nz * gRz) - 1))
            gR0, gz0 = kuzmin_exact(M, b, R[sel], 0.0)
            gM0 = nu_mcg(np.hypot(gR0, gz0) / A0) * gR0
            eQ0 = np.max(np.abs(qR[sel, 0] / gM0 - 1)); eA0 = np.max(np.abs(aR[sel, 0] / gM0 - 1))
            out(f"     dz0 = {dzf:5.3f} b, grid {g.NR} x {g.NZ}: Newton mid {eN:.1e}, z=b {eNz:.1e} | z=b QUMOND {eQz:.1e}, "
                f"AQUAL {eAz:.1e} | mid QUMOND {eQ0:.1e}, AQUAL {eA0:.1e} | residual {res:.0e}, {S.aqual_iters} iterations")
    out("   Box size: the same case (M = 1e10, b = 2, dz0 = 0.005 b) with the box at 500 b instead of 2000 b:")
    for box in (500.0, 2000.0):
        g = kuzmin_grid(2.0, 0.005, box=box)
        Rf = g.Rf; b = 2.0; M = 1e10
        rho = np.zeros((g.NR, g.NZ))
        rho[:, 0] = M * b * (1 / np.sqrt(Rf[:-1] ** 2 + b * b) - 1 / np.sqrt(Rf[1:] ** 2 + b * b)) / (4 * np.pi * g.V[:, 0])
        S = Solver(g); phi, FR, FZ = S.newton(rho); aR, aZ = S.aqual(FR, FZ, tol=1e-10)
        R = g.Rf[1:]; j = int(np.argmin(np.abs(g.zc - b)))
        for Rq in (1.0, 10.0, 30.0):
            k = int(np.argmin(np.abs(R - Rq * b)))
            gRz, gzz = kuzmin_exact(M, b, R[k], g.zc[j])
            out(f"     box {box:6.0f} b: AQUAL at R = {R[k]/b:5.1f} b, z = b: {aR[k, j]/(nu_mcg(np.hypot(gRz, gzz)/A0)*gRz)-1:+.2e}")

    out("\n2. THICK EXPONENTIAL DISK: Newton against the ring-potential quadrature, and the boosts at two cell sizes.")
    Sig0, Rd, h = 500.0, 3.0, 0.375
    out(f"   Sigma_0 = {Sig0} Msun/pc^2, R_d = {Rd} kpc, h_z = {h} kpc (exponential), cut at 10 R_d. Rows at the midplane.")
    Rq = np.array([0.5, 1.0, 1.5, 2.2, 3.0, 4.0, 6.0]) * Rd
    sigf = lambda a: Sig0 * 1e6 * np.exp(-a / Rd) * (a < 10 * Rd)
    v2ring = np.array([ring_vsq(R, sigf, 10 * Rd, h) for R in Rq])
    for (dRf, dzf) in ((0.05, 0.125), (0.025, 0.0625)):
        g = Grid(faces_stretched(dRf * Rd, 12 * Rd, 300 * Rd, 1.06), faces_stretched(dzf * h, 8 * h, 300 * Rd, 1.06))
        Sig = Sig0 * 1e6 * np.exp(-g.Rc / Rd) * (g.Rc < 10 * Rd)
        pz = (np.exp(-g.zf[:-1] / h) - np.exp(-g.zf[1:] / h)) / 2.0 / g.dz
        rho = Sig[:, None] * pz[None, :]
        S = Solver(g); phi, FR, FZ = S.newton(rho)
        R = g.Rf[1:]
        v2N = Rq * np.interp(Rq, R, FR[:, 0])
        out(f"   dR0 = {dRf} R_d, dz0 = {dzf} h, grid {g.NR} x {g.NZ}. R/R_d: " + " ".join(f"{x/Rd:6.2f}" for x in Rq))
        out("     Newton V^2 grid/ring - 1:   " + " ".join(f"{x:+6.4f}" for x in v2N / v2ring - 1))
        qR, qZ = S.qumond(FR, FZ); aR, aZ = S.aqual(FR, FZ)
        bq = np.log10(np.interp(Rq, R, qR[:, 0]) / np.interp(Rq, R, FR[:, 0]))
        ba = np.log10(np.interp(Rq, R, aR[:, 0]) / np.interp(Rq, R, FR[:, 0]))
        yN = np.interp(Rq, R, FR[:, 0]) / A0
        out("     y_N:                        " + " ".join(f"{x:6.3f}" for x in yN))
        out("     log10 nu(y_N) (algebraic):  " + " ".join(f"{x:6.3f}" for x in np.log10(nu_mcg(yN))))
        out("     boost QUMOND, AQUAL:        " + " ".join(f"{x:6.3f}" for x in bq) + "  |  " + " ".join(f"{x:6.3f}" for x in ba))
    out(f"\n[run time {time.time()-T00:.0f} s]")


# ================================================================== galaxy models
def load_inputs(datadir):
    pts = list(csv.DictReader(open(os.path.join(datadir, "aqual-disk-points.csv"), encoding="utf-8")))
    gas = {}
    for r in csv.DictReader(open(os.path.join(datadir, "aqual-disk-gas-profiles.csv"), encoding="utf-8")):
        gas.setdefault(r["galaxy"], []).append((float(r["node_R_kpc"]), float(r["Sigma_gas_Msun_pc2"])))
    gals = {}
    for p in pts:
        gals.setdefault(p["galaxy"], []).append(p)
    return gals, gas


def hernquist_fit(R, Vb):
    s = Vb > 0
    if s.sum() < 1:
        return None
    Mobs = UPS_B * R[s] * Vb[s] ** 2 / G
    best = None
    for a in np.geomspace(0.005, 20.0, 400):
        f = R[s] ** 2 / (R[s] + a) ** 2
        w = 1.0 / Mobs ** 2
        Mb = np.sum(w * f * Mobs) / np.sum(w * f * f)
        chi = np.sum(w * (Mb * f - Mobs) ** 2)
        if best is None or chi < best[0]:
            best = (chi, Mb, a)
    return best[1], best[2], math.sqrt(best[0] / s.sum())


class GalaxyModel:
    """One galaxy's baryons on its own grid: SPARC's disk at Ups_disk = 0.5 in a layer of scale height h_z, the
    deconvolved gas in a layer of H_GAS, and a Hernquist bulge. hz_shift moves h_z by that many times the 0.137 dex
    scatter of the flattening. density() returns the cell densities and keeps the three parts in rho_parts."""
    def __init__(self, name, rows, gasnodes, hz_shift=0.0):
        self.name = name
        rows = sorted(rows, key=lambda r: float(r["R_kpc"]))
        f = lambda k: np.array([float(r[k]) for r in rows])
        self.R = f("R_kpc"); self.Vo = f("Vobs"); self.eV = f("eV"); self.Vg = f("Vgas"); self.Vd = f("Vdisk")
        self.Vb = f("Vbul"); self.SBd = f("SBdisk"); self.y = f("y"); self.res = f("res"); self.selA = f("selA") > 0.5
        self.rho_kq = f("rho_kq"); self.KF = f("KF_kq"); self.V2bar = f("V2bar")
        self.Rd = float(rows[0]["Rdisk"])
        self.hz = 1.12 * self.Rd / (Q_CEN * 10 ** (-hz_shift * Q_SIG))
        gn = np.array(sorted(gasnodes))
        self.gas_R, self.gas_S = gn[:, 0], gn[:, 1] * 1e6
        self.bulge = hernquist_fit(self.R, self.Vb)
        Rlast = self.R.max()
        self.Rcut = Rlast + 8 * self.Rd
        dR0 = min(self.Rd / DR_DIV, Rlast / 100.0)
        R1 = max(1.3 * Rlast, Rlast + 3 * self.Rd)
        hmin = min(self.hz, H_GAS)
        box = 300.0 * Rlast
        self.grid = Grid(faces_stretched(dR0, R1, box, 1.07, dstart=min(dR0, 0.4 * self.R.min())),
                         faces_stretched(hmin / DZ_DIV, 4 * max(self.hz, H_GAS), box, 1.07))

    def sigma_disk(self, a):
        good = self.SBd > 0
        Rg, lg = self.R[good], np.log(self.SBd[good])
        v = np.interp(a, Rg, lg)
        v = np.where(a < Rg[0], lg[0] + (Rg[0] - a) / self.Rd, v)
        v = np.where(a > Rg[-1], lg[-1] - (a - Rg[-1]) / self.Rd, v)
        return UPS_D * np.exp(v) * 1e6 * (a < self.Rcut)

    def density(self):
        g = self.grid
        pz = lambda h: (np.exp(-g.zf[:-1] / h) - np.exp(-g.zf[1:] / h)) / 2.0 / g.dz
        # sech^2(z/z0) with z0 = 2 h: the same fall-off far from the plane, half the midplane density
        ps = lambda h: (np.tanh(g.zf[1:] / (2.0 * h)) - np.tanh(g.zf[:-1] / (2.0 * h))) / 2.0 / g.dz
        rho_d = self.sigma_disk(g.Rc)[:, None] * (ps(self.hz) if PROFILE == "sech2" else pz(self.hz))[None, :]
        rho_g = np.interp(g.Rc, self.gas_R, self.gas_S, right=0.0)[:, None] * pz(H_GAS)[None, :]
        rho_b = np.zeros_like(rho_d)
        if self.bulge is not None:
            Mb, ab, _ = self.bulge
            Rc, zc = g.Rc, g.zc
            # 3 x 3 sub-cell sampling of the Hernquist density
            fr = np.array([1 / 6, 0.5, 5 / 6])
            acc = np.zeros_like(rho_d)
            for u in fr:
                for v in fr:
                    RR = (g.Rf[:-1] + u * g.dR)[:, None]; ZZ = (g.zf[:-1] + v * g.dz)[None, :]
                    r = np.sqrt(RR ** 2 + ZZ ** 2)
                    acc += Mb * ab / (2 * np.pi * r * (r + ab) ** 3)
            rho_b = acc / 9.0
            mtot = (4 * np.pi * rho_b * g.V).sum()
            if mtot > 0:
                rho_b *= Mb / mtot
        self.rho_parts = (rho_d, rho_g, rho_b)
        return rho_d + rho_g + rho_b


def run_galaxy(name, rows, gasnodes):
    """Newton, QUMOND and AQUAL for one galaxy model; the midplane radial fields at the data radii."""
    m = GalaxyModel(name, rows, gasnodes)
    g = m.grid
    rho = m.density()
    S = Solver(g)
    phi, FR, FZ = S.newton(rho)
    Rf1 = g.Rf[1:]
    at = lambda F: np.interp(m.R, Rf1, F[:, 0])
    gN = at(FR)
    qR, qZ = S.qumond(FR, FZ); gQ = at(qR)
    r = dict(gN=gN, gQ=gQ)
    aR, aZ = S.aqual(FR, FZ); r["gA"] = at(aR); r["itA"] = S.aqual_iters
    r["grid"] = (g.NR, g.NZ); r["hz"] = m.hz; r["bulge"] = m.bulge
    return m, r


def part_galaxies():
    datadir = arg("--data", DATA_DIR)
    pred = arg("--pred")
    if pred is None:
        raise SystemExit("--part galaxies needs --pred FILE")
    k, n = (int(x) for x in arg("--slice", "0/1").split("/"))
    gals, gas = load_inputs(datadir)
    names = sorted(nm for nm, rows in gals.items() if any(r["selA"] == "1" for r in rows))
    mine = [nm for i, nm in enumerate(names) if i % n == k]
    if arg("--only"):
        mine = [x for x in arg("--only").split(",") if x in gals]
    cols = ["galaxy", "R", "selA", "y", "res", "rho_kq", "V2N_over_V2bar", "boostQ", "boostA", "hz", "itA"]
    fh = open(pred, "w", newline="\n", encoding="utf-8")
    fh.write(",".join(cols) + "\n")
    for nm in mine:
        t = time.time()
        m, r0 = run_galaxy(nm, gals[nm], gas[nm])
        L = lambda a, b: np.log10(a / b)
        for i in range(len(m.R)):
            vals = [nm, f"{m.R[i]:.4f}", str(int(m.selA[i])), f"{m.y[i]:.6e}", f"{m.res[i]:.5f}", f"{m.rho_kq[i]:.6e}",
                    f"{m.R[i]*r0['gN'][i]/m.V2bar[i] if m.V2bar[i] > 0 else float('nan'):.5f}",
                    f"{L(r0['gQ'][i], r0['gN'][i]):.5f}", f"{L(r0['gA'][i], r0['gN'][i]):.5f}", f"{r0['hz']:.4f}", str(r0['itA'])]
            fh.write(",".join(vals) + "\n")
        fh.flush()
        print(f"{nm}: grid {r0['grid']}, h_z {r0['hz']:.3f}, AQUAL iterations {r0['itA']}, bulge {r0['bulge']}, {time.time()-t:.1f} s", flush=True)
    fh.close()


# ================================================================== part: summary
def load_pred(files):
    rows = []
    for f in files:
        rows += list(csv.DictReader(open(f, encoding="utf-8")))
    cols = rows[0].keys()
    D = {}
    for c in cols:
        if c == "galaxy":
            D[c] = np.array([r[c] for r in rows])
        else:
            D[c] = np.array([float(r[c]) for r in rows])
    return D


def clustered(vals, gnames, rng, nboot=2000):
    if len(vals) == 0:
        return np.nan, np.nan
    ug = sorted(set(gnames)); idx = {u: np.where(gnames == u)[0] for u in ug}
    bs = []
    for _ in range(nboot):
        pick = rng.choice(len(ug), len(ug))
        bs.append(np.concatenate([vals[idx[ug[p]]] for p in pick]).mean())
    return vals.mean(), np.std(bs, ddof=1)


def part_summary():
    rng = np.random.default_rng(20261011)
    if arg("--pred") is None:
        raise SystemExit("--part summary needs --pred FILE1,FILE2")
    files = arg("--pred").split(",")
    D = load_pred(files)
    A = D["selA"] > 0.5
    y, res, nm = D["y"], D["res"], D["galaxy"]
    Gm = D["boostA"] - np.log10(nu_mcg(np.maximum(y, 1e-9)))       # full AQUAL minus the algebraic relation
    GQ = D["boostQ"] - np.log10(nu_mcg(np.maximum(y, 1e-9)))
    out("The exact AQUAL solutions in the disks: the departure from the algebraic relation at the SPARC points of sample A")
    out("=" * 104)
    out(f"Sample A: {A.sum()} points in {len(set(nm[A]))} galaxies; predictions on the midplane at the data radii.")
    out("Empirical function at a0 = 1.2e-10 m/s^2; mass-to-light ratios 0.5 (disk) and 0.7 (bulge).")
    out("\n1. THE BARYONIC MODELS AND THE FULL SOLUTIONS")
    r = D["V2N_over_V2bar"][A]
    out(f"   Newtonian V^2 of the solver's model over SPARC's V_bar^2 (Ups 0.5/0.7): median {np.nanmedian(r):.3f}, 16-84 percent "
        f"{np.nanpercentile(r,16):.3f} to {np.nanpercentile(r,84):.3f}; points beyond 10 percent: {np.sum(np.abs(r-1) > 0.1)} of {len(r)}.")
    out("   Full solution minus the algebraic relation, log10 g_full - log10[nu(y) g_bar] with SPARC's g_bar, by y bin:")
    out("     y bin        N    AQUAL median [16, 84]           QUMOND median     points denser than 0.1: AQUAL median")
    for lo, hi in ((0.01, 0.1), (0.1, 1), (1, 4), (4, 1e9)):
        s = A & (y >= lo) & (y < hi)
        d = s & (D["rho_kq"] > 0.1)
        out(f"     [{lo:g},{hi:g})  {s.sum():5d}   {np.median(Gm[s]):+.4f} [{np.percentile(Gm[s],16):+.4f}, {np.percentile(Gm[s],84):+.4f}]   "
            f"{np.median(GQ[s]):+.4f}          {np.median(Gm[d]) if d.sum() else float('nan'):+.4f} ({d.sum()} points)")
    m, se = clustered(res[A] - Gm[A], nm[A], rng)
    m0, se0 = clustered(res[A], nm[A], rng)
    out(f"   Mean observed residual, clustered: from the algebraic relation {m0:+.4f} +/- {se0:.4f}; from the full AQUAL solution {m:+.4f} +/- {se:.4f}.")
    out(f"\n[run time {time.time()-T00:.0f} s]")


if __name__ == "__main__":
    part = arg("--part", "validate")
    if part == "validate":
        part_validate()
    elif part == "galaxies":
        part_galaxies()
    elif part == "summary":
        part_summary()
    else:
        raise SystemExit(f"unknown --part {part}: validate, galaxies or summary")
    if OUT and part in ("validate", "summary"):
        with open(OUT, "w", newline="\n", encoding="utf-8") as fh:
            fh.write("\n".join(_lines) + "\n")
