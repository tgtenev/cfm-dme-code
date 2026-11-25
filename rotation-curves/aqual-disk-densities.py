"""Midplane baryon densities and deconvolved gas surface densities at every SPARC point: the inputs of the galaxy
models of the exact AQUAL solutions in the disks (aqual-disk-solver.py).

WHAT IT COMPUTES
  For the 153 SPARC galaxies of Paper I, Sect. 4.1 (quality flag Q < 3, inclination of 30 degrees or more; 3,168
  points), at every data radius:
  1. The stellar disk's exponential scale height h_z = 1.12 R_d/q. Here q = 8.06 is the log-mean flattening h_R/h_z of
     the 34 edge-on disks of Kregel, van der Kruit and Freeman 2005, Table 1 (I band), with a scatter of 0.137 dex. The
     I-band scale length is 1.12 times the K-band one (their Sect. 3.5.2), and SPARC's 3.6 micron scale length R_d is
     taken as the K-band one.
  2. The gas surface density. SPARC gives V_gas (helium included, factor 1.33), not the surface density. The surface
     density is recovered by a non-negative least-squares fit of V_gas|V_gas| with piecewise-linear rings, whose midplane
     rotation curve is computed exactly (ring potentials, complete elliptic integrals by the arithmetic-geometric mean)
     for a layer of exponential scale height 0.1 kpc. Soft constraints come from SPARC's H I mass and H I radius.
     Molecular gas is not in SPARC and is omitted.
  3. The bulge's luminosity density at r = R, from SPARC's SBbul deprojected as a sphere (Abel integral).
  4. The midplane density of each exponential layer of surface density Sigma and scale height h, rho_0 = Sigma/(2 h).
     The mass-to-light ratios at 3.6 micron are Ups_disk = 0.5 and Ups_bulge = 0.7 (Li et al. 2018, Sect. 2), and the
     gas lies in a layer of 0.2 kpc (Angus et al. 2015, Sect. 2.1.1).
  5. y = g_bar/a0 at a0 = 1.2e-10 m/s^2, and the residual from the empirical function of McGaugh, Lelli and Schombert,
     nu(y) = 1/(1 - exp(-sqrt(y))).
  6. K_F, the Frobenius norm of the Hessian of the Newtonian potential at the midplane over sqrt(6) G, from SPARC's
     V_bar^2 and the midplane density (exact at z = 0; a point mass gives M/r^3). No result of Papers I and II uses it.
  The script also checks the ring quadrature on the Kuzmin disk and the Abel deprojection on the Plummer sphere.

RESULTS SUPPORTED
  Paper II, Sect. 6.1 (the disk departure), and the fits of the full solution in Paper I, Sect. 4.2 and Paper II,
  Sect. 6.1. aqual-disk-solver.py builds every galaxy model from the two files written here, and the refits read the
  points file for their sample.

INPUTS
  data/sparc/SPARC_Lelli2016c.mrt, data/sparc/MassModels_Lelli2016c.mrt

OUTPUTS
  rotation-curves/aqual-disk-points.csv: one row per point. Columns: galaxy, R_kpc; SPARC's Vobs, eV, Vgas, Vdisk,
    Vbul (km/s) and SBdisk, SBbul (Lsun/pc^2); Sigma_gas (Msun/pc^2); j_bul_Lpc3, the bulge's luminosity density
    (Lsun/pc^3); vmax, the galaxy's largest Vobs; Rdisk (kpc); V2bar, SPARC's V_bar^2 at Ups 0.5/0.7, and dV2dR, its
    radial derivative; y; res = log10 g_obs - log10[nu(y) g_bar]; selA, 1 for sample A (e_V/V < 0.10 and V_bar^2 > 0:
    2,696 points in 147 galaxies); hz_kq, the scale height of item 1 (kpc); rho_star_kq, rho_gas, rho_bul and their sum
    rho_kq (Msun/pc^3); KF_kq (Msun/pc^3).
  rotation-curves/aqual-disk-gas-profiles.csv: the deconvolved gas surface density (Msun/pc^2, helium included) at each
    galaxy's nodes.
  A summary of the checks is printed to the screen.

COMMAND, from the repository root
  python rotation-curves/aqual-disk-densities.py
  The two files are written to rotation-curves/; --outdir DIR writes them to DIR instead. Needs numpy.
  Run time 1 to 2 minutes.
"""
import math, os, sys, time
import numpy as np

T0 = time.time()
T1_PATH = "data/sparc/SPARC_Lelli2016c.mrt"
T2_PATH = "data/sparc/MassModels_Lelli2016c.mrt"
CONV = 3.24078e-14                # (km/s)^2/kpc -> m/s^2
A0 = 1.2e-10
G_KPC = 4.30091e-6                # kpc (km/s)^2/Msun
UPS_D, UPS_B = 0.5, 0.7
H_GAS = 0.20                      # kpc, exponential scale height of the gas layer (Angus et al. 2015 Sect. 2.1.1)
H_GAS_INV = 0.10                  # kpc, layer thickness used when deconvolving V_gas

OUTDIR = sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv else os.path.dirname(os.path.abspath(__file__))


def out(*a):
    print(" ".join(str(x) for x in a))


def shown_path(p):
    """p relative to the working directory (the repository root) when it lies below it, else as given."""
    try:
        r = os.path.relpath(p)
    except ValueError:
        return p
    return p if r.startswith("..") else r.replace(os.sep, "/")


# ------------------------------------------------------------------ SPARC
def read_table1(path):
    gals = {}
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        lines = fh.read().splitlines()
    rules = [i for i, ln in enumerate(lines) if ln.strip() and set(ln.strip()) == {"-"}]
    for ln in lines[rules[-1] + 1:]:
        f = ln.split()
        if len(f) != 19:
            continue
        gals[f[0]] = dict(T=int(f[1]), D=float(f[2]), eD=float(f[3]), Inc=float(f[5]), eInc=float(f[6]),
                          L=float(f[7]), Reff=float(f[9]), SBeff=float(f[10]), Rdisk=float(f[11]),
                          SBdisk0=float(f[12]), MHI=float(f[13]), RHI=float(f[14]), Vflat=float(f[15]),
                          eVflat=float(f[16]), Q=int(f[17]))
    return gals


def read_table2(path):
    rows = []
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        lines = fh.read().splitlines()
    rules = [i for i, ln in enumerate(lines) if ln.strip() and set(ln.strip()) == {"-"}]
    for ln in lines[rules[-1] + 1:]:
        f = ln.split()
        if len(f) != 10:
            continue
        rows.append((f[0],) + tuple(float(x) for x in f[1:]))
    return rows


def nu_mcg(y):
    y = np.maximum(np.asarray(y, float), 1e-300)
    return 1.0 / (-np.expm1(-np.sqrt(y)))


# ------------------------------------------------------------------ elliptic integrals by the AGM
def ellipke(m):
    """Complete elliptic integrals K(m), E(m), parameter m = k^2, 0 <= m < 1 (vectorized AGM)."""
    m = np.asarray(m, float)
    a = np.ones_like(m)
    b = np.sqrt(np.maximum(1.0 - m, 0.0))
    c = np.sqrt(m)
    s = 0.5 * c * c          # sum of 2^(n-1) c_n^2, n = 0 term
    p = 1.0
    for _ in range(40):
        an = 0.5 * (a + b)
        bn = np.sqrt(a * b)
        c = 0.5 * (a - b)
        p *= 2.0
        s = s + 0.5 * p * c * c
        a, b = an, bn
        if np.max(np.abs(c)) < 1e-16:
            break
    K = np.pi / (2.0 * a)
    E = K * (1.0 - s)
    return K, E


# ------------------------------------------------------------------ midplane rotation curve of a thick disk
_LT, _LW = np.polynomial.laguerre.laggauss(32)


def ring_weights(R, amax, h, dmin_frac=1e-7, ratio=1.03):
    """Quadrature in a for V^2(R) = int Sigma(a) W(a) da of a disk whose density is Sigma(a) e^{-|z|/h}/(2h).
    Returns nodes a and weights W (in (km/s)^2 per (Msun/kpc^2) per kpc)."""
    dmin = dmin_frac * R
    nl = int(np.ceil(np.log(R / dmin) / np.log(ratio))) + 2
    dl = dmin * ratio ** np.arange(nl)
    dl = dl[dl < R]
    # inner nodes uniform in log a below R/2, so that the centre of the mass distribution is resolved
    a_in = np.geomspace(1e-4 * R, 0.5 * R, 400)
    aL = np.unique(np.concatenate([[0.0], a_in, (R - dl)[::-1]]))          # 0 ... R - dmin
    nr = int(np.ceil(np.log((amax - R) / dmin) / np.log(ratio))) + 2 if amax > R else 0
    dr = dmin * ratio ** np.arange(nr)
    dr = dr[dr < amax - R]
    aR = np.concatenate([R + dr, [amax]]) if amax > R else np.array([])
    out_a, out_w = [], []
    for seg in (aL, aR):
        if len(seg) < 2:
            continue
        w = np.zeros(len(seg))
        dseg = np.diff(seg)
        w[:-1] += 0.5 * dseg
        w[1:] += 0.5 * dseg
        out_a.append(seg)
        out_w.append(w)
    a = np.concatenate(out_a)
    wq = np.concatenate(out_w)
    # vertical integral with Gauss-Laguerre in t = |z|/h of the source
    z = h * _LT[None, :]
    A = a[:, None]
    s2 = (R + A) ** 2 + z * z
    m = 4.0 * A * R / s2
    K, E = ellipke(np.minimum(m, 1.0 - 1e-16))
    F = (K - (A * A - R * R + z * z) / ((A - R) ** 2 + z * z) * E) / np.sqrt(s2)
    Fz = (F * _LW[None, :]).sum(axis=1)
    W = 2.0 * G_KPC * a * Fz * wq
    return a, W


def vsq_profile(Rs, sigma_fun, amax, h):
    return np.array([np.sum(W * sigma_fun(a)) for a, W in (ring_weights(R, amax, h) for R in Rs)])


# ------------------------------------------------------------------ NNLS (Lawson-Hanson)
def nnls(A, b, maxiter=None, tol=None):
    m, n = A.shape
    x = np.zeros(n)
    P = np.zeros(n, bool)
    if tol is None:
        tol = 1e-12 * np.max(np.abs(A.T @ b)) + 1e-300
    w = A.T @ (b - A @ x)
    maxiter = maxiter or 30 * n
    it = 0
    while (~P).any() and np.max(np.where(~P, w, -np.inf)) > tol and it < maxiter:
        j = int(np.argmax(np.where(~P, w, -np.inf)))
        P[j] = True
        while True:
            it += 1
            z = np.zeros(n)
            z[P] = np.linalg.lstsq(A[:, P], b, rcond=None)[0]
            if np.all(z[P] > 0) or it >= maxiter:
                break
            neg = P & (z <= 0)
            alpha = np.min(x[neg] / (x[neg] - z[neg]))
            x = x + alpha * (z - x)
            P = P & (x > 1e-14 * max(1.0, np.max(np.abs(x))))
            x[~P] = 0.0
        x = z
        w = A.T @ (b - A @ x)
    return np.maximum(x, 0.0)


def hat(nodes, k, a):
    """Piecewise-linear hat function k on the node set, evaluated at a."""
    y = np.zeros(len(nodes))
    y[k] = 1.0
    return np.interp(a, nodes, y, left=0.0, right=0.0)


# ------------------------------------------------------------------ Abel deprojection of a spherical bulge
def pchip(x, y):
    """Monotone cubic Hermite interpolant (Fritsch-Carlson); returns f and f' as functions on [x0, xn]."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    h = np.diff(x); d = np.diff(y) / h
    m = np.zeros_like(y)
    if len(x) == 2:
        m[:] = d[0]
    else:
        for k in range(1, len(x) - 1):
            if d[k - 1] * d[k] > 0:
                w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
                m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
        m[0] = d[0]; m[-1] = d[-1]
    def seg(t):
        k = np.clip(np.searchsorted(x, t) - 1, 0, len(h) - 1)
        s = (t - x[k]) / h[k]
        return k, s
    def f(t):
        k, s = seg(t)
        h00 = 2 * s ** 3 - 3 * s ** 2 + 1; h10 = s ** 3 - 2 * s ** 2 + s; h01 = -2 * s ** 3 + 3 * s ** 2; h11 = s ** 3 - s ** 2
        return h00 * y[k] + h10 * h[k] * m[k] + h01 * y[k + 1] + h11 * h[k] * m[k + 1]
    def fp(t):
        k, s = seg(t)
        d00 = 6 * s ** 2 - 6 * s; d10 = 3 * s ** 2 - 4 * s + 1; d01 = -6 * s ** 2 + 6 * s; d11 = 3 * s ** 2 - 2 * s
        return (d00 * y[k] + d01 * y[k + 1]) / h[k] + d10 * m[k] + d11 * m[k + 1]
    return f, fp, m


def abel_density(Rd, Id, r_eval):
    """Spherical luminosity density j(r) from a projected profile I(R) sampled at Rd (I > 0): monotone cubic in
    (R, ln I) between samples, exponential continuation beyond them. j(r) = -(1/pi) int_0^inf I'(r cosh u) du."""
    Rd = np.asarray(Rd, float); Id = np.asarray(Id, float)
    ok = Id > 0
    Rd, Id = Rd[ok], Id[ok]
    if len(Rd) < 2:
        return np.zeros_like(np.asarray(r_eval, float))
    lI = np.log(Id)
    fI, fIp, mm = pchip(Rd, lI)
    s_in = mm[0]
    s_out = min(mm[-1], -1.0 / max(Rd[-1], 1e-3))        # decline at least as fast as e^{-R/R_last}
    def lnI(R):
        R = np.asarray(R, float)
        v = fI(np.clip(R, Rd[0], Rd[-1]))
        v = np.where(R < Rd[0], lI[0] + s_in * (R - Rd[0]), v)
        v = np.where(R > Rd[-1], lI[-1] + s_out * (R - Rd[-1]), v)
        return v
    def dlnI(R):
        R = np.asarray(R, float)
        v = fIp(np.clip(R, Rd[0], Rd[-1]))
        v = np.where(R < Rd[0], s_in, v)
        v = np.where(R > Rd[-1], s_out, v)
        return v
    Rend = Rd[-1] + 40.0 / max(-s_out, 1e-6)
    res = []
    for r in np.atleast_1d(r_eval):
        if r <= 0 or r >= Rend:
            res.append(0.0); continue
        umax = np.arccosh(Rend / r)
        u = np.linspace(0.0, umax, 4001)
        Rr = r * np.cosh(u)
        f = np.exp(lnI(Rr)) * dlnI(Rr)
        res.append(-np.trapezoid(f, u) / np.pi)
    return np.array(res)


def plummer_check():
    a = 1.0; M = 1.0
    R = np.linspace(0.05, 6.0, 40)
    Sig = M * a * a / (np.pi * (a * a + R * R) ** 2)
    r = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0])
    rho = abel_density(R, Sig, r)
    ex = 3 * M / (4 * np.pi * a ** 3) * (1 + r * r / a ** 2) ** -2.5
    return r, rho / ex - 1


# ------------------------------------------------------------------ main
def main():
    out("Midplane densities and gas profiles of the SPARC disks, the inputs of the exact AQUAL solutions in the disks")
    out("=" * 108)
    gals = read_table1(T1_PATH)
    rows = read_table2(T2_PATH)
    keep = {k: g for k, g in gals.items() if g["Q"] != 3 and g["Inc"] >= 30}
    bygal = {}
    for r in rows:
        bygal.setdefault(r[0], []).append(r)

    # ---------------- 1. the stellar scale height
    out("\n1. THE STELLAR SCALE HEIGHT FROM THE FLATTENING OF EDGE-ON DISKS")
    kregel_q = np.array([7.0, 9.2, 8.9, 7.0, 9.9, 9.7, 7.6, 8.4, 6.3, 14.0, 10.8, 8.7, 6.9, 7.4, 8.5, 7.1, 6.5, 8.2, 6.4,
                         6.0, 5.3, 19.4, 7.8, 5.4, 12.9, 11.1, 11.7, 6.4, 6.9, 9.9, 6.4, 3.9, 10.9, 6.0])
    lq = np.log10(kregel_q)
    out(f"   Kregel, van der Kruit and Freeman 2005, Table 1, h_R/h_z of {len(kregel_q)} edge-on disks (I band): mean {kregel_q.mean():.2f},")
    out(f"   median {np.median(kregel_q):.2f}, log-mean {10**lq.mean():.2f}, log standard deviation {lq.std(ddof=1):.3f} dex.")
    Q_CEN = 10 ** lq.mean(); Q_SIG = lq.std(ddof=1)
    out(f"   Adopted: q = {Q_CEN:.2f} (log-mean) with {Q_SIG:.3f} dex scatter; h_z = 1.12 R_d/q (h_R,I/h_R,K = 1.12, their Sect. 3.5.2).")

    def hz_of(g, shift=0.0):
        """Scale height in kpc; shift in units of the flattening's 1 sigma."""
        Rd = g["Rdisk"]
        return 1.12 * Rd / (Q_CEN * 10 ** (-shift * Q_SIG))

    # ---------------- 2. checks of the ring quadrature and the Abel deprojection
    out("\n2. MIDPLANE ROTATION CURVES OF THICK DISKS (ring potentials, AGM elliptic integrals) AND THE ABEL DEPROJECTION: CHECKS")
    Mk, bk = 1e10, 2.0
    sigk = lambda a: Mk * bk / (2 * np.pi * (a * a + bk * bk) ** 1.5)
    Rt = np.array([0.2, 0.5, 1, 2, 4, 8, 16]) * bk
    vsq = vsq_profile(Rt, sigk, 5000 * bk, 1e-4 * bk)
    ex = G_KPC * Mk * Rt ** 2 / (Rt ** 2 + bk ** 2) ** 1.5
    out("   Kuzmin disk (M = 1e10, b = 2 kpc), layer of h = 1e-4 b, against the exact razor-thin V^2 = GM R^2/(R^2+b^2)^1.5:")
    out("   R/b " + " ".join(f"{x/bk:8.2f}" for x in Rt))
    out("   rel " + " ".join(f"{x:+8.1e}" for x in vsq / ex - 1))
    r_, e_ = plummer_check()
    out("   Abel deprojection of a projected Plummer profile sampled at 40 radii: relative error at r/a = "
        + ", ".join(f"{a:.1f}: {b:+.1e}" for a, b in zip(r_, e_)))

    # ---------------- 3. per-galaxy model: gas deconvolution and bulge deprojection
    out("\n3. PER-GALAXY BARYONIC MODEL: GAS DECONVOLUTION AND BULGE DEPROJECTION")
    gasprof = {}
    stats = []
    for nm in sorted(keep):
        if nm not in bygal:
            continue
        rr = sorted(bygal[nm], key=lambda r: r[2])
        R = np.array([r[2] for r in rr]); Vg = np.array([r[5] for r in rr])
        Vb = np.array([r[7] for r in rr]); SBb = np.array([r[9] for r in rr])
        g = keep[nm]
        # gas: hats at the data radii (merged when closer than 0.4 of the median spacing), at 0, and beyond
        Rlast = R.max()
        dR = np.median(np.diff(R)) if len(R) > 1 else Rlast
        nodes = [0.0]
        for x in R:
            if x - nodes[-1] > 0.4 * dR:
                nodes.append(x)
        for f in (1.15, 1.35, 1.6, 2.0, 2.5):
            nodes.append(f * Rlast)
        nodes.append(3.0 * Rlast)
        nodes = np.array(nodes)
        amax = nodes[-1]
        WW = [ring_weights(x, amax, H_GAS_INV) for x in R]
        A = np.array([[np.sum(W * hat(nodes, k, a)) for k in range(len(nodes) - 1)] for (a, W) in WW])
        bvec = Vg * np.abs(Vg)
        sc = 0.02 * np.max(np.abs(bvec)) + 0.02 * np.abs(bvec) + 1e-6
        nb = len(nodes) - 1
        D2 = np.zeros((max(nb - 2, 0), nb))
        for k in range(nb - 2):
            D2[k, k:k + 3] = (1, -2, 1)
        colnorm = np.sqrt((A / sc[:, None]) ** 2).sum(axis=0).mean() if nb else 1.0
        lam = 0.03 * colnorm
        # soft constraints from SPARC Table 1: total gas mass 1.33 M_HI (10 percent) and Sigma_gas(R_HI) = 1.33 Msun/pc^2
        # (30 percent), where R_HI is the radius at which the H I surface density is 1 Msun/pc^2
        afine = np.linspace(0.0, amax, 6001)
        Mk = np.array([np.trapezoid(2 * np.pi * afine * hat(nodes, k, afine), afine) for k in range(nb)])
        extra_A, extra_b = [], []
        if g["MHI"] > 0:
            Mt = 1.33 * g["MHI"] * 1e9
            extra_A.append(Mk / (0.10 * Mt)); extra_b.append(1.0 / 0.10)
        if g["RHI"] > 0 and g["RHI"] < nodes[-1]:
            hk = np.array([hat(nodes, k, np.array([g["RHI"]]))[0] for k in range(nb)])
            extra_A.append(hk / (0.30 * 1.33e6)); extra_b.append(1.0 / 0.30)
        Aa = np.vstack([A / sc[:, None], lam * D2] + ([np.array(extra_A)] if extra_A else []))
        ba = np.concatenate([bvec / sc, np.zeros(D2.shape[0]), np.array(extra_b)])
        c = nnls(Aa, ba) if np.max(np.abs(bvec)) > 0 else np.zeros(nb)
        fit = A @ c
        rms = np.sqrt(np.mean((fit - bvec) ** 2)) / max(np.max(np.abs(bvec)), 1e-6)
        sig_nodes = np.concatenate([c, [0.0]])        # Msun/kpc^2 at the nodes (last node zero)
        Mgas = np.trapezoid(2 * np.pi * nodes * sig_nodes, nodes)
        gasprof[nm] = (nodes, sig_nodes / 1e6)       # Msun/pc^2
        # bulge mass from the deprojection against R Vbul^2/G
        if np.sum(SBb > 0) >= 2:
            rgrid = np.linspace(1e-3, R.max(), 400)
            jb = abel_density(R, SBb * 1e6, rgrid)          # Lsun/kpc^3
            Mb_r = np.concatenate([[0.0], np.cumsum(0.5 * (4 * np.pi * rgrid[1:] ** 2 * jb[1:] + 4 * np.pi * rgrid[:-1] ** 2 * jb[:-1]) * np.diff(rgrid))])
            sb = Vb > 0
            mb_ratio = np.median(np.interp(R[sb], rgrid, Mb_r) / (R[sb] * Vb[sb] ** 2 / G_KPC)) if sb.sum() else np.nan
        else:
            mb_ratio = np.nan
        stats.append((nm, rms, Mgas / (1.33 * g["MHI"] * 1e9) if g["MHI"] > 0 else np.nan, mb_ratio,
                      sig_nodes[1:-6].max() / 1e6 if len(sig_nodes) > 7 else np.nan))
    rmss = np.array([s[1] for s in stats]); mrat = np.array([s[2] for s in stats])
    out(f"   Gas: {len(stats)} galaxies; rms of the fit to V_gas|V_gas| over max|V_gas|^2: median {np.median(rmss):.3f}, "
        f"90th percentile {np.percentile(rmss, 90):.3f}, max {rmss.max():.3f}.")
    ok = np.isfinite(mrat)
    out(f"   Deconvolved gas mass over 1.33 M_HI (Table 1): median {np.median(mrat[ok]):.2f}, 16-84 percent "
        f"{np.percentile(mrat[ok],16):.2f} to {np.percentile(mrat[ok],84):.2f} (gas beyond 3 R_last is not constrained).")
    pk = np.array([s[4] for s in stats]); pk = pk[np.isfinite(pk)]
    out(f"   Peak deconvolved gas surface density inside the data: median {np.median(pk):.1f} Msun/pc^2, 16-84 percent "
        f"{np.percentile(pk,16):.1f} to {np.percentile(pk,84):.1f} (helium included).")
    mbr = np.array([s[3] for s in stats]); okb = np.isfinite(mbr)
    out(f"   Bulges with SBbul at two or more points: {okb.sum()}; deprojected light inside r over R Vbul^2/G (Ups = 1): median "
        f"{np.median(mbr[okb]):.2f}, 16-84 percent {np.percentile(mbr[okb],16):.2f} to {np.percentile(mbr[okb],84):.2f}.")

    # ---------------- 4. midplane densities at every point
    out("\n4. MIDPLANE DENSITIES (stars + gas + bulge) AT EVERY POINT")
    pts = []
    for nm in sorted(keep):
        if nm not in bygal:
            continue
        g = keep[nm]
        rr = sorted(bygal[nm], key=lambda r: r[2])
        R = np.array([r[2] for r in rr]); Vo = np.array([r[3] for r in rr]); eV = np.array([r[4] for r in rr])
        Vg = np.array([r[5] for r in rr]); Vd = np.array([r[6] for r in rr]); Vb = np.array([r[7] for r in rr])
        SBd = np.array([r[8] for r in rr]); SBb = np.array([r[9] for r in rr])
        vmax = Vo.max()
        nodes, sgn = gasprof[nm]
        Sg = np.interp(R, nodes, sgn)                                 # Msun/pc^2
        jb = abel_density(R, SBb, R) if np.sum(SBb > 0) >= 2 else np.where(SBb > 0, 0.0, 0.0)   # Lsun/pc^2/kpc
        jb = jb / 1000.0                                              # Lsun/pc^3
        V2 = Vg * np.abs(Vg) + UPS_D * Vd * np.abs(Vd) + UPS_B * Vb * np.abs(Vb)
        dV2 = np.gradient(V2, R) if len(R) > 2 else np.zeros_like(R)
        for k in range(len(R)):
            pts.append(dict(name=nm, R=R[k], Vo=Vo[k], eV=eV[k], Vg=Vg[k], Vd=Vd[k], Vb=Vb[k], SBd=SBd[k], SBb=SBb[k],
                            Sg=Sg[k], jb=jb[k], V2=V2[k], dV2=dV2[k], vmax=vmax))
    # vectors
    names = np.array([p["name"] for p in pts])
    R = np.array([p["R"] for p in pts]); Vo = np.array([p["Vo"] for p in pts]); eV = np.array([p["eV"] for p in pts])
    V2 = np.array([p["V2"] for p in pts]); dV2 = np.array([p["dV2"] for p in pts])
    SBd = np.array([p["SBd"] for p in pts]); Sg = np.array([p["Sg"] for p in pts]); jb = np.array([p["jb"] for p in pts])
    vmaxs = np.array([p["vmax"] for p in pts])
    Rdisk = np.array([keep[n]["Rdisk"] for n in names])
    gb = np.where(V2 > 0, V2 / np.maximum(R, 1e-9) * CONV, np.nan)
    go = Vo ** 2 / R * CONV
    y = gb / A0
    res = np.log10(go) - np.log10(nu_mcg(np.where(np.isfinite(y), y, 1.0)) * np.where(np.isfinite(gb), gb, 1.0))
    selA = (Vo > 0) & (R > 0) & (eV / np.maximum(Vo, 1e-9) < 0.10) & (V2 > 0)
    out(f"   {len(pts)} points in {len(set(names))} galaxies; sample A (e_V/V < 0.10, V_bar^2 > 0): {selA.sum()} points in "
        f"{len(set(names[selA]))} galaxies.")

    def rho_mid(shift=0.0, ups=(UPS_D, UPS_B)):
        """Midplane density (Msun/pc^3) of the exponential stellar layer, the gas layer of H_GAS and the bulge."""
        hzs = np.array([hz_of(keep[n], shift) for n in names])
        fac = 2.0
        rs = ups[0] * SBd / (fac * hzs * 1000.0)
        rg = Sg / (2.0 * H_GAS * 1000.0)
        return rs + rg + ups[1] * jb, rs, rg, ups[1] * jb, hzs

    base, bs_, bg_, bb_, hz0 = rho_mid()
    hz_gal = np.array([hz0[selA & (names == u)][0] for u in sorted(set(names[selA]))])
    out(f"   Scale height h_z = 1.12 R_d/{Q_CEN:.2f}: over the sample-A points median {np.median(hz0[selA]):.3f} kpc, "
        f"16-84 percent {np.percentile(hz0[selA],16):.3f} to {np.percentile(hz0[selA],84):.3f};")
    out(f"   over the {len(hz_gal)} sample-A galaxies median {np.median(hz_gal):.3f} kpc, 16-84 percent {np.percentile(hz_gal,16):.3f} "
        f"to {np.percentile(hz_gal,84):.3f}.")
    out(f"   Midplane density over the sample-A points: median {np.median(base[selA]):.4f} Msun/pc^3, 16-84 percent "
        f"{np.percentile(base[selA],16):.4f} to {np.percentile(base[selA],84):.4f}.")

    # K_F: the Hessian of the Newtonian potential at the midplane, exact at z = 0 for an axisymmetric field,
    #   H_RR = (1/R) dV^2/dR - V^2/R^2,  H_phiphi = V^2/R^2,  H_zz = 4 pi G rho_0 - (1/R) dV^2/dR,
    # in density units, K_F = |H|_F/(sqrt(6) G), so that a point mass gives M/r^3
    Rk = np.maximum(R, 1e-6)
    Hpp = np.where(V2 > 0, V2 / Rk ** 2, 0.0)                       # (km/s/kpc)^2
    HRR = dV2 / Rk - V2 / Rk ** 2
    Hzz = 4 * np.pi * G_KPC * base * 1e9 - dV2 / Rk
    HF = np.sqrt(HRR ** 2 + Hpp ** 2 + Hzz ** 2)
    KF = HF / (math.sqrt(6) * G_KPC) / 1e9                         # Msun/pc^3

    # ---------------- data files
    os.makedirs(OUTDIR, exist_ok=True)
    f_pts = os.path.join(OUTDIR, "aqual-disk-points.csv")
    f_gas = os.path.join(OUTDIR, "aqual-disk-gas-profiles.csv")
    with open(f_pts, "w", newline="\n", encoding="utf-8") as fh:
        fh.write("galaxy,R_kpc,Vobs,eV,Vgas,Vdisk,Vbul,SBdisk,SBbul,Sigma_gas,j_bul_Lpc3,vmax,Rdisk,V2bar,dV2dR,y,res,selA,"
                 "hz_kq,rho_star_kq,rho_gas,rho_bul,rho_kq,KF_kq\n")
        for k in range(len(R)):
            fh.write(f"{names[k]},{R[k]:.4f},{Vo[k]:.3f},{eV[k]:.3f},{pts[k]['Vg']:.3f},{pts[k]['Vd']:.3f},{pts[k]['Vb']:.3f},"
                     f"{SBd[k]:.4f},{pts[k]['SBb']:.4f},{Sg[k]:.5f},{jb[k]:.6e},{vmaxs[k]:.2f},{Rdisk[k]:.3f},{V2[k]:.3f},{dV2[k]:.4f},"
                     f"{y[k] if np.isfinite(y[k]) else -1:.6e},{res[k] if np.isfinite(res[k]) else 0:.5f},{int(selA[k])},"
                     f"{hz0[k]:.5f},{bs_[k]:.6e},{bg_[k]:.6e},{bb_[k]:.6e},{base[k]:.6e},{KF[k]:.6e}\n")
    with open(f_gas, "w", newline="\n", encoding="utf-8") as fh:
        fh.write("galaxy,node_R_kpc,Sigma_gas_Msun_pc2\n")
        for nm in sorted(gasprof):
            nodes, sgn = gasprof[nm]
            for a, s in zip(nodes, sgn):
                fh.write(f"{nm},{a:.5f},{s:.6f}\n")
    out(f"\nData files written: {shown_path(f_pts)}, {shown_path(f_gas)}")
    out(f"\n[run time {time.time()-T0:.0f} s]")


if __name__ == "__main__":
    main()
