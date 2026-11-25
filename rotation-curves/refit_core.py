"""Core of the AQUAL refits: the SPARC data and the fit machinery of two_constant_fit.pl in numpy, with the full AQUAL
solution entered through the departure Gamma tabulated by aqual-refit-solve.py. A library: it prints nothing.

The machinery is that of two_constant_fit.pl, the fits of Paper I, Sect. 4.1 and 4.2:
  - sample: SPARC galaxies with quality flag not 3, inclination at least 30 degrees and at least 3 points (153
    galaxies, 3,168 points); a point enters a galaxy's chi^2 only while its baryonic acceleration is positive;
  - chi^2 = sum w (V_obs - q V_pred)^2, w = 1/(eV^2 + (0.05 V_obs)^2), V_pred = sqrt(g_pred R) (Paper I, Eq. (14));
  - the disk ratio, and the bulge ratio of each galaxy with a bulge, free in the box 0.03 to 10, with lognormal
    priors of 0.10 dex about 0.5 and 0.7 (Li et al. 2018);
  - q = sqrt(f) sin(i)/sin(i_c), f = D/D_cat, profiled exactly at every trial ratio under the prior of Paper I,
    Eq. (15), P(q) = min over i of [((f-1)/sig_f)^2 + ((i-i_c)/sig_i)^2], sig_f = max(e_D/D, 0.02),
    sig_i = max(e_Inc, 0.5 deg), on q bounds from f in 1 +/- 4 sig_f and i in i_c +/- 4 sig_i; P is tabulated per
    galaxy (2,001 nodes, each the golden-section minimum over i, 60 steps, as in two_constant_fit.pl);
  - every galaxy converged: a coarse grid over the whole box (51 nodes for one ratio, 21 x 21 for two, as in
    two_constant_fit.pl), the three lowest local minima refined by a stencil descent (a 5-point or 5 x 5 stencil about
    each centre and the vertex of the least-squares quadratic through it, the stencil shrinking by 4 or 3 per round),
    then restarted from the optimum until a round gains less than 1e-7;
  - a0 by a parabola in log10 a0 through converged totals (the ladder of two_constant_fit.pl, fit_a0), and the free
    volume delta of the boost law by the parabola through the three lowest delta nodes, each node with its a0 refitted;
  - errors clustered by galaxy from 2,000 bootstrap draws, each refitting every law along its stored profile (a
    parabola in log10 a0 at each node, the three-lowest-node parabola across delta), as in two_constant_fit.pl.
The port differs from two_constant_fit.pl in its search (the stencil descent instead of Brent and Nelder-Mead) and in
its random numbers. It reproduces the totals of two_constant_fit.pl (aqual-refit-fit.py --part machinery).

Law codes: T11 is the empirical function of McGaugh, Lelli and Schombert, nu(y) = 1/(1 - exp(-sqrt(y))); T3 is the
boost law of Paper I, Eq. (8), at free volume delta.

Two baryonic inputs are offered for each law:
  'sparc': SPARC's tabulated g_bar = gg + Yd gd + Yb gb (the input of two_constant_fit.pl);
  'model': the Newtonian midplane field of the disk models of finite thickness of aqual-disk-solver.py,
           g_N = mg + Yd md + Yb mb, its disk and bulge parts per unit ratio and its gas fixed (aqual-refit-solve.py).
The algebraic relation predicts g = nu(g_b/a0) g_b. The full AQUAL solution predicts g = nu(g_b/a0) g_b 10^Gamma, with
Gamma the departure computed for the model at the prior centres (Yd 0.5, Yb 0.7). On the model's field this is the
full solution itself at the prior centres. At other ratios Gamma is carried by the exact scaling property
Gamma(s rho, a0) = Gamma(rho, a0/s): it is read at a0/s, with s = g_N(Yd, Yb)/g_N(0.5, 0.7) at the point (mode
'scaled'). The step is exact for a uniform change of all masses and approximate when only the stars change. Mode
'corrected', used for every deposited result, adds the scaled reading's remaining error, measured against direct
solutions at other ratios (aqual-refit-solve.py parts upsvar and upsvar2; attach_eps and eps_corr): over the disk
ratio from -1.0 to +1.0 dex and, for bulge galaxies, over the grid of disk ratios -0.4 to +0.8 dex by bulge ratios
-0.8 to +0.8 dex, for T11 and for the boost law at delta 0.03 and 0.3. Mode 'fixed' keeps Gamma at the prior
centres. Mode 'exact' takes Gamma per point from direct solutions at the evaluated ratios (constrained_solve.py part
atopt), for evaluation at fixed ratios only.
"""
import csv, math, os, re
import numpy as np

CONV = 3.24078e-14                       # (km/s)^2/kpc -> m/s^2
FLOOR = 0.05
LGY0, LGYB0 = math.log10(0.5), math.log10(0.7)
PRIOR = 0.10
ULO, UHI = math.log10(0.03), math.log10(10.0)
D2R = math.pi / 180.0
GR = 0.6180339887
NQ = 2000                                # intervals of the P(q) table
NQC = 40                                 # intervals of the coarse q search (every 50th table node)

LA_T11 = np.array([round(-10.7 + 0.1 * k, 1) for k in range(15)])
LA_T3 = np.array([round(-10.7 + 0.2 * k, 1) for k in range(8)])
T3_DELTAS = np.array([0.05, 0.2, 0.4, 0.7, 1.2])
EXTRAP_DEX = 0.5                         # Gamma is extrapolated linearly at most this far beyond the a0 nodes
EPS_D = np.round(np.arange(-1.0, 1.0001, 0.2), 1)             # disk grid of the direct solutions behind 'corrected'
EPS_D2 = np.round(np.arange(-0.4, 0.8001, 0.2), 1)             # disk ratios of the bulge galaxies' grid
EPS_B = np.round(np.arange(-0.8, 0.8001, 0.2), 1)              # bulge ratios of that grid


# ------------------------------------------------------------------ the laws, as two_constant_fit.pl evaluates them
def nu_t11(y):
    y = np.maximum(y, 1e-300)
    return 1.0 / (-np.expm1(-np.sqrt(y)))


class NuTable:
    """The tabulation of nu for a law given by mu (two_constant_fit.pl build_table and tab_nu):
    log10 nu at log10 y = -9 + 0.02 k, k = 0..900, from X mu(X) = y solved by 70 bisections in log X on
    [1e-10, 1e10]; linear interpolation in log10 y, extrapolated linearly from the end intervals."""
    LY0, STEP, NT = -9.0, 0.02, 900

    def __init__(self, mu):
        y = 10.0 ** (self.LY0 + np.arange(self.NT + 1) * self.STEP)
        lo = np.full_like(y, 1e-10); hi = np.full_like(y, 1e10)
        for _ in range(70):
            m = np.sqrt(lo * hi)
            low = m * mu(m) < y
            lo = np.where(low, m, lo); hi = np.where(low, hi, m)
        X = np.sqrt(lo * hi)
        self.t = np.log10(X / y)

    def __call__(self, y):
        x = (np.log10(np.maximum(y, 1e-300)) - self.LY0) / self.STEP
        i = np.clip(np.trunc(x).astype(np.int64), 0, self.NT - 1)
        f = x - i
        return 10.0 ** (self.t[i] * (1 - f) + self.t[i + 1] * f)


def mu_boost(delta):
    """The boost law of Paper I, Eq. (8): mu(X) = E/(E + delta), E = exp(lambda X) - 1, lambda = ln(1 + delta)."""
    lam = math.log1p(delta)

    def mu(X):
        u = lam * X
        E = np.expm1(np.minimum(u, 40.0))
        return np.where(u > 40.0, 1.0, E / (E + delta))
    return mu


_NU_CACHE = {}


def law_nu(code, delta=None):
    key = (code, None if delta is None else round(float(delta), 8))
    if key not in _NU_CACHE:
        _NU_CACHE[key] = nu_t11 if code == "T11" else NuTable(mu_boost(float(delta)))
    return _NU_CACHE[key]


# ------------------------------------------------------------------ data
def read_table1(path):
    """The parser of SPARC Table 1 of two_constant_fit.pl (distance, its error, inclination, its error, quality), plus SBeff,
    L[3.6] and M_HI for the galaxy classes."""
    T = {}
    rule = 0
    for line in open(path, encoding="ascii", errors="replace"):
        l = line.rstrip("\r\n")
        if re.fullmatch(r"-+", l):
            rule += 1
            continue
        if rule < 4:
            continue
        f = l.split()
        if len(f) < 18 or not re.match(r"[A-Za-z0-9]", f[0]):
            continue
        if not re.fullmatch(r"[\d.]+", f[5]) or not re.fullmatch(r"\d+", f[17]):
            continue
        T[f[0]] = dict(D=float(f[2]), eD=float(f[3]), inc=float(f[5]), einc=float(f[6]), Q=int(f[17]),
                       L36=float(f[7]), SBeff=float(f[10]), MHI=float(f[13]))
    return T


def read_rotmod(rotdir, T1):
    """The sample of Paper I, Sect. 4.1: rotmod files of galaxies in Table 1 with Q != 3 and inclination >= 30, >= 3 points."""
    gals = {}
    for fn in sorted(os.listdir(rotdir)):
        if not fn.endswith("_rotmod.dat"):
            continue
        name = fn[:-len("_rotmod.dat")]
        if name not in T1 or T1[name]["Q"] == 3 or T1[name]["inc"] < 30:
            continue
        pts = []
        for line in open(os.path.join(rotdir, fn), encoding="ascii", errors="replace"):
            if line.startswith("#"):
                continue
            c = line.split()
            if len(c) < 6 or not re.match(r"[\d.]", c[0]) or float(c[0]) <= 0:
                continue
            pts.append([float(x) for x in c[:6]])
        if len(pts) < 3:
            continue
        a = np.array(pts)
        t = T1[name]
        gals[name] = dict(name=name, R=a[:, 0], Vo=a[:, 1], eV=a[:, 2], Vg=a[:, 3], Vd=a[:, 4], Vb=a[:, 5],
                          hasb=bool(np.any(a[:, 5] != 0)), ic=t["inc"], si=max(t["einc"], 0.5),
                          sf=max(t["eD"] / t["D"], 0.02), T1=t)
    return gals


def read_gamma(paths):
    rows = []
    for p in paths:
        rows += list(csv.DictReader(open(p, encoding="utf-8-sig")))
    G = {}
    for r in rows:
        G[(r["galaxy"], round(float(r["R"]), 4))] = r
    return G


def read_points(path):
    P = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        P[(r["galaxy"], round(float(r["R_kpc"]), 4))] = r
    return P


def attach(gals, GAM, PTS, strict=True):
    """Attach the model's field components, the Gamma tables and columns of aqual-disk-points.csv to each galaxy.
    With strict=False, points without solver rows get NaN (they are then excluded wherever the model is used)."""
    nanrow = {k: "nan" for k in (["gN", "gNd", "gNg", "gNb"] + [f"G_T11_{la:.1f}" for la in LA_T11]
                                  + [f"G_T3_{d:g}_{la:.1f}" for d in T3_DELTAS for la in LA_T3])}
    for name, g in gals.items():
        keys = [(name, round(float(R), 4)) for R in g["R"]]
        miss = [k for k in keys if k not in GAM or k not in PTS]
        if miss and (strict or any(k not in PTS for k in keys)):
            raise KeyError(f"{name}: {len(miss)} points without solver or points-file rows")
        rr = [GAM.get(k, nanrow) for k in keys]
        pp = [PTS[k] for k in keys]
        f = lambda c: np.array([float(r[c]) for r in rr])
        g["gN0"] = f("gN") * CONV
        g["md"] = f("gNd") / 0.5 * CONV
        g["mg"] = f("gNg") * CONV
        g["mb"] = f("gNb") / 0.7 * CONV
        g["gam11"] = np.array([[float(r[f"G_T11_{la:.1f}"]) for la in LA_T11] for r in rr])
        g["gam3"] = np.array([[[float(r[f"G_T3_{d:g}_{la:.1f}"]) for la in LA_T3] for d in T3_DELTAS] for r in rr])
        pf = lambda c: np.array([float(p[c]) for p in pp])
        g["selA"] = pf("selA") > 0.5
        g["Sigma"] = 0.5 * pf("SBdisk") + 0.7 * pf("SBbul") + pf("Sigma_gas")          # Msun/pc^2, at the prior centres
        g["y16"] = pf("y")                                                              # SPARC g_bar/1.2e-10 at 0.5/0.7
    return gals


# ------------------------------------------------------------------ the prior P(q), vectorized golden section
def golden_vec(cost, a, b, n):
    """The golden-section minimizer of two_constant_fit.pl (sub golden), applied elementwise."""
    c = b - GR * (b - a); d = a + GR * (b - a)
    fc, fd = cost(c), cost(d)
    for _ in range(n):
        m = fc < fd
        a, b = np.where(m, a, c), np.where(m, d, b)
        c_new = np.where(m, b - GR * (b - a), d)
        d_new = np.where(m, c, a + GR * (b - a))
        c, d = c_new, d_new
        fc, fd = cost(c), cost(d)
    x = 0.5 * (a + b)
    return cost(x)


def split_prior(q, ic, si, sf, n=60):
    sic = np.sin(ic * D2R)
    ilo = np.maximum(ic - 4.0 * si, 10.0); ihi = np.minimum(ic + 4.0 * si, 89.9)
    cost = lambda i: ((q * sic / np.sin(i * D2R)) ** 2 - 1.0) ** 2 / sf ** 2 + ((i - ic) / si) ** 2
    return golden_vec(cost, ilo + 0 * q, ihi + 0 * q, n)


def qbounds(ic, si, sf):
    sic = math.sin(ic * D2R)
    flo = max(1.0 - 4.0 * sf, 0.1); fhi = 1.0 + 4.0 * sf
    ilo = max(ic - 4.0 * si, 10.0); ihi = min(ic + 4.0 * si, 89.9)
    return math.sqrt(flo) * math.sin(ilo * D2R) / sic, math.sqrt(fhi) * math.sin(ihi * D2R) / sic


# ------------------------------------------------------------------ Gamma interpolation
def cr_interp(tab, x):
    """Catmull-Rom cubic on uniform nodes; tab (N, n), x (N, m) in node units; linear extrapolation beyond the ends
    (capped by the caller)."""
    n = tab.shape[1]
    xl = np.clip(x, 0.0, n - 1.0)
    k = np.clip(np.floor(xl).astype(np.int64), 0, n - 2)
    t = xl - k
    tk = lambda j: np.take_along_axis(tab, np.clip(j, 0, n - 1), axis=1)
    y0, y1 = tk(k), tk(k + 1)
    ym = np.where(k >= 1, tk(k - 1), 2 * y0 - y1)
    y2 = np.where(k + 2 <= n - 1, tk(k + 2), 2 * y1 - y0)
    v = 0.5 * (2 * y0 + (-ym + y1) * t + (2 * ym - 5 * y0 + 4 * y1 - y2) * t * t + (-ym + 3 * y0 - 3 * y1 + y2) * t ** 3)
    lo = tab[:, :1] + x * (tab[:, 1:2] - tab[:, :1])
    hi = tab[:, -1:] + (x - (n - 1)) * (tab[:, -1:] - tab[:, -2:-1])
    return np.where(x < 0, lo, np.where(x > n - 1, hi, v))


def _cr_weights(t, k, n):
    """Catmull-Rom weights and node indices for positions k + t on n uniform nodes; a ghost node beyond either end is
    the linear extrapolation of the last two (as cr_interp)."""
    w = [0.5 * (-t + 2 * t * t - t ** 3), 0.5 * (2 - 5 * t * t + 3 * t ** 3), 0.5 * (t + 4 * t * t - 3 * t ** 3),
         0.5 * (-t * t + t ** 3)]
    idx = [k - 1, k, k + 1, k + 2]
    lo = idx[0] < 0
    w[1] = np.where(lo, w[1] + 2 * w[0], w[1]); w[2] = np.where(lo, w[2] - w[0], w[2]); w[0] = np.where(lo, 0.0, w[0])
    hi = idx[3] > n - 1
    w[2] = np.where(hi, w[2] + 2 * w[3], w[2]); w[1] = np.where(hi, w[1] - w[3], w[1]); w[3] = np.where(hi, 0.0, w[3])
    return w, [np.clip(i, 0, n - 1) for i in idx]


def cr2_interp(tab, x, y):
    """Bicubic Catmull-Rom on a uniform grid: tab (N, nx, ny), x and y (N, m) in node units, inside the grid."""
    N, nx, ny = tab.shape
    xc = np.clip(x, 0.0, nx - 1.0); yc = np.clip(y, 0.0, ny - 1.0)
    kx = np.clip(np.floor(xc).astype(np.int64), 0, nx - 2); ky = np.clip(np.floor(yc).astype(np.int64), 0, ny - 2)
    wx, ix = _cr_weights(xc - kx, kx, nx)
    wy, iy = _cr_weights(yc - ky, ky, ny)
    flat = tab.reshape(N, nx * ny)
    v = 0.0
    for a in range(4):
        for b in range(4):
            v = v + wx[a] * wy[b] * np.take_along_axis(flat, ix[a] * ny + iy[b], axis=1)
    return v


def delta_weights(delta):
    """Lagrange weights of the quartic through the five delta nodes."""
    w = np.ones(len(T3_DELTAS))
    for i, xi in enumerate(T3_DELTAS):
        for j, xj in enumerate(T3_DELTAS):
            if j != i:
                w[i] *= (delta - xj) / (xi - xj)
    return w


# ------------------------------------------------------------------ a group of galaxies with the same number of ratios
class Group:
    def __init__(self, gl, two, pmask=None):
        self.gl, self.two, self.G = gl, two, len(gl)
        if self.G == 0:
            return
        sel = [np.ones(len(g["R"]), bool) if pmask is None else pmask[g["name"]] for g in gl]
        cat = lambda key: np.concatenate([g[key][s] for g, s in zip(gl, sel)])
        self.npt = np.array([int(s.sum()) for s in sel], dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.npt)[:-1]])
        self.gid = np.repeat(np.arange(self.G), self.npt)
        self.sel = sel
        R = cat("R"); Vg, Vd, Vb = cat("Vg"), cat("Vd"), cat("Vb")
        self.R = R
        self.gg = Vg * np.abs(Vg) / R * CONV
        self.gd = Vd * Vd / R * CONV
        self.gb = Vb * Vb / R * CONV
        self.Vo = cat("Vo"); self.eV = cat("eV")
        self.w = 1.0 / (self.eV ** 2 + (FLOOR * self.Vo) ** 2)
        self.sR = np.sqrt(R / CONV)
        self.mg, self.md, self.mb, self.gN0 = cat("mg"), cat("md"), cat("mb"), cat("gN0")
        self.gam11 = np.concatenate([g["gam11"][s] for g, s in zip(gl, sel)])
        self.gam3 = np.concatenate([g["gam3"][s] for g, s in zip(gl, sel)])
        # points where the model's field at the prior centres is positive and every Gamma node is defined
        self.gok = (self.gN0 > 0) & np.all(np.isfinite(self.gam11), axis=1) & np.all(np.isfinite(self.gam3), axis=(1, 2))
        self.gam11 = np.where(np.isfinite(self.gam11), self.gam11, 0.0)
        self.gam3 = np.where(np.isfinite(self.gam3), self.gam3, 0.0)
        self.Sigma, self.y16 = cat("Sigma"), cat("y16")
        self.ic = np.array([g["ic"] for g in gl]); self.si = np.array([g["si"] for g in gl])
        self.sf = np.array([g["sf"] for g in gl])
        qb = np.array([qbounds(g["ic"], g["si"], g["sf"]) for g in gl])
        self.qlo, self.qhi = qb[:, 0], qb[:, 1]
        self.dq = (self.qhi - self.qlo) / NQ
        qn = self.qlo[:, None] + self.dq[:, None] * np.arange(NQ + 1)[None, :]
        self.P = split_prior(qn, self.ic[:, None], self.si[:, None], self.sf[:, None], 60)
        self.qc = qn[:, ::NQ // NQC]; self.Pc = self.P[:, ::NQ // NQC]

    # P(q), P'(q), P''(q) from the table by a local parabola through the nearest three nodes
    def Pq(self, q):
        """q of shape (G, m)."""
        d = self.dq[:, None]
        x = (q - self.qlo[:, None]) / d
        k = np.clip(np.rint(x).astype(np.int64), 1, NQ - 1)
        t = x - k
        p0 = np.take_along_axis(self.P, k, axis=1)
        pm = np.take_along_axis(self.P, k - 1, axis=1); pp = np.take_along_axis(self.P, k + 1, axis=1)
        a1 = 0.5 * (pp - pm); a2 = 0.5 * (pp - 2 * p0 + pm)
        return p0 + a1 * t + a2 * t * t, (a1 + 2 * a2 * t) / d, 2 * a2 / (d * d)


# ------------------------------------------------------------------ the dataset and its fits
class Dataset:
    """A sample of galaxies (sorted names) with optional point masks, fitted by the machinery of two_constant_fit.pl."""

    def __init__(self, gals, names=None, pmask=None, frozen=False):
        names = sorted(gals) if names is None else list(names)
        self.names = names
        self.frozen = frozen
        one = [gals[n] for n in names if not gals[n]["hasb"]]
        two = [gals[n] for n in names if gals[n]["hasb"]]
        self.groups = [Group(one, False, pmask), Group(two, True, pmask)]
        order = [g["name"] for g in one] + [g["name"] for g in two]
        self.pos = {n: i for i, n in enumerate(order)}
        self.idx = np.array([self.pos[n] for n in names])     # names order -> group order
        self.hasb = np.array([gals[n]["hasb"] for n in names])
        self.evals = 0

    # --- the prediction at trial ratios
    def gpred(self, grp, U, V, ctx):
        """g_pred (m/s^2) at trial log ratios U (and V), shapes (G, m) -> (Npts, m), and the validity mask."""
        Yd = 10.0 ** U[grp.gid]
        Yb = 10.0 ** V[grp.gid] if grp.two else 0.0
        if ctx["input"] == "model":
            gbar = grp.mg[:, None] + Yd * grp.md[:, None] + (Yb * grp.mb[:, None] if grp.two else 0.0)
        else:
            gbar = grp.gg[:, None] + Yd * grp.gd[:, None] + (Yb * grp.gb[:, None] if grp.two else 0.0)
        valid = gbar > 0
        if ctx.get("use_gok", True):
            valid &= grp.gok[:, None]          # points where the model's field and Gamma are defined
        if "pmask" in ctx:
            valid &= ctx["pmask"][grp is self.groups[1]][:, None]
        gbs = np.where(valid, gbar, 1.0)
        g = ctx["nu"](gbs / ctx["a0"]) * gbs
        if ctx.get("full"):
            g = g * 10.0 ** self.gamma(grp, Yd, Yb, ctx)
        return g, valid

    def logg(self, grp, U, V, ctx):
        g, valid = self.gpred(grp, U, V, ctx)
        return np.log10(g), valid

    def la_eff(self, grp, Yd, Yb, ctx):
        """log10 of the a0 at which Gamma is read: a0/s in modes 'scaled' and 'corrected', a0 in mode 'fixed'."""
        la = math.log10(ctx["a0"])
        if ctx.get("gmode", "scaled") in ("scaled", "corrected"):
            gm = grp.mg[:, None] + Yd * grp.md[:, None] + (Yb * grp.mb[:, None] if grp.two else 0.0)
            s = np.where((gm > 0) & grp.gok[:, None], gm / np.where(grp.gok, grp.gN0, 1.0)[:, None], 1.0)
            return la - np.log10(s)
        return la + 0.0 * Yd

    def gamma(self, grp, Yd, Yb, ctx):
        mode = ctx.get("gmode", "scaled")
        if mode == "exact":                    # direct solutions at the evaluated ratios (constrained_fit.py), per point
            return ctx["gexact"][grp is self.groups[1]][:, None] + 0.0 * Yd
        lae = self.la_eff(grp, Yd, Yb, ctx)
        tab, la0, step = self.gtab(grp, ctx)
        x = (lae - la0) / step
        nmax = tab.shape[1] - 1
        cap = EXTRAP_DEX / step
        x = np.clip(x, -cap, nmax + cap)
        G = cr_interp(tab, x)
        if mode == "corrected":
            G = G + self.eps_corr(grp, Yd, Yb, ctx)
        return G

    def eps_corr(self, grp, Yd, Yb, ctx):
        """Mode 'corrected': the scaled reading plus, point by point, its error against direct solutions at other
        ratios (attach_eps). For T3 the corrections at delta 0.03 and 0.3 are interpolated linearly in delta and held
        constant outside that range."""
        if ctx["law"] == "T11":
            return self._eps_tag(grp, Yd, Yb, grp.eps["T11"])
        have = [t for t in ("T3d0.03", "T3d0.3") if t in grp.eps]
        if len(have) == 1:
            return self._eps_tag(grp, Yd, Yb, grp.eps[have[0]])
        w = min(max((ctx["delta"] - 0.03) / 0.27, 0.0), 1.0)
        return (1 - w) * self._eps_tag(grp, Yd, Yb, grp.eps["T3d0.03"]) + w * self._eps_tag(grp, Yd, Yb, grp.eps["T3d0.3"])

    def _eps_tag(self, grp, Yd, Yb, E):
        """The disk ratio enters through a Catmull-Rom interpolation over the disk grid (held constant beyond it).
        For bulge galaxies with the full grid of disk by bulge ratios, a bicubic Catmull-Rom over that grid replaces it,
        plus the disk grid's change beyond the grid's disk range; without that grid, the bulge ratio enters by the
        quadratic through its +/-0.2 dex solutions, held constant beyond +/-0.3 dex."""
        du = np.log10(Yd) - LGY0
        lo, hi = E["drange"]
        e1 = lambda u: cr_interp(E["d1"], (np.clip(u, lo, hi) - lo) / 0.2)
        if not grp.two:
            return e1(du)
        dv = np.log10(Yb) - LGYB0
        if "d2" in E:
            duc = np.clip(du, EPS_D2[0], EPS_D2[-1])
            c = cr2_interp(E["d2"], (duc - EPS_D2[0]) / 0.2, (np.clip(dv, EPS_B[0], EPS_B[-1]) - EPS_B[0]) / 0.2)
            out = (du < EPS_D2[0]) | (du > EPS_D2[-1])
            if np.any(out):
                c = c + np.where(out, e1(du) - e1(duc), 0.0)
            return c
        c = e1(du)
        dvq = np.clip(dv, -0.3, 0.3)
        a2 = (E["bp"] - E["bm"])[:, None] / 0.4; b2 = (E["bp"] + E["bm"])[:, None] / 0.08
        return c + a2 * dvq + b2 * dvq * dvq

    def attach_eps(self, rows):
        """The errors of the scaled reading against the direct solutions (solve parts upsvar and upsvar2), per point,
        for each law tag present (T11; T3d0.3 and T3d0.03, the boost law at delta 0.3 and 0.03): on the disk grid
        (dlog10 Yd from -1.0 to +1.0 dex in steps of 0.2, the widest contiguous range present) and, for bulge
        galaxies, on the grid of dlog10 Yd -0.4 to +0.8 by dlog10 Yb -0.8 to +0.8 dex when it is complete."""
        D = {}
        refs = {}
        for r in rows:
            D[(r["galaxy"], round(float(r["R"]), 4), r["law"], round(float(r["dlogYd"]), 2), round(float(r["dlogYb"]), 2))] = float(r["Gamma"])
            refs.setdefault(r["law"], float(r["a0"]))
        for grp in self.groups:
            if grp.G == 0:
                continue
            grp.eps = {}
            keys = [(g["name"], round(float(R), 4)) for g, s in zip(grp.gl, grp.sel) for R in g["R"][s]]
            for tag in ("T11", "T3d0.3", "T3d0.03"):
                if tag not in refs:
                    continue
                law, delta = ("T11", None) if tag == "T11" else ("T3", float(tag[3:]))
                ctx = dict(input="model", law=law, delta=delta, a0=refs[tag], gmode="scaled", full=True, _cache={})

                def err_at(du, dv):
                    U = np.full((grp.G, 1), LGY0 + du); V = np.full((grp.G, 1), LGYB0 + dv)
                    Yd = 10.0 ** U[grp.gid]; Yb = 10.0 ** V[grp.gid] if grp.two else 0.0
                    Gd = np.array([D.get((k[0], k[1], tag, round(du, 2), round(dv, 2)), np.nan) for k in keys])
                    if not np.any(np.isfinite(Gd)):
                        return None
                    Gs = self.gamma(grp, Yd, Yb, ctx)[:, 0]
                    return np.where(np.isfinite(Gd) & grp.gok, Gd - Gs, 0.0)
                E = {}
                col = {0.0: np.zeros(len(keys))}
                for du in EPS_D:
                    if abs(du) > 1e-9:
                        e = err_at(float(du), 0.0)
                        if e is not None:
                            col[round(float(du), 1)] = e
                lo = 0.0
                while round(lo - 0.2, 1) in col:
                    lo = round(lo - 0.2, 1)
                hi = 0.0
                while round(hi + 0.2, 1) in col:
                    hi = round(hi + 0.2, 1)
                E["drange"] = (lo, hi)
                E["d1"] = np.stack([col[round(lo + 0.2 * k, 1)] for k in range(int(round((hi - lo) / 0.2)) + 1)], axis=1)
                if grp.two:
                    for tg, dv in (("bp", 0.2), ("bm", -0.2)):
                        e = err_at(0.0, dv)
                        E[tg] = e if e is not None else np.zeros(len(keys))
                    T2, ok = np.zeros((len(keys), len(EPS_D2), len(EPS_B))), True
                    for i, du in enumerate(EPS_D2):
                        for j, dv in enumerate(EPS_B):
                            if abs(dv) < 1e-9:
                                if round(float(du), 1) in col:
                                    T2[:, i, j] = col[round(float(du), 1)]
                                else:
                                    ok = False
                                continue
                            e = err_at(float(du), float(dv))
                            if e is None:
                                ok = False
                            else:
                                T2[:, i, j] = e
                    if ok:
                        E["d2"] = T2
                grp.eps[tag] = E

    def gtab(self, grp, ctx):
        key = ("gtab", id(grp), ctx["law"], ctx.get("delta"))
        cache = ctx.setdefault("_cache", {})
        if key not in cache:
            if ctx["law"] == "T11":
                cache[key] = (grp.gam11, LA_T11[0], 0.1)
            else:
                w = delta_weights(ctx["delta"])
                cache[key] = (np.tensordot(grp.gam3, w, axes=([1], [0])), LA_T3[0], 0.2)
        return cache[key]

    # --- the objective with q profiled exactly
    def objective(self, grp, U, V, ctx):
        self.evals += U.size
        g, valid = self.gpred(grp, U, V, ctx)
        Vo = ctx["Vo"][grp is self.groups[1]] if "Vo" in ctx else grp.Vo
        w = np.where(valid, grp.w[:, None], 0.0)
        Vp = np.sqrt(g) * grp.sR[:, None]
        wVp = w * Vp
        Soo = np.add.reduceat(w * (Vo * Vo)[:, None], grp.starts, axis=0)
        Sop = np.add.reduceat(wVp * Vo[:, None], grp.starts, axis=0)
        Spp = np.add.reduceat(wVp * Vp, grp.starts, axis=0)
        pri = ((U - LGY0) / PRIOR) ** 2
        if grp.two:
            pri = pri + ((V - LGYB0) / PRIOR) ** 2
        if self.frozen:
            return pri + Soo - 2 * Sop + Spp, np.ones_like(U)
        q, h = self.qprofile(grp, Sop, Spp)
        return pri + Soo + h, q

    def qprofile(self, grp, Sop, Spp):
        """min over q of Spp q^2 - 2 Sop q + P(q): the coarse table nodes, then Newton steps inside the bracket of the
        best node, with P from the fine table."""
        G, m = Sop.shape
        qc, Pc = grp.qc, grp.Pc
        hc = Spp[:, :, None] * qc[:, None, :] ** 2 - 2 * Sop[:, :, None] * qc[:, None, :] + Pc[:, None, :]
        j = np.argmin(hc, axis=2)
        hmin = np.take_along_axis(hc, j[:, :, None], axis=2)[:, :, 0]
        gi = np.repeat(np.arange(G)[:, None], m, axis=1)
        qa = qc[gi, np.maximum(j - 1, 0)]; qb = qc[gi, np.minimum(j + 1, NQC)]
        q0 = qc[gi, j]
        q = q0
        for _ in range(6):
            P, P1, P2 = grp.Pq(q)
            g1 = 2 * Spp * q - 2 * Sop + P1; g2 = 2 * Spp + P2
            step = np.where(g2 > 0, -g1 / np.where(g2 > 0, g2, 1.0), -np.sign(g1) * (qb - qa) / 8)
            q = np.clip(q + step, qa, qb)
        P, _, _ = grp.Pq(q)
        h = Spp * q * q - 2 * Sop * q + P
        better = h <= hmin
        return np.where(better, q, q0), np.where(better, h, hmin)

    # --- the converged search over the ratios
    def _descend(self, grp, ctx, X, FX, h0, rounds, shrink):
        """Stencil descent from centres X (G, S, k) with values FX (G, S): each round evaluates a 5-point (k = 1) or
        5 x 5 (k = 2) stencil of spacing h about every centre and the vertex of the least-squares quadratic through it,
        moves each centre to the best point seen, and divides h by shrink."""
        G, S, k = X.shape
        t = np.array([-2.0, -1.0, 1.0, 2.0]) if k == 1 else None
        if k == 2:
            s_, t_ = np.meshgrid(np.arange(-2.0, 3.0), np.arange(-2.0, 3.0), indexing="ij")
            s_, t_ = s_.ravel(), t_.ravel()
            keep = (s_ != 0) | (t_ != 0)
            OS, OT = s_[keep], t_[keep]                     # 24 offsets
        h = h0
        for _ in range(rounds):
            if k == 1:
                cu = np.clip(X[:, :, 0][:, :, None] + h * t[None, None, :], ULO, UHI)       # (G, S, 4)
                Fc, _ = self.objective(grp, cu.reshape(G, -1), None, ctx)
                Fc = Fc.reshape(G, S, 4)
                ym2, ym1, yp1, yp2 = Fc[..., 0], Fc[..., 1], Fc[..., 2], Fc[..., 3]
                b = (-2 * ym2 - ym1 + yp1 + 2 * yp2) / 10.0
                c = (2 * ym2 - ym1 - 2 * FX - yp1 + 2 * yp2) / 14.0
                tv = np.where(c > 0, np.clip(-b / (2 * np.where(c > 0, c, 1.0)), -2.5, 2.5), 0.0)
                xv = np.clip(X[:, :, 0] + h * tv, ULO, UHI)
                Fv, _ = self.objective(grp, xv.reshape(G, -1), None, ctx)
                Fv = Fv.reshape(G, S)
                allx = np.concatenate([X[:, :, 0][..., None], cu, xv[..., None]], axis=2)   # (G, S, 6)
                allf = np.concatenate([FX[..., None], Fc, Fv[..., None]], axis=2)
                kb = np.argmin(allf, axis=2)
                X = np.take_along_axis(allx, kb[..., None], axis=2)
                FX = np.take_along_axis(allf, kb[..., None], axis=2)[..., 0]
            else:
                cu = np.clip(X[:, :, 0][:, :, None] + h * OS[None, None, :], ULO, UHI)       # (G, S, 24)
                cv = np.clip(X[:, :, 1][:, :, None] + h * OT[None, None, :], ULO, UHI)
                Fc, _ = self.objective(grp, cu.reshape(G, -1), cv.reshape(G, -1), ctx)
                Fc = Fc.reshape(G, S, 24)
                full = np.zeros((G, S, 25)); full[..., keep] = Fc; full[..., ~keep] = FX[..., None]
                b1 = (full * s_).sum(-1) / 50.0; b2 = (full * t_).sum(-1) / 50.0
                c11 = (full * (s_ ** 2 - 2)).sum(-1) / 70.0; c22 = (full * (t_ ** 2 - 2)).sum(-1) / 70.0
                c12 = (full * (s_ * t_)).sum(-1) / 100.0
                H11, H22, H12 = 2 * c11, 2 * c22, c12
                det = H11 * H22 - H12 * H12
                pd = (H11 > 0) & (det > 0)
                dsafe = np.where(pd, det, 1.0)
                ds = np.where(pd, -(H22 * b1 - H12 * b2) / dsafe, 0.0)
                dt = np.where(pd, -(H11 * b2 - H12 * b1) / dsafe, 0.0)
                ds, dt = np.clip(ds, -2.5, 2.5), np.clip(dt, -2.5, 2.5)
                xu = np.clip(X[:, :, 0] + h * ds, ULO, UHI); xv = np.clip(X[:, :, 1] + h * dt, ULO, UHI)
                Fv, _ = self.objective(grp, xu.reshape(G, -1), xv.reshape(G, -1), ctx)
                Fv = Fv.reshape(G, S)
                allu = np.concatenate([X[:, :, 0][..., None], cu, xu[..., None]], axis=2)
                allv = np.concatenate([X[:, :, 1][..., None], cv, xv[..., None]], axis=2)
                allf = np.concatenate([FX[..., None], Fc, Fv[..., None]], axis=2)
                kb = np.argmin(allf, axis=2)
                X = np.stack([np.take_along_axis(allu, kb[..., None], axis=2)[..., 0],
                              np.take_along_axis(allv, kb[..., None], axis=2)[..., 0]], axis=2)
                FX = np.take_along_axis(allf, kb[..., None], axis=2)[..., 0]
            h /= shrink
        return X, FX

    def fit_group(self, grp, ctx):
        G = grp.G
        if G == 0:
            return dict(F=np.zeros(0), u=np.zeros(0), v=np.zeros(0), q=np.zeros(0))
        if not grp.two:
            ug = ULO + (UHI - ULO) * np.arange(51) / 50.0
            F, _ = self.objective(grp, np.tile(ug, (G, 1)), None, ctx)
            lm = np.ones_like(F, bool)
            lm[:, 1:] &= F[:, 1:] <= F[:, :-1]
            lm[:, :-1] &= F[:, :-1] <= F[:, 1:]
            Fm = np.where(lm, F, np.inf)
            o = np.argsort(Fm, axis=1, kind="stable")[:, :3]
            ok = np.isfinite(np.take_along_axis(Fm, o, axis=1))
            o = np.where(ok, o, o[:, :1])
            X = ug[o][..., None]; FX = np.take_along_axis(F, o, axis=1)
            span0 = (UHI - ULO) / 50.0
            X, FX = self._descend(grp, ctx, X, FX, span0 / 2, 9, 4.0)
        else:
            ug = ULO + (UHI - ULO) * np.arange(21) / 20.0
            UU, VV = np.meshgrid(ug, ug, indexing="ij")
            F, _ = self.objective(grp, np.tile(UU.ravel(), (G, 1)), np.tile(VV.ravel(), (G, 1)), ctx)
            F3 = F.reshape(G, 21, 21)
            Fp = np.pad(F3, ((0, 0), (1, 1), (1, 1)), constant_values=np.inf)
            lm = np.ones_like(F3, bool)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        lm &= ~(Fp[:, 1 + di:22 + di, 1 + dj:22 + dj] < F3)
            Fm = np.where(lm, F3, np.inf).reshape(G, -1)
            o = np.argsort(Fm, axis=1, kind="stable")[:, :3]
            ok = np.isfinite(np.take_along_axis(Fm, o, axis=1))
            o = np.where(ok, o, o[:, :1])
            X = np.stack([UU.ravel()[o], VV.ravel()[o]], axis=2); FX = np.take_along_axis(F, o, axis=1)
            span0 = (UHI - ULO) / 20.0
            X, FX = self._descend(grp, ctx, X, FX, span0 / 2, 11, 3.0)
        kb = np.argmin(FX, axis=1)
        Xb = X[np.arange(G), kb][:, None, :]; Fb = FX[np.arange(G), kb][:, None]
        # convergence rounds: restart the descent from the optimum until a round gains less than 1e-7
        for _ in range(4):
            X2, F2 = self._descend(grp, ctx, Xb, Fb, (span0 / 16) if not grp.two else (span0 / 16), 7 if not grp.two else 8,
                                   4.0 if not grp.two else 3.0)
            gain = Fb[:, 0] - F2[:, 0]
            Xb, Fb = X2, F2
            if np.all(gain < 1e-7):
                break
        u = Xb[:, 0, 0]
        v = Xb[:, 0, 1] if grp.two else np.full(G, np.nan)
        F, q = self.objective(grp, u[:, None], v[:, None] if grp.two else None, ctx)
        return dict(F=F[:, 0], u=u, v=v, q=q[:, 0])

    def fit(self, ctx):
        """Converged per-galaxy fits at one global node. Returns arrays in names order: F (objective), u, v, q."""
        r = [self.fit_group(g, ctx) for g in self.groups]
        cat = {k: np.concatenate([r[0][k], r[1][k]])[self.idx] for k in ("F", "u", "v", "q")}
        return cat

    def points(self, ctx, res):
        """Per-point quantities at fitted nuisances res (names order): galaxy name, log10 g_pred, validity, the
        residual log10(g_obs/q^2) - log10 g_pred, chi^2 contribution, and the baryonic acceleration."""
        outp = {}
        for gi_grp, grp in enumerate(self.groups):
            if grp.G == 0:
                continue
            names = [g["name"] for g in grp.gl]
            ii = np.array([self.names.index(n) for n in names])
            U = res["u"][ii][:, None]; V = res["v"][ii][:, None]; q = res["q"][ii]
            lg, valid = self.logg(grp, U, V if grp.two else None, ctx)
            lg, valid = lg[:, 0], valid[:, 0]
            Vo = ctx["Vo"][gi_grp] if "Vo" in ctx else grp.Vo
            qp = q[grp.gid]
            Vp = np.sqrt(10.0 ** lg) * grp.sR
            r = np.log10(np.maximum(Vo, 1e-6) ** 2 / grp.R * CONV / qp ** 2) - lg
            chi = grp.w * (Vo - qp * Vp) ** 2
            Yd = 10.0 ** U[grp.gid, 0]; Yb = 10.0 ** V[grp.gid, 0] if grp.two else 0.0
            if ctx["input"] == "model":
                gbar = grp.mg + Yd * grp.md + (Yb * grp.mb if grp.two else 0.0)
            else:
                gbar = grp.gg + Yd * grp.gd + (Yb * grp.gb if grp.two else 0.0)
            full_ = ctx.get("full")
            gam = (self.gamma(grp, Yd[:, None], (Yb[:, None] if grp.two else 0.0), ctx)[:, 0] if ctx.get("full")
                   else np.zeros(len(gbar)))
            lae = (self.la_eff(grp, Yd[:, None], (Yb[:, None] if grp.two else 0.0), ctx)[:, 0] if full_
                   else np.full(len(gbar), np.nan))
            for nm_i, n in enumerate(names):
                s = slice(grp.starts[nm_i], grp.starts[nm_i] + grp.npt[nm_i])
                outp[n] = dict(lg=lg[s], valid=valid[s], r=np.where(valid[s], r[s], np.nan), chi=np.where(valid[s], chi[s], 0.0),
                               gbar=gbar[s], R=grp.R[s], Vo=Vo[s], w=grp.w[s], sR=grp.sR[s], Sigma=grp.Sigma[s],
                               y16=grp.y16[s], gN0=grp.gN0[s], gam=gam[s], lae=lae[s])
        return outp

    def mock(self, ctx):
        """A noiseless mock of the model in ctx at the prior centres (ratios 0.5 and 0.7, q = 1): returns the Vo and
        point-mask overrides for later contexts (points invalid at the prior centres are masked)."""
        Vo, pm = [], []
        for grp in self.groups:
            if grp.G == 0:
                Vo.append(np.zeros(0)); pm.append(np.zeros(0, bool)); continue
            U = np.full((grp.G, 1), LGY0); V = np.full((grp.G, 1), LGYB0)
            lg, valid = self.logg(grp, U, V if grp.two else None, ctx)
            Vo.append(np.where(valid[:, 0], np.sqrt(10.0 ** lg[:, 0]) * grp.sR, 0.0))
            pm.append(valid[:, 0].copy())
        return Vo, pm


# ------------------------------------------------------------------ the global fits (the ladder and scan of two_constant_fit.pl)
def parab_ls(xs, ys, win):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    lo = ys.min()
    u = np.where(ys <= lo + win)[0]
    if len(u) < 3:
        u = np.argsort(ys)[:3]
    if len(u) < 3:
        return None
    M = np.vstack([np.ones(len(u)), xs[u], xs[u] ** 2]).T
    c = np.linalg.lstsq(M, ys[u], rcond=None)[0]
    if not c[2] > 0:
        return None
    xm = -c[1] / (2 * c[2])
    rr = ys[u] - (c[0] + c[1] * xs[u] + c[2] * xs[u] ** 2)
    return xm, c[0] + c[1] * xm + c[2] * xm * xm, math.sqrt(1.0 / c[2]), math.sqrt(np.mean(rr ** 2))


def parab3(n3):
    (x1, y1), (x2, y2), (x3, y3) = n3
    den = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / den
    B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / den
    if not A > 0:
        return None
    xm = -B / (2 * A)
    return xm, math.sqrt(1.0 / A), y1 - A * (x1 - xm) ** 2


class Ladder:
    """fit_a0 of two_constant_fit.pl: converged totals on a ladder in log10 a0, minimum by a least-squares parabola; every
    per-galaxy objective is stored for the bootstrap."""

    def __init__(self, ds, ctx_base, la_c, log=None):
        self.ds, self.base, self.v, self.per, self.fits = ds, ctx_base, {}, {}, {}
        self.log = log
        self.run(la_c)

    def ev(self, la):
        k = round(la, 6)
        if k in self.v:
            return self.v[k]
        ctx = dict(self.base); ctx["a0"] = 10.0 ** la
        ctx["_cache"] = self.base.setdefault("_cache", {})
        r = self.ds.fit(ctx)
        self.v[k] = float(r["F"].sum()); self.per[k] = r["F"].copy(); self.fits[k] = r
        return self.v[k]

    def run(self, la_c):
        for d in (-0.02, 0.0, 0.02):
            self.ev(la_c + d)
        for _ in range(8):
            ks = sorted(self.v); r = parab_ls(ks, [self.v[k] for k in ks], 15)
            if r is None:
                nx = ks[0] - 0.03 if self.v[ks[0]] < self.v[ks[-1]] else ks[-1] + 0.03
            else:
                nx = r[0]
            nx = min(max(nx, ks[0] - 0.04), ks[-1] + 0.04)
            near = min(ks, key=lambda k: abs(k - nx))
            if abs(near - nx) < 0.003:
                break
            self.ev(nx)
        for ds_, tolr in (((-0.01, 0.01), 0.004), ((-0.10, -0.05, 0.05, 0.10), 0.012)):
            ks = sorted(self.v); r = parab_ls(ks, [self.v[k] for k in ks], 10)
            if r is not None:
                for d in ds_:
                    t = r[0] + d
                    if min(abs(k - t) for k in ks) > tolr:
                        self.ev(t)
        ks = sorted(self.v); r = parab_ls(ks, [self.v[k] for k in ks], 10)
        if r is None:
            xm = min(ks, key=lambda k: self.v[k]); sx = 0.0
        else:
            xm, _, sx, _ = r
        self.ev(xm)
        self.la, self.sig_la = round(xm, 6), sx
        self.chi2 = self.v[self.la]
        self.best = self.fits[self.la]

    def node_min(self, w=None):
        """Minimum for galaxy weights w (names order): parabola through the stored totals (two_constant_fit.pl)."""
        ks = sorted(self.v)
        y = [float(np.dot(w, self.per[k])) if w is not None else self.v[k] for k in ks]
        r = parab_ls(ks, y, 10)
        if r is None or r[0] < ks[0] or r[0] > ks[-1]:
            i = int(np.argmin(y)); return y[i], ks[i]
        return r[1], r[0]


class Scan:
    """The boost law (T3): a ladder at each delta node, the minimum by the parabola through the three lowest nodes,
    and the a0 refitted at the minimum. The scan extends below its first node down to delta = 0.03, as in
    two_constant_fit.pl."""

    def __init__(self, ds, ctx_base, grid, la_start, log=None, extend=True, dmax=1.2):
        self.ds, self.base, self.nodes = ds, ctx_base, {}
        la = la_start
        for d in grid:
            la = self.node(d, la).la
        if extend:
            for _ in range(12):
                s = sorted(self.nodes); mn = min(s, key=lambda d: self.nodes[d].chi2)
                if mn == s[-1] and s[-1] < dmax - 1e-9:
                    nd = round(min(s[-1] + (0.05 if s[-1] < 0.6 else 0.1), dmax), 4)
                    la = self.node(nd, self.nodes[s[-1]].la).la
                elif mn == s[0] and s[0] > 0.03:
                    nd = round(s[0] * 0.6, 4)
                    la = self.node(nd, self.nodes[s[0]].la).la
                else:
                    break
            s = sorted(self.nodes)
            if log:
                log(f"      scan nodes (delta: chi^2 at its refitted a0): " + ", ".join(f"{d:g}: {self.nodes[d].chi2:.1f}" for d in s))
        s = sorted(self.nodes)
        n3 = sorted(sorted(s, key=lambda d: self.nodes[d].chi2)[:3])
        r = parab3([(d, self.nodes[d].chi2) for d in n3])
        mn = min(s, key=lambda d: self.nodes[d].chi2)
        self.edge = mn in (s[0], s[-1])
        if r is None or self.edge:
            self.dmin, self.sig = mn, float("nan")
        else:
            self.dmin, self.sig = round(r[0], 4), r[1]
        near = min(s, key=lambda d: abs(d - self.dmin))
        self.at = Ladder(ds, self.ctx(self.dmin), self.nodes[near].la)
        self.chi2, self.la, self.best = self.at.chi2, self.at.la, self.at.best

    def ctx(self, d):
        c = dict(self.base); c["delta"] = d; c["nu"] = law_nu("T3", d); c["_cache"] = {}
        return c

    def node(self, d, la):
        d = round(d, 4)
        if d not in self.nodes:
            self.nodes[d] = Ladder(self.ds, self.ctx(d), la)
        return self.nodes[d]

    def scan_min(self, w=None):
        """scan_min of two_constant_fit.pl for galaxy weights w."""
        n = sorted((d, *self.nodes[d].node_min(w)) for d in self.nodes)
        if len(n) < 3:
            return None
        ib = int(np.argmin([x[1] for x in n]))
        if w is not None and (ib == 0 or ib == len(n) - 1):
            return n[ib][1], n[ib][0]
        n3 = sorted(sorted(n, key=lambda x: x[1])[:3])
        r = parab3([(x[0], x[1]) for x in n3])
        if r is None or r[0] < n3[0][0] or r[0] > n3[2][0]:
            if 0 < ib < len(n) - 1:
                r = parab3([(x[0], x[1]) for x in n[ib - 1:ib + 2]])
                if r is None or r[0] < n[ib - 1][0] or r[0] > n[ib + 1][0]:
                    return n[ib][1], n[ib][0]
            else:
                return n[ib][1], n[ib][0]
        return r[2], r[0]


def boot_weights(n, nboot, seed):
    rng = np.random.default_rng(seed)
    W = np.zeros((nboot, n))
    for b in range(nboot):
        W[b] = np.bincount(rng.integers(0, n, n), minlength=n)
    return W
