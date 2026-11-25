"""The radial condition of the globular-cluster test: the preference for Newton over ambient rescaling when the
stellar mass-to-light ratio is allowed the radial trend that stellar populations predict.

P tracks galactocentric radius, so a fall of the stellar ratio Y_* with radius could mask the rise that ambient
rescaling predicts. The script fits log10 Y_dyn = a + b log10 P [+ trend] + N(0, s^2 + e^2) to the 133 clusters
with rich kinematics, profiles over a, s and the trend coefficients, and gives Delta lnL = max lnL(b = 0) -
max lnL(b = 1), positive for the scale-selective coupling (Newton). The trends come from stellar-population
predictions of Y_* (popmodel.py) evaluated on the metallicities of the Harris catalog, regressed on log R_GC, and
imposed as fixed indices, Gaussian priors or flat windows on a power law Y_* ~ R_GC^c. A Gaussian prior on a trend
coefficient enters as a penalty -(c - c0)^2 / (2 sigma^2) inside the profile.

Supports Paper II section 6.3: the metallicities permit at most a factor of 1.4 to 1.75 decline of Y_* from 0.7 to
96 kpc (section 2 of the output, in/out of the variants Z and Zsteep+young), and capping the radial trend at that
limit keeps Delta lnL = 32.5 to 62.6, or 6.3 to 14.5 beyond 8 kpc (section 4, the envelope over the seven rows
across 58.254 to 200 pc). The variants marked "young" assume outer-halo clusters 1 Gyr younger.

Inputs: globular-clusters/gc_dynamical-rich-sample.tsv (written by globular-clusters/gc_dynamical.py, which runs
first) and data/harris-mwgc-2010-2026-11-15/mwgc.dat.txt and data/baumgardt-gc/combined_table.txt (fetched by the
reader, not redistributed), read through globular-clusters/popmodel.py.
Output: globular-clusters/gc_radial_trends-results.txt.
Run from the repository root, after gc_dynamical.py (about a minute):
    python globular-clusters/gc_radial_trends.py
"""
import hashlib
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "gc_dynamical-rich-sample.tsv")
OUT = os.path.join(HERE, "gc_radial_trends-results.txt")
LN10 = math.log(10.0)
SG_C = np.linspace(0.005, 0.5, 496)          # the scatter grid of gc_dynamical.py
BS_C = np.linspace(-2.0, 2.0, 4001)          # the slope grid of gc_dynamical.py
WIN = (58.254, 90.0, 100.0, 200.0)       # catchment lengths in pc: the floor at H0 = 67.4 km/s/Mpc, 90 pc, 100 pc, the ceiling
LC0 = WIN[0]                             # single values are quoted at the floor
SPAN = None
L = []

sys.path.insert(0, HERE)
import popmodel as pm  # noqa: E402
HARRIS_SHA = hashlib.sha256(open(pm.HARRIS, "rb").read()).hexdigest()


def out(s=""):
    L.append(s)
    print(s, flush=True)


# ------------------------------------------------------------------ sample
# the boost columns are read by their header names P<lc>
lines_ = open(SAMPLE, encoding="utf-8").read().splitlines()
hdr_ = lines_[0].split("\t")
pcols = {float(h[1:]): j for j, h in enumerate(hdr_) if h.startswith("P")}
for lc in WIN:
    if lc not in pcols:
        sys.exit("sample has no column P%g" % lc)
rows = []
for ln in lines_[1:]:
    t = ln.split("\t")
    rows.append(dict(name=t[0], rgc=float(t[1]), ml=float(t[2]), dml=float(t[3]), floor=float(t[4]),
                     mass=float(t[5]), trh=float(t[6]), mf=float(t[7]),
                     P={lc: float(t[j]) for lc, j in pcols.items()}))
extra = pm.catalog_extra()
feh_h, ltc_h = pm.harris_tables()
for r in rows:
    r.update(extra[r["name"]])
    hid = pm.harris_id(r["name"])
    r["feh"] = feh_h.get(hid)
    r["ltc"] = ltc_h.get(hid)


# ------------------------------------------------------------------ the fitting machinery of gc_dynamical.py, copied
def arrays(sample, lc):
    x = np.array([math.log10(r["P"][lc]) for r in sample])
    y = np.array([math.log10(r["ml"]) for r in sample])
    e = np.array([max(r["dml"] / r["ml"], r["floor"]) / LN10 for r in sample])
    return x, y, e


def wls_scan(y, X, e, j, sgrid):
    llm, bh, vb, cf = [], [], [], []
    for s in sgrid:
        w = 1.0 / (s * s + e * e)
        A = X * np.sqrt(w)[:, None]
        M = A.T @ A
        c = np.linalg.solve(M, A.T @ (y * np.sqrt(w)))
        res = y - X @ c
        llm.append(-0.5 * (w * res * res).sum() + 0.5 * np.log(w).sum())
        bh.append(c[j])
        vb.append(np.linalg.inv(M)[j, j])
        cf.append(c)
    return np.array(llm), np.array(bh), np.array(vb), cf


def slope_fits(sample, lc, covars=()):
    x, y, e = arrays(sample, lc)
    cols = [np.ones_like(x), x] + [np.array([cv(r) for r in sample]) for cv in covars]
    X = np.vstack(cols).T
    llm, bh, vb, cf = wls_scan(y, X, e, 1, SG_C)
    prof = (llm[:, None] - 0.5 * (BS_C[None, :] - bh[:, None]) ** 2 / vb[:, None]).max(axis=0)
    i = int(np.argmax(prof))
    ok = BS_C[prof >= prof[i] - 0.5]

    def at_b(bv):
        v = llm - 0.5 * (bv - bh) ** 2 / vb
        return v.max()
    return dict(d=at_b(0.0) - at_b(1.0), b=BS_C[i], blo=ok.min(), bhi=ok.max())


def bounded_fit(sample, lc, cmax, step=0.005):
    x, y, e = arrays(sample, lc)
    lr = np.array([math.log10(r["rgc"]) for r in sample])
    X = np.vstack([np.ones_like(x), x]).T
    best = {0.0: -np.inf, 1.0: -np.inf}
    for c in np.arange(-cmax, cmax + 1e-9, step):
        llm, bh, vb, _ = wls_scan(y - c * lr, X, e, 1, SG_C)
        for bv in (0.0, 1.0):
            best[bv] = max(best[bv], float(np.max(llm - 0.5 * (bv - bh) ** 2 / vb)))
    return best[0.0] - best[1.0]


# ------------------------------------------------------------------ the same likelihood, with priors on covariates
def prof_at_b(y, x, e, bfix, cols, priors=None, offset=None):
    """max over intercept, covariate coefficients and scatter of lnL at a fixed slope b.
    cols: list of covariate arrays; priors: {index in cols: (mean, sigma)} Gaussian penalties (sigma 0 = fixed)."""
    yy = y - bfix * x - (offset if offset is not None else 0.0)
    X = np.vstack([np.ones_like(y)] + list(cols)).T
    k = X.shape[1]
    prec = np.zeros(k)                      # prior precision 1/sigma^2 per coefficient (0 = no prior)
    mean = np.zeros(k)
    for j, (mu, sg) in (priors or {}).items():
        prec[j + 1] = 1.0 / sg ** 2
        mean[j + 1] = mu
    best = (-np.inf, None, None)
    for s in SG_C:
        w = 1.0 / (s * s + e * e)
        A = X * np.sqrt(w)[:, None]
        M = A.T @ A + np.diag(prec)
        c = np.linalg.solve(M, A.T @ (yy * np.sqrt(w)) + prec * mean)
        res = yy - X @ c
        pen = float((prec * (c - mean) ** 2).sum())
        ll = -0.5 * (w * res * res).sum() + 0.5 * np.log(w).sum() - 0.5 * pen
        if ll > best[0]:
            best = (ll, s, c)
    return best


def delta(sample, lc, covs=(), priors=None, offset=None):
    x, y, e = arrays(sample, lc)
    cols = [np.array([cv(r) for r in sample]) for cv in covs]
    l0 = prof_at_b(y, x, e, 0.0, cols, priors, offset)
    l1 = prof_at_b(y, x, e, 1.0, cols, priors, offset)
    return l0[0] - l1[0], l0, l1


def best_slope(sample, lc, covs=(), priors=None, offset=None):
    x, y, e = arrays(sample, lc)
    cols = [np.array([cv(r) for r in sample]) for cv in covs]
    bs = np.linspace(-1.0, 1.5, 251)
    ll = np.array([prof_at_b(y, x, e, b, cols, priors, offset)[0] for b in bs])
    i = int(np.argmax(ll))
    ok = bs[ll >= ll[i] - 0.5]
    return bs[i], ok.min(), ok.max()


lgR = lambda r: math.log10(r["rgc"])
lgR2 = lambda r: math.log10(r["rgc"]) ** 2
lgR3 = lambda r: math.log10(r["rgc"]) ** 3

out("THE RADIAL CONDITION OF THE GLOBULAR-CLUSTER PREFERENCE WITH A REALISTIC STELLAR M/L TREND")
out("=" * 108)
out("Delta lnL = max lnL(slope 0, Newton) - max lnL(slope 1, ambient rescaling); positive favors Newton.")
out("Sample %s; catchment lengths %s pc; single values at %g pc." % (
    os.path.basename(SAMPLE), ", ".join("%g" % lc for lc in WIN), LC0))
# the catalog read, with its SHA-256 and what was parsed from it
out("Harris catalog %s, SHA-256 %s: [Fe/H] for %d clusters (Part II), lg(tc) for %d (Part III)." % (
    os.path.basename(pm.HARRIS), HARRIS_SHA, len(feh_h), len(ltc_h)))
out()

lr_all = np.array([math.log10(r["rgc"]) for r in rows])
SPAN = lr_all.max() - lr_all.min()


def trend_c(sample, lc, bfix, covs=(lgR,)):
    x, y, e = arrays(sample, lc)
    cols = [np.array([cv(r) for r in sample]) for cv in covs]
    return prof_at_b(y, x, e, bfix, cols)[2][1]


cN, cR = trend_c(rows, LC0, 0.0), trend_c(rows, LC0, 1.0)     # the free power law's index under each account

# =============================================================================================
# 1. The observable
# =============================================================================================
out("1. THE OBSERVABLE AND THE RADIUS OF THE CONDITION")
out("   y = log10 of the catalog's GLOBAL dynamical M/L_V (Baumgardt N-body fit, segregation included), one value")
out("   per cluster; x = log10 P(l_c); the radial trend is a covariate in log10 R_GC, the GALACTOCENTRIC radius,")
out("   across clusters (0.7 to 96 kpc). It is not a radius inside a cluster.")
x90, y90, e90 = arrays(rows, LC0)
rho = np.corrcoef(lr_all, x90)[0, 1]
out("   Pearson correlation of log P(%g pc) with log R_GC over the 133: %.3f." % (LC0, rho))
out()

# =============================================================================================
# 2. Population priors on the stellar ratio
# =============================================================================================
out("2. STELLAR-POPULATION PREDICTIONS OF Y_* AND THEIR TREND WITH R_GC")
have = [r for r in rows if r["feh"] is not None]
miss = [r for r in rows if r["feh"] is None]
out("   Harris 2010 [Fe/H] found for %d of %d clusters; missing: %s." % (
    len(have), len(rows), ", ".join("%s (%.1f kpc)" % (r["name"], r["rgc"]) for r in miss)))
TAGE = 12.0


def kroupa_alpha_can(m1, m2):
    """Maximum-likelihood single power-law slope of a Kroupa IMF (-1.3 below 0.5, -2.3 above) over [m1, m2]."""
    m = np.geomspace(m1, m2, 4000)
    f = np.where(m < 0.5, (m / 0.5) ** -1.3, (m / 0.5) ** -2.3)
    w = f * m                                 # weights per log m
    w = w / w.sum()
    mean_ln = (w * np.log(m)).sum()

    def nll(a):
        if abs(a + 1) < 1e-9:
            Z = math.log(m2 / m1)
        else:
            Z = (m2 ** (a + 1) - m1 ** (a + 1)) / (a + 1)
        return -(a * mean_ln - math.log(Z))
    al = np.linspace(-3.5, 0.5, 4001)
    return float(al[np.argmin([nll(a) for a in al])])


out("   Kroupa-equivalent ML slope over 0.2-0.8 Msun: %.2f (Baumgardt and Hilker 2018: -1.5); over 0.3-0.8: %.2f; over 0.5-0.8: %.2f." % (
    kroupa_alpha_can(0.2, 0.8), kroupa_alpha_can(0.3, 0.8), kroupa_alpha_can(0.5, 0.8)))
for r in rows:
    m1 = r["mlow"] if r["mlow"] < 50 else 0.2
    m2 = r["mhigh"] if r["mhigh"] < 50 else 0.8
    acan = kroupa_alpha_can(m1, m2)
    r["dalpha"] = r["mf"] - acan                       # flattening relative to Kroupa over the cluster's own range
    r["Q_mf"] = pm.q_from_alpha_km(1.74 - r["dalpha"])
    r["Q_td"] = pm.q_from_tfrac(TAGE / (TAGE + r["tdiss"]))
    r["nrel"] = TAGE * 1e9 / 10 ** r["ltc"] if r["ltc"] is not None else None


def u_of(r, var):
    """log10 of the predicted stellar ratio (up to a constant) for a prior variant."""
    f = r["feh"]
    if var == "Z":
        return pm.ssp_kroupa(f)
    if var == "Zage":
        return pm.ssp_ageadj(f)
    if var == "Z+MF":
        return pm.ssp_kroupa(f) + math.log10(r["Q_mf"])
    if var == "Zage+MF":
        return pm.ssp_ageadj(f) + math.log10(r["Q_mf"])
    if var == "Z+TD":
        return pm.ssp_kroupa(f) + math.log10(r["Q_td"])
    if var == "Z+MF+B17":
        return pm.ssp_kroupa(f) + math.log10(r["Q_mf"]) + math.log10(pm.g_bianchini(r["nrel"]))
    if var == "Z+young":       # assumption, not a verified input: R_GC > 15 kpc clusters 1 Gyr younger
        return pm.ssp_kroupa(f) + (math.log10(1 - 0.45 / 3.5 / 1.83) if r["rgc"] > 15 else 0.0)
    if var == "Zsteep":        # stress test: the SSP slope above [Fe/H] = -0.76 doubled in log M/L
        base = pm.ssp_kroupa(f)
        return base if f <= -0.76 else 2 * base - pm.ssp_kroupa(-0.76)
    if var == "Zsteep+young":
        return u_of(r, "Zsteep") + (math.log10(1 - 0.45 / 3.5 / 1.83) if r["rgc"] > 15 else 0.0)
    if var in ("Z+alpha", "Z+alpha+MF"):
        # BSH20 Sec. 3.5: [alpha/Fe] = +0.2 below [Fe/H] = -0.8, falling linearly to 0 at [Fe/H] = 0. The
        # alpha-enhanced (Dartmouth +0.2) prediction is about 1.28 times the solar-scaled one at all [Fe/H]
        # (read off BSH20 Fig. 8: the same clusters sit near 1.35 against MIST and 1.05 against Dartmouth).
        fa = 1.0 if f >= 0 else (1.0 + 0.28 * min(1.0, -f / 0.8))
        v = pm.ssp_kroupa(f) + math.log10(fa)
        return v + (math.log10(r["Q_mf"]) if var.endswith("MF") else 0.0)
    if var == "MF":
        return math.log10(r["Q_mf"])
    if var == "TD":
        return math.log10(r["Q_td"])
    raise KeyError(var)


VARS = [("Z", "metallicity only (Kroupa SSP; KM09 + D20)"),
        ("Zage", "metallicity, age-metallicity relation (D20)"),
        ("Z+young", "metallicity, outer halo 1 Gyr younger (assumed)"),
        ("Zsteep", "metallicity, metal-rich slope doubled (stress)"),
        ("Zsteep+young", "the stress test plus the younger outer halo"),
        ("Z+alpha", "metallicity with BSH20's alpha-enhancement"),
        ("Z+alpha+MF", "the same x MF depletion"),
        ("MF", "MF depletion only (catalog MF via KM09)"),
        ("TD", "dissolution only (T_Diss via KM09 Eq. 5)"),
        ("Z+MF", "metallicity x MF depletion"),
        ("Zage+MF", "metallicity (age-adj.) x MF depletion"),
        ("Z+TD", "metallicity x dissolution (T_Diss)"),
        ("Z+MF+B17", "metallicity x MF x remnant retention (B17)")]


def sample_for(var):
    s = have
    if "B17" in var:
        s = [r for r in s if r["nrel"] is not None]
    return s


def ols_trend(sample, var, nboot=2000, seed=1):
    lr = np.array([math.log10(r["rgc"]) for r in sample])
    u = np.array([u_of(r, var) for r in sample])
    A = np.vstack([np.ones_like(lr), lr]).T
    c = np.linalg.lstsq(A, u, rcond=None)[0]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(nboot):
        i = rng.integers(0, len(lr), len(lr))
        bs.append(np.linalg.lstsq(A[i], u[i], rcond=None)[0][1])
    res = u - A @ c
    return c[1], float(np.std(bs)), float(np.std(res)), c[0]


out()
out("   Predicted log Y_* regressed on log R_GC (OLS over the clusters with [Fe/H]); c is the power-law index;")
out("   'in/out' is Y_*(0.7 kpc) / Y_*(96 kpc) = 10^(-c x %.3f). Rescaling needs c = %+.2f (in/out %.1f) at %g pc." % (
    SPAN, cR, 10 ** (-cR * SPAN), LC0))
TR = {}
NTR = {}
for var, lab in VARS:
    smp = sample_for(var)
    c, sc, rs, c0 = ols_trend(smp, var)
    TR[var] = (c, sc, rs)
    NTR[var] = len(smp)
    out("   %-9s %-46s N %3d  c %+.3f +- %.3f  in/out %.2f  rms about trend %.3f dex" % (
        var, lab, len(smp), c, sc, 10 ** (-c * SPAN), rs))
out()
out()

# =============================================================================================
# 3. The preference with the priors imposed
# =============================================================================================
out("3. THE PREFERENCE WITH THE POPULATION PRIORS (all / R_GC > 8 kpc; %s pc)" % ", ".join("%g" % lc for lc in WIN))


def row(label, fn, sample):
    s8 = [r for r in sample if r["rgc"] > 8]
    vals = [fn(sample, lc) for lc in WIN] + [fn(s8, lc) for lc in WIN]
    out("   %-58s %s" % (label, " ".join("%+6.1f" % v for v in vals)))
    return vals


RES = {}
out("   %-58s %s %s" % ("", " ".join("%6s" % ("%g" % lc) for lc in WIN), " ".join("%6s" % ("8:%g" % lc) for lc in WIN)))
# the row of per-cluster metallicity offsets, kappa = 1, for the variant Z runs on the clusters with Harris [Fe/H]
for var in ("Z",):
    smp = sample_for(var)
    uu = {id(r): u_of(r, var) for r in smp}
    U = lambda r, uu=uu: uu[id(r)]

    def fixed(s, lc, U=U):
        off = np.array([U(r) for r in s])
        return delta(s, lc, offset=off)[0]
    RES[var + ":k1"] = row("%s: offsets imposed (kappa = 1)" % var, fixed, smp)
N_K1 = (len(sample_for("Z")), len([r for r in sample_for("Z") if r["rgc"] > 8]))
out("   Power-law prior on the trend, Y_* ~ R_GC^c with c ~ N(c0, sigma), all 133 clusters (no [Fe/H] needed):")
for c0, sg, lab in ((0.0, 0.05, "c0 0, sigma 0.05"), (0.0, 0.10, "c0 0, sigma 0.10"),
                    (TR["Z"][0], 0.05, "c0 = metallicity-only trend, sigma 0.05"),
                    (TR["Z"][0], 0.10, "c0 = metallicity-only trend, sigma 0.10"),
                    (TR["Z+MF"][0], 0.05, "c0 = metallicity x MF trend, sigma 0.05"),
                    (TR["Z+TD"][0], 0.05, "c0 = metallicity x T_Diss trend, sigma 0.05"),
                    (cN, 0.05, "c0 = Newton's own fitted trend, sigma 0.05")):
    RES["pl:" + lab] = row("%s (c0 %+.3f)" % (lab, c0), lambda s, lc, c0=c0, sg=sg: delta(s, lc, (lgR,), {0: (c0, sg)})[0], rows)
SIDE = [TR[v][0] for v in ("Z", "Zage", "Z+young", "Zsteep", "Zsteep+young")]
c_side, s_side = float(np.mean(SIDE)), float(math.hypot(np.std(SIDE), max(TR[v][1] for v in ("Z", "Zsteep"))))
RES["pl:side"] = row("c0 = mean of metallicity-side variants, sigma %.3f (c0 %+.3f)" % (s_side, c_side),
                     lambda s, lc: delta(s, lc, (lgR,), {0: (c_side, s_side)})[0], rows)
out()
out("   Power law imposed at a fixed index (the trend alone, without per-cluster scatter):")
for var in ("Z", "Z+young", "Zsteep+young", "Z+alpha", "Z+MF", "Z+alpha+MF"):
    cf = TR[var][0]
    RES["fix:" + var] = row("c fixed at the %s trend (%+.3f, in/out %.2f)" % (var, cf, 10 ** (-cf * SPAN)),
                            lambda s, lc, cf=cf: delta(s, lc, offset=np.array([cf * math.log10(r["rgc"]) for r in s]))[0], rows)
cf = min(SIDE) - 2 * max(TR[v][1] for v in ("Z", "Zsteep"))
RES["fix:edge"] = row("c fixed at the rescaling-side edge (%+.3f, in/out %.2f)" % (cf, 10 ** (-cf * SPAN)),
                      lambda s, lc, cf=cf: delta(s, lc, offset=np.array([cf * math.log10(r["rgc"]) for r in s]))[0], rows)
out()
out("   Power law held to the population envelope (flat prior on c, profile as the bounded runs of gc_dynamical.py):")


def window_fit(sample, lc, clo, chi, step=0.005):
    x, y, e = arrays(sample, lc)
    lr = np.array([math.log10(r["rgc"]) for r in sample])
    best = {0.0: -np.inf, 1.0: -np.inf}
    for c in np.arange(clo, chi + 1e-9, step):
        for bv in (0.0, 1.0):
            best[bv] = max(best[bv], prof_at_b(y - c * lr, x, e, bv, [])[0])
    return best[0.0] - best[1.0]


cmin = min(TR[v][0] - 2 * TR[v][1] for v, _ in VARS)
cmax = max(TR[v][0] + 2 * TR[v][1] for v, _ in VARS)
RES["env"] = row("c in [%+.3f, %+.3f] (all variants, 2 s.e.)" % (cmin, cmax), lambda s, lc: window_fit(s, lc, cmin, cmax, 0.01), rows)
cz = TR["Z"][0] - 2 * TR["Z"][1]          # the metallicity-only edge, two standard errors
RES["envZ"] = row("c in [%+.3f, 0] (metallicity-only, 2 s.e., rescaling's side)" % cz, lambda s, lc: window_fit(s, lc, cz, 0.0, 0.01), rows)
out()

out()

# 4. the range that Paper II quotes, over the seven rows that bound the trend by metallicity
out("4. THE METALLICITY-BOUNDED PREFERENCE (the seven rows of section 3 that bound the trend by metallicity)")
BOUND = (("fix:Z+alpha", "c fixed at %+.3f (Z+alpha)" % TR["Z+alpha"][0]), ("fix:Z", "c fixed at %+.3f (Z)" % TR["Z"][0]),
         ("fix:Z+young", "c fixed at %+.3f (Z+young)" % TR["Z+young"][0]),
         ("fix:Zsteep+young", "c fixed at %+.3f (Zsteep+young)" % TR["Zsteep+young"][0]),
         ("envZ", "flat c in [%+.3f, 0]" % cz), ("pl:side", "c ~ N(%+.3f, %.4f)" % (c_side, s_side)),
         ("env", "flat c in [%+.3f, %+.3f]" % (cmin, cmax)))
# the eighth row, the per-cluster metallicity offsets, is listed with the seven, and the envelope is given over the
# seven rows (as Paper II quotes it) and over all eight; every row is also printed at three decimals
K1 = ("Z:k1", "per-cluster metallicity offsets (kappa 1)")
nw = len(WIN)
for key, lab in BOUND + (K1,):
    v = RES[key]
    out("   %-40s all %+6.1f to %+6.1f; beyond 8 kpc %+6.1f to %+6.1f" % (lab, min(v[:nw]), max(v[:nw]), min(v[nw:]), max(v[nw:])))
out("   The eighth row runs on the %d clusters of the sample with Harris [Fe/H], %d of them beyond 8 kpc." % N_K1)
for rowset, setname in ((BOUND, "the seven rows"), (BOUND + (K1,), "all eight rows")):
    out("   Envelope over %s:" % setname)
    for lc_set, name in ((WIN, "across %s pc" % ", ".join("%g" % lc for lc in WIN)),) + tuple(((lc,), "at %g pc" % lc) for lc in WIN):
        idx = [WIN.index(lc) for lc in lc_set]
        va = [RES[k][i] for k, _ in rowset for i in idx]
        v8 = [RES[k][nw + i] for k, _ in rowset for i in idx]
        ka = min(((RES[k][i], k, WIN[i]) for k, _ in rowset for i in idx))
        kA = max(((RES[k][i], k, WIN[i]) for k, _ in rowset for i in idx))
        k8 = min(((RES[k][nw + i], k, WIN[i]) for k, _ in rowset for i in idx))
        k8A = max(((RES[k][nw + i], k, WIN[i]) for k, _ in rowset for i in idx))
        out("   envelope %-30s all: %+6.1f to %+6.1f (%s at %g pc; %s at %g pc)" % (name, min(va), max(va), ka[1], ka[2], kA[1], kA[2]))
        out("   %-39s beyond 8 kpc: %+6.1f to %+6.1f (%s at %g pc; %s at %g pc)" % ("", min(v8), max(v8), k8[1], k8[2], k8A[1], k8A[2]))
out()
out("   Every row at three decimals:")
out("   %-18s %s | %s" % ("row", " ".join("%8s" % ("%g" % lc) for lc in WIN), " ".join("%8s" % ("8:%g" % lc) for lc in WIN)))
for key, lab in BOUND + (K1,):
    out("   %-18s %s | %s" % (key, " ".join("%+8.3f" % v for v in RES[key][:nw]), " ".join("%+8.3f" % v for v in RES[key][nw:])))
lo7, hi7 = min(RES[k][i] for k, _ in BOUND for i in range(nw)), max(RES[k][i] for k, _ in BOUND for i in range(nw))
lo7b, hi7b = min(RES[k][nw + i] for k, _ in BOUND for i in range(nw)), max(RES[k][nw + i] for k, _ in BOUND for i in range(nw))
vk = RES["Z:k1"]
out("   The eighth row against the envelope of the seven across %s pc (%.3f to %.3f; beyond 8 kpc %.3f to %.3f):" % (
    ", ".join("%g" % lc for lc in WIN), lo7, hi7, lo7b, hi7b))
for i, lc in enumerate(WIN):
    out("   %8g pc: all %+8.3f %s; beyond 8 kpc %+8.3f %s" % (
        lc, vk[i], "inside" if lo7 <= vk[i] <= hi7 else "OUTSIDE",
        vk[nw + i], "inside" if lo7b <= vk[nw + i] <= hi7b else "OUTSIDE"))

with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(L) + "\n")
print("wrote globular-clusters/gc_radial_trends-results.txt")
