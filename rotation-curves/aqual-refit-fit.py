"""Does the full AQUAL solution in the disks fit the SPARC rotation curves as well as the algebraic relation, with the
scale a0, and for the boost law its free volume, refitted for each? The fits behind Paper I, Sect. 4.2 and Paper II,
Sect. 6.1, for the empirical function and for the two-constant fit of the boost law. The constrained fit of the boost
law (Paper I, Eq. (11)) and MOND's simple and standard functions are fitted in constrained_fit.py, which reuses this
script's loader and fits.

RESULTS SUPPORTED
  Paper II, Sect. 6.1: the full solution degrades the fits of the empirical function and the two-constant fit by 1.6 to
  2.0 SD (aqual-refit-main-T11-results.txt and aqual-refit-main-T3-results.txt, SUMMARY); the penalty disappears with
  catalog distances and inclinations (aqual-refit-main-T11-frozen-results.txt) and weakens under mass-to-light priors
  of 0.25 dex (aqual-refit-main-T11-prior0.25-results.txt); the residual-slope tests (2.4 to 2.7 SD); the rms of
  0.06 dex of the departure Gamma and its median of -0.085 dex, 18 percent, where the vertical field exceeds the radial
  one at y < 0.1 (aqual-refit-main-T11-results.txt); in the two-constant fit on the full solution the free volume
  falls to the scan's lower stop, 0.03, with chi^2 still falling at the probes down to 0.004, and with the algebraic
  relation on the model's own field it falls to 0.03 as well (aqual-refit-main-T3-results.txt). Paper I, Sect. 4.2:
  the penalty of the full solution narrows under wider mass-to-light priors (aqual-refit-main-T11-prior0.25-results.txt).
  Part machinery checks the numpy port of the fit machinery, refit_core.py, against two_constant_fit.pl: the second
  implementation of the fit.

LAW CODES
  T11: the empirical function of McGaugh, Lelli and Schombert, nu(y) = 1/(1 - exp(-sqrt(y))), with a0 fitted.
  T3: the boost law of Paper I, Eq. (8), with a0 and the free volume delta fitted (the two-constant fit).

WHAT PART MAIN COMPUTES
  The comparison for one law on two baryonic inputs: the Newtonian field of the disk models of finite thickness for
  both models ('model', where the mismatch between the model and SPARC's tabulated curves cancels), and SPARC's
  tabulated g_bar with Gamma as a predicted change on the relation ('sparc'). For each: the best fits of the algebraic
  relation and of the full solution with a0 (and delta for T3) and every galaxy's nuisances refitted; Delta chi^2 =
  full minus relation with its galaxy-clustered SD; its expectations on noiseless mocks of each model (each fitted with
  everything refitted); the residual-slope tests; where the difference concentrates.
  How Gamma follows the mass-to-light ratios: Gamma is tabulated at the prior-centre ratios. It is read at a0/s (s the
  local ratio of the model's Newtonian field to its prior-centre value, exact for a uniform change of all masses), and,
  point by point, the remaining error against direct solutions at other ratios is added (mode 'corrected' of
  refit_core.py, from aqual-refit-solve.py parts upsvar and upsvar2): over the disk ratio from -1.0 to +1.0 dex and,
  for bulge galaxies, over the grid of disk ratios -0.4 to +0.8 dex by bulge ratios -0.8 to +0.8 dex (steps of 0.2),
  for T11 and for the boost law at delta 0.03 and 0.3 (interpolated linearly in delta).
  --frozen holds the catalog distances and inclinations (q = 1); --prior 0.25 widens the disk and bulge mass-to-light
  priors to 0.25 dex. Both are treatments of Paper I, Sect. 4.1.

INPUTS
  data/sparc/SPARC_Lelli2016c.mrt, data/sparc/rotmod_ltg/; rotation-curves/aqual-disk-points.csv;
  rotation-curves/aqual-refit-gamma-0.csv and -1.csv, aqual-refit-upsvar-0.csv and -1.csv, aqual-refit-upsvar2-0.csv
  and -1.csv (from aqual-refit-solve.py). A missing table stops the run.

OUTPUTS AND COMMANDS, from the repository root, after aqual-refit-solve.py
  python rotation-curves/aqual-refit-fit.py --part machinery --out rotation-curves/aqual-refit-machinery-results.txt
      (under a minute)
  python rotation-curves/aqual-refit-fit.py --part main --law T11 --out rotation-curves/aqual-refit-main-T11-results.txt
  python rotation-curves/aqual-refit-fit.py --part main --law T11 --frozen --out rotation-curves/aqual-refit-main-T11-frozen-results.txt
  python rotation-curves/aqual-refit-fit.py --part main --law T11 --prior 0.25 --out rotation-curves/aqual-refit-main-T11-prior0.25-results.txt
      (3 to 8 min each)
  python rotation-curves/aqual-refit-fit.py --part main --law T3 --out rotation-curves/aqual-refit-main-T3-results.txt
      (about 50 min; 2,892 s)
  --nboot N sets the bootstrap draws (2,000). --nomock skips the mocks, the slope tests and the point tables (used by
  coarse_grained_fit.py for the boost law). Needs numpy.
"""
import csv, math, os, re, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import refit_core as RC

T00 = time.time()
ARGS = sys.argv


def arg(name, default=None):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default


_lines = []


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def stamp(lab):
    out(f"   [{lab}: {time.time() - T00:.0f} s]")


GAMMA_FILES = ["rotation-curves/aqual-refit-gamma-0.csv", "rotation-curves/aqual-refit-gamma-1.csv"]
NBOOT = int(arg("--nboot", "2000"))
CONVK = RC.CONV
A0_T16 = 1.2e-10
G_KPC = 4.30091e-6                       # kpc (km/s)^2 / Msun


def load(strict=None):
    """--partial (testing only) accepts an incomplete Gamma table; the missing points are then excluded."""
    strict = ("--partial" not in ARGS) if strict is None else strict
    T1 = RC.read_table1("data/sparc/SPARC_Lelli2016c.mrt")
    gals = RC.read_rotmod("data/sparc/rotmod_ltg", T1)
    missing = [f for f in GAMMA_FILES if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError("Gamma tables missing (aqual-refit-solve.py --part gamma): " + ", ".join(missing))
    GAM = RC.read_gamma(GAMMA_FILES)
    PTS = RC.read_points("rotation-curves/aqual-disk-points.csv")
    if "--partial" in ARGS and "--solved-only" in ARGS:      # testing only: the galaxies the solver has finished
        done = sorted({k[0] for k in GAM})
        if arg("--test-n"):
            done = done[:int(arg("--test-n"))]
        gals = {n: g for n, g in gals.items() if n in set(done)}
    RC.attach(gals, GAM, PTS, strict=strict)
    return T1, gals


UPSVAR = ([f"rotation-curves/aqual-refit-upsvar-{k}.csv" for k in (0, 1)]       # solve part upsvar
          + [f"rotation-curves/aqual-refit-upsvar2-{k}.csv" for k in (0, 1)])    # solve part upsvar2
NOMOCK = "--nomock" in ARGS
PREPARE = None          # coarse_grained_fit.py sets a hook that swaps in its Gamma tables after the correction




def upsvar_rows(paths=None):
    """The direct solutions at other mass-to-light ratios (solve parts upsvar and upsvar2, or the files in paths: one
    row per point, one column per case named <law>_a<a0>_d<dlog Yd>_b<dlog Yb>, <law> = T11 or T3d<delta>), as one
    row per point and case. A missing file stops the run."""
    rows = []
    for p in (UPSVAR if paths is None else paths):
        if not os.path.exists(p):
            raise FileNotFoundError(f"direct solutions missing: {p}")
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            for k, v in r.items():
                m = re.fullmatch(r"(T11|T3d[0-9.]+)_a([0-9.]+e[+-][0-9]+)_d([+-][0-9.]+)_b([+-][0-9.]+)", k)
                if m and v != "":
                    rows.append(dict(galaxy=r["galaxy"], R=r["R"], law=m.group(1), a0=m.group(2), dlogYd=m.group(3),
                                     dlogYb=m.group(4), Gamma=v))
    return rows


def ctx_of(inp, law, delta=None, full=False, gmode="scaled", use_gok=True):
    c = dict(input=inp, law=law, full=full, gmode=gmode, use_gok=use_gok)
    if law == "T3":
        if delta is not None:              # a scan sets delta and nu node by node
            c["delta"] = delta
            c["nu"] = RC.law_nu("T3", delta)
    else:
        c["nu"] = RC.law_nu("T11")
    return c


DGRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


class LawFit:
    """The best fit of one model (a law, an input, algebraic or full) over its global constants, with every galaxy's
    nuisances refitted: the a0 ladder of two_constant_fit.pl (T11) or its delta scan with a ladder per node (T3)."""

    def __init__(self, ds, base, la_start, extra=None, dgrid=None, log=None):
        self.ds, self.law = ds, base["law"]
        base = dict(base)
        if extra:
            base.update(extra)
        self.base = base
        if self.law == "T11":
            self.L = RC.Ladder(ds, base, la_start)
            self.chi2, self.la, self.best, self.delta = self.L.chi2, self.L.la, self.L.best, None
            self.sig_la = self.L.sig_la
        else:
            self.S = RC.Scan(ds, base, dgrid or DGRID, la_start, log=log)
            self.chi2, self.la, self.best, self.delta = self.S.chi2, self.S.la, self.S.best, self.S.dmin
            self.sig_la, self.edge = self.S.at.sig_la, self.S.edge

    def ctx(self):
        c = dict(self.base)
        c["a0"] = 10.0 ** self.la
        if self.law == "T3":
            c["delta"] = self.delta
            c["nu"] = RC.law_nu("T3", self.delta)
        c["_cache"] = {}
        return c

    def minw(self, w):
        return self.L.node_min(w) if self.law == "T11" else self.S.scan_min(w)

    def desc(self):
        s = f"a0 = {10 ** self.la * 1e10:.4f}e-10"
        if self.law == "T3":
            s += f", delta = {self.delta:.4f}" + (" (minimum on the edge of the delta grid)" if self.edge else "")
        return s + f", chi^2 = {self.chi2:.2f}"


def boot_sd(vals):
    v = np.asarray(vals)
    return np.std(v, ddof=1), 0.5 * (np.percentile(v, 84) - np.percentile(v, 16))


def ols_slope(x, yv, cov, gid, nboot, seed):
    """OLS slope of yv on x with covariates cov, and its galaxy-bootstrap SE (resampling galaxies)."""
    M = np.vstack([np.ones(len(x)), x] + cov).T
    b0 = np.linalg.lstsq(M, yv, rcond=None)[0][1]
    ug = np.unique(gid)
    pos = np.searchsorted(ug, gid)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(nboot):
        cnt = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)).astype(float)
        w = cnt[pos]
        if np.ptp(x[w > 0]) <= 0:
            continue
        Mw = M * w[:, None]
        A = M.T @ Mw; bvec = Mw.T @ yv
        bs.append(np.linalg.solve(A, bvec)[1])
    return b0, np.std(bs, ddof=1)


# ====================================================================== part machinery
def part_machinery():
    T1, gals = load()
    out("AQUAL refit: the numpy port of the fit machinery (refit_core.py) against two_constant_fit.pl")
    out("=" * 110)
    ds = RC.Dataset(gals)
    out(f"Sample: {len(ds.names)} galaxies, {sum(len(g['R']) for g in gals.values())} points, {int(ds.hasb.sum())} with a bulge "
        "(SPARC, quality flag not 3, inclination at least 30 degrees, at least 3 points).")
    c = ctx_of("sparc", "T3", 0.2953, use_gok=False); c["a0"] = 0.92459e-10
    out(f"   T3 (the boost law) at delta = 0.2953, a0 = 0.92459e-10: total objective {ds.fit(c)['F'].sum():.4f}; two_constant_fit.pl 4319.22 "
        "(two_constant_fit-results.txt, section B)")
    dsf = RC.Dataset(gals, frozen=True)
    c = ctx_of("sparc", "T11", use_gok=False); c["a0"] = 0.8127e-10
    out(f"   T11 with catalog distances and inclinations (frozen), a0 = 0.8127e-10: {dsf.fit(c)['F'].sum():.4f}; two_constant_fit.pl 6701.430 "
        "(two_constant_fit-frozen-results.txt)")
    L = LawFit(ds, ctx_of("sparc", "T11", use_gok=False), math.log10(1.178e-10 * 0.90))
    out(f"   T11 (the empirical function) a0 refitted on the ladder: a0 = {10 ** L.la * 1e10:.4f}e-10 +/- "
        f"{10 ** L.la * math.log(10) * L.sig_la * 1e10:.4f} (stat), chi^2 {L.chi2:.4f}; two_constant_fit.pl 1.0475 +/- 0.0296, 4326.672")
    r = L.best
    out("   Per-galaxy objective at the T11 best fit: this port at its own best fit, against the converged column of")
    out("   two_constant_fit-results.txt, section L:")
    for g, recv in (("UGC06787", 75.36872), ("NGC5371", 68.89333), ("UGC02953", 78.40555), ("IC4202", 245.82530),
                    ("UGC09133", 42.88147), ("NGC7331", 11.46839)):
        i = ds.names.index(g)
        out(f"      {g:<9s} {r['F'][i]:10.5f} (Yd {10 ** r['u'][i]:.4f}, Yb {10 ** r['v'][i] if np.isfinite(r['v'][i]) else float('nan'):.4f}, "
            f"q {r['q'][i]:.4f}); two_constant_fit.pl {recv:.5f}")
    stamp("machinery")


# ====================================================================== the comparison and its tests
def report_comparison(ds, alg, full, c_alg, c_full, lab, seed, pointtables=True, T1=None, gals=None):
    n = len(ds.names)
    W = RC.boot_weights(n, NBOOT, seed)
    dd, la_a, la_f, de_a, de_f = [], [], [], [], []
    for b in range(NBOOT):
        ya, xa = alg.minw(W[b]); yf, xf = full.minw(W[b])
        dd.append(yf - ya)
        if alg.law == "T11":
            la_a.append(xa); la_f.append(xf)
        else:
            de_a.append(xa); de_f.append(xf)
    obs = full.chi2 - alg.chi2
    sd, hw = boot_sd(dd)
    dg = full.best["F"] - alg.best["F"]
    sd_dm = math.sqrt(np.var(dg, ddof=1) * n)
    out(f"   Delta chi^2 = full minus relation at the respective best fits: {obs:+.1f} +/- {sd:.1f} (galaxy-clustered SD, {NBOOT} draws; "
        f"16-84 half-width {hw:.1f}; delta-method {sd_dm:.1f}); {obs / sd:+.2f} SD.")
    if alg.law == "T11":
        out(f"   Clustered SD of log10 a0 over the draws: relation {np.std(la_a, ddof=1):.4f} dex, full {np.std(la_f, ddof=1):.4f} dex; "
            f"a0(full)/a0(relation) = {10 ** (full.la - alg.la):.4f}, its draws 16-84 percent "
            f"{10 ** np.percentile(np.array(la_f) - np.array(la_a), 16):.4f} to {10 ** np.percentile(np.array(la_f) - np.array(la_a), 84):.4f}")
    else:
        out(f"   Clustered SD of delta over the draws: relation {np.std(de_a, ddof=1):.4f}, full {np.std(de_f, ddof=1):.4f}; "
            f"draws with the minimum on the grid edge are taken at the edge node (as in two_constant_fit.pl)")
    if NOMOCK:                             # coarse_grained_fit.py, boost law: scans and Delta chi^2 only
        out("   (--nomock: mock expectations, slope tests and point tables skipped)")
        return dict(obs=obs, sd=sd, E_F=float("nan"), E_A=float("nan"))

    # expectations on noiseless mocks
    t0 = time.time()
    exp = {}
    fits_on = {}
    for mk, src in (("F", full), ("A", alg)):
        Vo, pmk = ds.mock(src.ctx())
        extra = dict(Vo=Vo, pmask=pmk)
        fa = LawFit(ds, c_alg, src.la, extra=extra, dgrid=mock_grid(src))
        ff = LawFit(ds, c_full, src.la, extra=extra, dgrid=mock_grid(src))
        exp[mk] = ff.chi2 - fa.chi2
        fits_on[mk] = (fa, ff, extra)
        out(f"   mock of the {'full solution' if mk == 'F' else 'relation'} at its best fit (prior-centre ratios, q = 1, {int(sum(p.sum() for p in pmk))} points): "
            f"relation refitted {fa.desc()}; full refitted {ff.desc()}; Delta chi^2 {exp[mk]:+.1f}")
    out(f"   Expected Delta chi^2: {exp['F']:+.1f} if the full solution is right, {exp['A']:+.1f} if the relation is. Observed {obs:+.1f} +/- {sd:.1f}:")
    out(f"      rejection of the full solution (obs - E_full)/SD = {(obs - exp['F']) / sd:+.2f}; against the relation's expectation "
        f"(obs - E_rel)/SD = {(obs - exp['A']) / sd:+.2f}  [{time.time() - t0:.0f} s]")

    # residual-slope tests
    Pa = ds.points(alg.ctx(), alg.best)
    fa_F, _, extraF = fits_on["F"]
    PaF = ds.points(dict(fa_F.ctx(), **extraF), fa_F.best)
    Pf = ds.points(full.ctx(), full.best)
    _, ff_A, extraA = fits_on["A"]
    PfA = ds.points(dict(ff_A.ctx(), **extraA), ff_A.best)
    for lab2, P0, Pm, sense in (("refit residuals of the relation on the unabsorbed Gamma (1 if the full solution is right, 0 if the relation is)", Pa, PaF, 1),
                                ("refit residuals of the full solution on the unabsorbed -Gamma (1 if the relation is right, 0 if the full solution is)", Pf, PfA, -1)):
        x, yv, ly, gid = [], [], [], []
        for gi, nm in enumerate(ds.names):
            a, m = P0[nm], Pm[nm]
            ok = a["valid"] & m["valid"] & np.isfinite(a["r"]) & np.isfinite(m["r"])
            x.append(m["r"][ok]); yv.append(a["r"][ok]); ly.append(np.log10(a["gbar"][ok] / (10 ** alg.la)))
            gid.append(np.full(ok.sum(), gi))
        x, yv, ly, gid = map(np.concatenate, (x, yv, ly, gid))
        b, sb = ols_slope(x, yv, [ly], gid, NBOOT, seed + 7)
        if sense == 1:
            out(f"   Slope test, {lab2}: slope {b:+.3f} +/- {sb:.3f}; rejection of the full solution {(1 - b) / sb:.1f} SD, of the relation "
                f"{b / sb:+.1f} SD; rms of the unabsorbed part {np.sqrt(np.mean(x ** 2)):.4f} dex ({len(x)} points)")
            ax = np.abs(x); cut = np.percentile(ax, 99)
            out(f"      the unabsorbed part is carried by a tail: median |.| {np.median(ax):.4f} dex, 1.4826 MAD "
                f"{1.4826 * np.median(np.abs(x - np.median(x))):.4f}, rms without the largest 1 percent {np.sqrt(np.mean(x[ax <= cut] ** 2)):.4f}, "
                f"points beyond 0.1 dex {int(np.sum(ax > 0.1))} of {len(x)}")
        else:
            out(f"   Reverse slope test, {lab2}: slope {b:+.3f} +/- {sb:.3f}; rejection of the relation {(1 - b) / sb:.1f} SD, of the full "
                f"solution {b / sb:+.1f} SD; rms {np.sqrt(np.mean(x ** 2)):.4f} dex")
    # the parts of the difference: data, mass-to-light priors, distance-inclination prior
    def parts(fit, P):
        dat = np.array([P[nm]["chi"].sum() for nm in ds.names])
        u, v = fit.best["u"], fit.best["v"]
        pri = ((u - RC.LGY0) / RC.PRIOR) ** 2 + np.where(np.isfinite(v), ((np.nan_to_num(v) - RC.LGYB0) / RC.PRIOR) ** 2, 0.0)
        return dat, pri, fit.best["F"] - dat - pri
    da, pa, qa = parts(alg, Pa); df_, pf_, qf_ = parts(full, Pf)
    duf = full.best["u"] - RC.LGY0
    hi_ = duf > 0.3
    out(f"   Parts of Delta chi^2 (full minus relation): data {np.sum(df_ - da):+.1f}, mass-to-light priors {np.sum(pf_ - pa):+.1f}, "
        f"distance-inclination prior {np.sum(qf_ - qa):+.1f}. In the {hi_.sum()} galaxies whose full-solution disk ratio lies more than "
        f"0.3 dex above 0.5: total {np.sum(dg[hi_]):+.1f} (data {np.sum((df_ - da)[hi_]):+.1f}, mass-to-light {np.sum((pf_ - pa)[hi_]):+.1f}, "
        f"distance-inclination {np.sum((qf_ - qa)[hi_]):+.1f}); the other {np.sum(~hi_)}: {np.sum(dg[~hi_]):+.1f}. Median fitted disk ratio "
        f"{10 ** np.median(alg.best['u']):.3f} (relation), {10 ** np.median(full.best['u']):.3f} (full); in those {hi_.sum()} galaxies "
        f"{10 ** np.median(alg.best['u'][hi_]):.2f} and {10 ** np.median(full.best['u'][hi_]):.2f}.")
    # scatter of the residuals about each model, and the size of Gamma at the full solution's fit
    ra_ = np.concatenate([Pa[nm]["r"][Pa[nm]["valid"] & Pf[nm]["valid"]] for nm in ds.names])
    rf_ = np.concatenate([Pf[nm]["r"][Pa[nm]["valid"] & Pf[nm]["valid"]] for nm in ds.names])
    gf_ = np.concatenate([Pf[nm]["gam"][Pa[nm]["valid"] & Pf[nm]["valid"]] for nm in ds.names])
    out(f"   Scatter of the residuals log10(g_obs/q^2) - log10 g_model about their mean ({len(ra_)} points): relation {np.std(ra_):.4f} dex, "
        f"full solution {np.std(rf_):.4f} dex. Gamma at the full solution's fit: rms {np.sqrt(np.mean(gf_ ** 2)):.4f} dex, mean {np.mean(gf_):+.4f}, "
        f"16-84 percent {np.percentile(gf_, 16):+.4f} to {np.percentile(gf_, 84):+.4f}.")
    le_ = np.concatenate([Pf[nm]["lae"][Pf[nm]["valid"]] for nm in ds.names])
    if np.any(np.isfinite(le_)):
        lo_, hi_ = (RC.LA_T11[0], RC.LA_T11[-1]) if alg.law == "T11" else (RC.LA_T3[0], RC.LA_T3[-1])
        out(f"   Gamma read beyond the solver's a0 nodes at the full solution's fit (linear extrapolation, at most {RC.EXTRAP_DEX} dex): "
            f"{np.sum(le_ < lo_)} points below, {np.sum(le_ > hi_)} above, of {np.sum(np.isfinite(le_))}; beyond the cap "
            f"{np.sum((le_ < lo_ - RC.EXTRAP_DEX) | (le_ > hi_ + RC.EXTRAP_DEX))}")
    if not pointtables:
        return dict(obs=obs, sd=sd, E_F=exp["F"], E_A=exp["A"])

    # where the difference concentrates
    out("   Where it concentrates (per-galaxy Delta chi^2 = full minus relation, objective with priors, at the respective best fits):")
    cls = {}
    for nm in ds.names:
        t = gals[nm]["T1"]
        fg = 1.33 * t["MHI"] / (1.33 * t["MHI"] + 0.5 * t["L36"]) if t["MHI"] + t["L36"] > 0 else np.nan
        sbc = "HSB" if t["SBeff"] > 200 else ("LSB" if t["SBeff"] < 100 else "mid")
        cls[nm] = dict(sb=sbc, fgas=fg, bulge=gals[nm]["hasb"])
    groups = [("bulge galaxies", lambda c: c["bulge"]), ("bulgeless", lambda c: not c["bulge"]),
              ("HSB (SBeff > 200)", lambda c: c["sb"] == "HSB"), ("intermediate", lambda c: c["sb"] == "mid"),
              ("LSB (SBeff < 100)", lambda c: c["sb"] == "LSB"), ("gas-rich (f_gas > 0.5)", lambda c: c["fgas"] > 0.5),
              ("gas-poor (f_gas <= 0.5)", lambda c: not (c["fgas"] > 0.5)),
              ("gas-rich LSB", lambda c: c["fgas"] > 0.5 and c["sb"] == "LSB")]
    for glab, fn in groups:
        s = np.array([fn(cls[nm]) for nm in ds.names])
        if s.sum():
            dsum = dg[s].sum()
            dsd = math.sqrt(np.var(dg[s], ddof=1) * s.sum()) if s.sum() > 1 else float("nan")
            out(f"      {glab:<26s} {s.sum():3d} galaxies: {dsum:+8.1f} +/- {dsd:6.1f} (delta method); worse by > 1: {np.sum(dg[s] > 1):3d}, better by > 1: {np.sum(dg[s] < -1):3d}")
    o = np.argsort(-dg)
    out("      largest increases: " + ", ".join(f"{ds.names[k]} {dg[k]:+.0f}" for k in o[:10]))
    out("      largest decreases: " + ", ".join(f"{ds.names[k]} {dg[k]:+.0f}" for k in o[::-1][:6]))
    # per point, by the ratio of the vertical to the radial field and by y
    ch, rv, yb, ra, rf, gm = [], [], [], [], [], []
    for nm in ds.names:
        a, f = Pa[nm], Pf[nm]
        ok = a["valid"] & f["valid"]
        ch.append((f["chi"] - a["chi"])[ok])
        gbar16 = a["y16"] * A0_T16                    # SPARC g_bar at 0.5/0.7 (m/s^2)
        rv.append((2 * math.pi * G_KPC * 1e6 * a["Sigma"] / np.maximum(gbar16 / CONVK, 1e-30))[ok])
        yb.append(a["y16"][ok]); ra.append(a["r"][ok]); rf.append(f["r"][ok])
        gm.append(f["gam"][ok])
    ch, rv, yb, ra, rf, gm = map(np.concatenate, (ch, rv, yb, ra, rf, gm))
    lab_ = lambda lo, hi: f"[{lo:g}, {'inf' if hi > 1e8 else format(hi, 'g')})"
    out("      per point (data chi^2 only, no priors). Residual = log10(g_obs/q^2) - log10 g_model at each model's own fit;")
    out("      Gamma at the full solution's fit (its a0 and ratios). By 2 pi G Sigma/g_R, the ratio of the vertical field")
    out("      just outside the layer to the radial one (Sigma = 0.5 SB_disk + 0.7 SB_bul + Sigma_gas, g_R = SPARC g_bar at 0.5/0.7):")
    out("        bin           points   sum dchi^2   mean residual: relation    full     Gamma: median   mean")
    for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e9)):
        s = (rv >= lo) & (rv < hi)
        if s.sum():
            out(f"        {lab_(lo, hi):<12s} {s.sum():6d}   {ch[s].sum():+9.1f}          {np.mean(ra[s]):+.4f}   {np.mean(rf[s]):+.4f}"
                f"          {np.median(gm[s]):+.4f}  {np.mean(gm[s]):+.4f}")
    out("      by y = g_bar/a0 (SPARC g_bar at 0.5/0.7 over 1.2e-10, the bins of aqual-disk-solver-results.txt):")
    for lo, hi in ((0, 0.1), (0.1, 1.0), (1.0, 4.0), (4.0, 1e9)):
        s = (yb >= lo) & (yb < hi)
        if s.sum():
            out(f"        {lab_(lo, hi):<12s} {s.sum():6d}   {ch[s].sum():+9.1f}          {np.mean(ra[s]):+.4f}   {np.mean(rf[s]):+.4f}"
                f"          {np.median(gm[s]):+.4f}  {np.mean(gm[s]):+.4f}")
    s = (yb < 0.1) & (rv > 1)
    out(f"      the {s.sum()} points at y < 0.1 with 2 pi G Sigma > g_R: sum dchi^2 {ch[s].sum():+.1f}; mean residual {np.mean(ra[s]):+.4f} "
        f"(relation), {np.mean(rf[s]):+.4f} (full); Gamma median {np.median(gm[s]):+.4f}")
    return dict(obs=obs, sd=sd, E_F=exp["F"], E_A=exp["A"])


def mock_grid(src):
    """Delta nodes for the mock fits: the data grid, extended as the scan requires."""
    return None


# ====================================================================== part main
def part_main():
    law = arg("--law", "T11")
    gmode = "corrected"
    T1, gals = load()
    out(f"AQUAL refit: the full AQUAL solution against the algebraic relation, {law} "
        + ("(the empirical function, a0 free)" if law == "T11" else "(the boost law, a0 and delta free)"))
    out("=" * 110)
    out("Fit machinery of two_constant_fit.pl as ported in refit_core.py: 153 SPARC galaxies, 3,168 points; disk")
    out("ratio and each bulge ratio free under 0.10 dex priors about 0.5 and 0.7; distance and inclination through q, profiled")
    out("within their catalog priors; converged per-galaxy fits; a0 by a ladder in log10 a0; errors clustered by galaxy.")
    out("The algebraic relation predicts g = nu(g_b/a0) g_b; the full AQUAL solution g = nu(g_b/a0) g_b 10^Gamma, Gamma from the")
    out("disk solver (aqual-disk-solver.py) for the model at the current a0 (and delta), carried to the fitted ratios by")
    out("Gamma(s rho, a0) = Gamma(rho, a0/s) plus, point by point, its remaining error against direct solutions at other"
        " disk and bulge ratios (mode 'corrected'). Both models see the same points.")
    frozen = "--frozen" in ARGS
    if arg("--prior"):
        RC.PRIOR = float(arg("--prior"))
        out(f"VARIANT: mass-to-light priors of {RC.PRIOR:g} dex for disks and bulges (a treatment of Paper I, Sect. 4.1).")
    if frozen:
        out("REDUCTION: catalog distances and inclinations held fixed (q = 1), a treatment of Paper I, Sect. 4.1.")
    ds = RC.Dataset(gals, frozen=frozen)
    if gmode == "corrected":
        ds.attach_eps(upsvar_rows())
    if PREPARE is not None:
        PREPARE(ds, gals)
    nbad = sum(int((~g.gok).sum()) for g in ds.groups if g.G)
    out(f"Points where the model's field or Gamma is undefined (excluded from both models): {nbad}")
    summary = []
    for inp, title in (("model", "INPUT 1: THE MODEL'S OWN NEWTONIAN FIELD WITH THICKNESS FOR BOTH MODELS (model mismatch cancels)"),
                       ("sparc", "INPUT 2: SPARC'S TABULATED g_bar, Gamma AS A PREDICTED CHANGE ON THE RELATION")):
        out(f"\n{title}")
        c_alg = ctx_of(inp, law); c_full = ctx_of(inp, law, full=True, gmode=gmode)
        la0 = math.log10((1.05e-10 if law == "T11" else 0.95e-10) * (0.8 if frozen else 1.0))
        t0 = time.time()
        alg = LawFit(ds, c_alg, la0, log=out)
        out(f"   relation: {alg.desc()}  [{time.time() - t0:.0f} s]")
        t0 = time.time()
        full = LawFit(ds, c_full, alg.la, log=out)
        out(f"   full solution: {full.desc()}  [{time.time() - t0:.0f} s]")
        if law == "T3":
            for lab_, f_ in (("relation", alg), ("full solution", full)):
                if f_.edge and f_.delta <= min(f_.S.nodes) + 1e-9:
                    row = []
                    la_ = f_.la
                    for d_ in (0.015, 0.008, 0.004):
                        Lp = RC.Ladder(ds, f_.S.ctx(d_), la_)
                        la_ = Lp.la
                        row.append(f"delta {d_:g}: chi^2 {Lp.chi2:.1f} (a0 {10 ** Lp.la * 1e10:.4f}e-10)")
                    out(f"   probe below the scan's lower stop for the {lab_} (Gamma's quartic in delta extrapolated below its "
                        f"lowest node): " + "; ".join(row))
        cf = alg.ctx(); cf.update(full=True, gmode=gmode, _cache={})
        rfx = ds.fit(cf)
        out(f"   full solution at the relation's best-fit constants (no refit): chi^2 {rfx['F'].sum():.2f}, Delta {rfx['F'].sum() - alg.chi2:+.1f}; "
            f"the refit recovers {rfx['F'].sum() - full.chi2:.1f} of it")
        res = report_comparison(ds, alg, full, c_alg, c_full, inp, seed=20261011 if inp == "model" else 20261012, T1=T1, gals=gals)
        summary.append((inp, alg, full, rfx["F"].sum(), res))
        stamp(inp)
    out("\nSUMMARY")
    for inp, alg, full, fx, res in summary:
        out(f"   {inp:<6s}: relation {alg.desc()}; full {full.desc()}; Delta chi^2 fixed calibration {fx - alg.chi2:+.1f}, refitted "
            f"{res['obs']:+.1f} +/- {res['sd']:.1f} ({res['obs'] / res['sd']:+.2f} SD); expected {res['E_F']:+.1f} (full right), {res['E_A']:+.1f} "
            f"(relation right); rejection of the full solution {(res['obs'] - res['E_F']) / res['sd']:.2f} SD")
    stamp("main")


if __name__ == "__main__":
    part = arg("--part", "main")
    {"machinery": part_machinery, "main": part_main}[part]()
    out(f"\n[run time {time.time() - T00:.0f} s]")
    if arg("--out"):
        with open(arg("--out"), "w", newline="\n", encoding="utf-8") as fh:
            fh.write("\n".join(_lines) + "\n")
