"""The gas-layer baseline test: does the full AQUAL solution in the disks describe the rotation curves better than the
algebraic relation averaged over the gas layer?

RESULTS SUPPORTED
  Paper II, Sect. 6.1: against the vertically gas-averaged accelerations (R below), the slope test disfavors the full
  offset (F below) at about 3 SD in the inner disks (3.1 SD on the model's own field, 2.8 SD on SPARC's tables, line
  "Slope test" of part 2), and the observed Delta chi^2 sits about 5 SD from its value for noiseless rotation curves
  made by the full solution (4.82 and 4.96 SD, line "observed" of part 2). Paper II, Sect. 7.3, item 3: the share of
  AQUAL's own part that the data show, 0.40 to 0.45 by the slope test (the slope itself), and lower by the Delta chi^2
  comparison (its position between the two expectations of part 2).

WHAT IT COMPUTES
  Three models of each rotation curve, all in the fit machinery of two_constant_fit.pl as ported in refit_core.py
  (distance and inclination through q within their catalog errors, priors of 0.10 dex on the mass-to-light ratios),
  with T11, the empirical function of McGaugh, Lelli and Schombert, and a0 fitted on the ladder:
    M  the algebraic relation on the midplane field, as the fits of Paper I use it;
    R  the relation applied at each height and averaged over the gas layer (gas - Pgas of gas_layer_solve.py), which
       adds only Newton's geometry above the plane;
    F  the full AQUAL solution averaged over the gas layer (gas of gas_layer_solve.py), which adds AQUAL's own part,
       the flux steered away from the midplane, on top of R.
  F differs from R only by AQUAL's own part. Both are carried to the fitted mass-to-light ratios by mode 'corrected'
  of refit_core.py, with the direct solutions of gas_layer_solve.py parts upsvar and relvar.
  Part 1 fits the three models to the data, with the Delta chi^2 differences and their SDs over galaxy-bootstrap draws.
  Part 2 is the test. The expected Delta chi^2 (F minus R) on noiseless mocks of F and of R at their best fits
  (prior-centre ratios, q = 1); the observed Delta chi^2 measured from each expectation in SDs. The slope test: the
  residuals of R refitted to the data regressed on the residuals of R refitted to the mock of F (the part of AQUAL's
  own effect that no fitted constant or nuisance absorbs), with log10(g_bar/a0) as a covariate; slope 1 if F is right,
  0 if R is; its SD from galaxy-bootstrap draws. The reverse test swaps R and F. This is the construction of the slope
  test of aqual-refit-fit.py, with R in the place of the midplane relation.

INPUTS
  data/sparc/SPARC_Lelli2016c.mrt, data/sparc/rotmod_ltg/; rotation-curves/aqual-disk-points.csv,
  rotation-curves/aqual-refit-gamma-0.csv and -1.csv (through aqual-refit-fit.py's loader);
  rotation-curves/gas-layer-gamma-0..5.csv, gas-layer-upsvar-0..5.csv, gas-layer-relvar-0..1.csv (gas_layer_solve.py).
  A missing table stops the run.

OUTPUTS AND COMMANDS, from the repository root (about 5 min each; the two can run at the same time)
  python rotation-curves/gas_layer_baseline.py --input model --out rotation-curves/gas_layer_baseline-model-results.txt
  python rotation-curves/gas_layer_baseline.py --input sparc --out rotation-curves/gas_layer_baseline-sparc-results.txt
  --input model uses the model's own Newtonian field for every model, sparc SPARC's tabulated g_bar. --nboot N sets the
  draws for the Delta chi^2 SDs (300), --nslope N those for the slope SDs (2000). Needs numpy.
"""
import csv, importlib.util, math, os, re, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
_spec = importlib.util.spec_from_file_location("aqual_refit_fit", os.path.join(HERE, "aqual-refit-fit.py"))
AF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AF)
RC = AF.RC

T00 = time.time()
ARGS = sys.argv


def arg(name, default=None):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default


NBOOT = int(arg("--nboot", "300"))
NSLOPE = int(arg("--nslope", "2000"))
INP = arg("--input", "model")
LA = RC.LA_T11                           # log10 a0 = -10.7 to -9.3 by 0.1
GAMMA_T = [f"rotation-curves/gas-layer-gamma-{k}.csv" for k in range(6)]
UPSVAR_T = [f"rotation-curves/gas-layer-upsvar-{k}.csv" for k in range(6)]
RELVAR_T = [f"rotation-curves/gas-layer-relvar-{k}.csv" for k in range(2)]
_lines = []


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def stamp(lab):
    out(f"   [{lab}: {time.time() - T00:.0f} s]")


def need(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("tables missing (gas_layer_solve.py): " + ", ".join(missing))
    return paths


def read_gamma_t():
    rows = []
    for p in need(GAMMA_T):
        rows += list(csv.DictReader(open(p, encoding="utf-8-sig")))
    return {(r["galaxy"], round(float(r["R"]), 4)): r for r in rows}


def case_rows(paths, var):
    """Direct solutions at other mass-to-light ratios (columns <var>_T11_a<a0>_d<dlog Yd>_b<dlog Yb>), one row per point
    and case, as Dataset.attach_eps reads them."""
    rows = []
    for p in need(paths):
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            for k, v in r.items():
                m = re.fullmatch(r"(gas|rgas)_(T11)_a([0-9.]+e[+-][0-9]+)_d([+-][0-9.]+)_b([+-][0-9.]+)", k)
                if m and m.group(1) == var and v != "":
                    rows.append(dict(galaxy=r["galaxy"], R=r["R"], law=m.group(2), a0=m.group(3), dlogYd=m.group(4),
                                     dlogYb=m.group(5), Gamma=v))
    return rows


def set_tables(gals, GT, fn):
    """Replace each galaxy's T11 table gam11 (points x 15 a0 nodes) by fn(row, la) of the gas-layer tables."""
    for name, g in gals.items():
        keys = [(name, round(float(R), 4)) for R in g["R"]]
        miss = [k for k in keys if k not in GT]
        if miss:
            raise KeyError(f"{name}: {len(miss)} points without gas-layer rows")
        g["gam11"] = np.array([[fn(GT[k], la) for la in LA] for k in keys])


def boot_sd(vals):
    return float(np.std(np.asarray(vals), ddof=1))


def slope(Py, Px, ds, a0_cov, seed):
    """OLS slope of the residuals in Py on those in Px over points valid in both, covariate log10(g_bar/a0) from Py."""
    x, yv, ly, gid = [], [], [], []
    for gi, nm in enumerate(ds.names):
        a, m = Py[nm], Px[nm]
        ok = a["valid"] & m["valid"] & np.isfinite(a["r"]) & np.isfinite(m["r"])
        x.append(m["r"][ok]); yv.append(a["r"][ok]); ly.append(np.log10(a["gbar"][ok] / a0_cov))
        gid.append(np.full(ok.sum(), gi))
    x, yv, ly, gid = map(np.concatenate, (x, yv, ly, gid))
    b, sb = AF.ols_slope(x, yv, [ly], gid, NSLOPE, seed)
    return b, sb, x, len(x)


def main():
    out(f"The gas-layer baseline test, T11 (the empirical function), input '{INP}'.")
    out(f"Delta chi^2 SDs from {NBOOT} galaxy-bootstrap draws; slope SDs from {NSLOPE}.")
    out("=" * 112)
    T1, gals = AF.load()
    GT = read_gamma_t()
    set_tables(gals, GT, lambda r, la: float(r[f"gas_{la:.1f}"]))
    dsF = RC.Dataset(gals)
    dsF.attach_eps(case_rows(UPSVAR_T, "gas"))
    set_tables(gals, GT, lambda r, la: float(r[f"gas_{la:.1f}"]) - float(r[f"Pgas_{la:.1f}"]))
    dsR = RC.Dataset(gals)
    dsR.attach_eps(case_rows(RELVAR_T, "rgas"))
    gF = np.concatenate([g.gok for g in dsF.groups if g.G]); gR = np.concatenate([g.gok for g in dsR.groups if g.G])
    out(f"   {len(dsF.names)} galaxies, {gF.size} points; validity masks identical: {bool(np.all(gF == gR))}")
    if not np.all(gF == gR):
        raise SystemExit("masks differ")
    seed = 20261011 if INP == "model" else 20261012
    c_alg = AF.ctx_of(INP, "T11"); c_full = AF.ctx_of(INP, "T11", full=True, gmode="corrected")
    la0 = math.log10(1.05e-10)

    # ---------------------------------------------------------------- part 1
    out("\nPart 1. The three models fitted to the data (M is the relation's fit of aqual-refit-main-T11-results.txt)")
    t0 = time.time()
    M = AF.LawFit(dsF, c_alg, la0)
    R = AF.LawFit(dsR, c_full, M.la)
    F = AF.LawFit(dsF, c_full, M.la)
    out(f"   M midplane relation:        {M.desc()}")
    out(f"   R gas-layer relation:       {R.desc()}")
    out(f"   F gas-layer full solution:  {F.desc()}  [{time.time() - t0:.0f} s]")
    W = RC.boot_weights(len(dsF.names), NBOOT, seed)
    dFR, dRM, dFM = [], [], []
    for b in range(NBOOT):
        ym = M.minw(W[b])[0]; yr = R.minw(W[b])[0]; yf = F.minw(W[b])[0]
        dFR.append(yf - yr); dRM.append(yr - ym); dFM.append(yf - ym)
    res = {}
    for lab, obs, dd in (("F minus R", F.chi2 - R.chi2, dFR), ("R minus M", R.chi2 - M.chi2, dRM), ("F minus M", F.chi2 - M.chi2, dFM)):
        sd = boot_sd(dd)
        res[lab] = dict(obs=float(obs), sd=sd)
        out(f"   Delta chi^2 {lab:<10s} {obs:+8.1f} +/- {sd:6.1f} ({obs / sd:+.2f} SD)")
    stamp("part 1")

    # ---------------------------------------------------------------- mocks
    mockF = dict(zip(("Vo", "pmask"), dsF.mock(F.ctx())))
    mockR = dict(zip(("Vo", "pmask"), dsR.mock(R.ctx())))

    def fit_on(ds, base, la, mock):
        f = AF.LawFit(ds, base, la, extra=mock)
        return f, ds.points(dict(f.ctx(), **mock), f.best)

    # ---------------------------------------------------------------- part 2
    out("\nPart 2. The baseline test: R (gas-layer relation) as the baseline, F (gas-layer full solution) as the alternative")
    t0 = time.time()
    rF, PrF = fit_on(dsR, c_full, F.la, mockF)          # R refitted to the mock of F: x of the slope test
    fF, _ = fit_on(dsF, c_full, F.la, mockF)
    rR, _ = fit_on(dsR, c_full, R.la, mockR)
    fR, PfR = fit_on(dsF, c_full, R.la, mockR)          # F refitted to the mock of R: x of the reverse test
    E_F = fF.chi2 - rF.chi2; E_R = fR.chi2 - rR.chi2
    obs = F.chi2 - R.chi2; sd = res["F minus R"]["sd"]
    out(f"   mock of F: R refitted {rF.desc()}; F refitted {fF.desc()}; expected Delta chi^2 (F minus R) {E_F:+.1f}")
    out(f"   mock of R: R refitted {rR.desc()}; F refitted {fR.desc()}; expected Delta chi^2 (F minus R) {E_R:+.1f}")
    out(f"   observed {obs:+.1f} +/- {sd:.1f}: from F's expectation {(obs - E_F) / sd:+.2f} SD, from R's expectation {(obs - E_R) / sd:+.2f} SD  [{time.time() - t0:.0f} s]")
    pts_R = dsR.points(R.ctx(), R.best); pts_F = dsF.points(F.ctx(), F.best)
    b, sb, x, n = slope(pts_R, PrF, dsR, 10 ** R.la, seed + 7)
    out(f"   Slope test, refit residuals of R on the unabsorbed AQUAL part (1 if F is right, 0 if R is): slope {b:+.3f} +/- {sb:.3f}; "
        f"rejection of F {(1 - b) / sb:.1f} SD, of R {b / sb:+.1f} SD; rms of the unabsorbed part {np.sqrt(np.mean(x ** 2)):.4f} dex ({n} points)")
    ax = np.abs(x)
    out(f"      the unabsorbed part: median |.| {np.median(ax):.4f} dex, points beyond 0.05 dex {int(np.sum(ax > 0.05))}, beyond 0.1 dex {int(np.sum(ax > 0.1))}")
    b2, sb2, x2, _ = slope(pts_F, PfR, dsF, 10 ** F.la, seed + 7)
    out(f"   Reverse slope test, refit residuals of F on the unabsorbed -(AQUAL part) (1 if R is right, 0 if F is): slope {b2:+.3f} +/- {sb2:.3f}; "
        f"rejection of R {(1 - b2) / sb2:.1f} SD, of F {b2 / sb2:+.1f} SD; rms {np.sqrt(np.mean(x2 ** 2)):.4f} dex")
    rx = (np.concatenate([pts_R[nm]["r"][pts_R[nm]["valid"] & pts_F[nm]["valid"]] for nm in dsR.names]),
          np.concatenate([pts_F[nm]["r"][pts_R[nm]["valid"] & pts_F[nm]["valid"]] for nm in dsR.names]))
    out(f"   Scatter of the residuals about each model: R {np.std(rx[0]):.4f} dex, F {np.std(rx[1]):.4f} dex")
    stamp("part 2")


if __name__ == "__main__":
    main()
    out(f"\n[run time {time.time() - T00:.0f} s]")
    if arg("--out"):
        with open(arg("--out"), "w", newline="\n", encoding="utf-8") as fh:
            fh.write("\n".join(_lines) + "\n")
