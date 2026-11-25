"""The fits of aqual-refit-fit.py with the full AQUAL solution coarse-grained at the catchment length: does the
coarse-grained full solution fit the SPARC rotation curves as well as the algebraic relation?

RESULTS SUPPORTED
  Paper II, Sect. 6.1: coarse-grained at catchment lengths of 100 and 200 pc, the full solution trails the algebraic
  relation by 1.5 to 1.8 SD for the empirical function and for the two-constant fit (the SUMMARY of each output), and in
  the two-constant fit on the coarse-grained full solution the free volume falls to the scan's lower stop, 0.03, with
  chi^2 still falling at the probes down to 0.004 (coarse_grained_fit-B100-T3-results.txt).

WHAT IT COMPUTES
  Part main of aqual-refit-fit.py, unchanged, with the Gamma tables of the coarse-grained solution
  (coarse_grained_solve.py) in place of those of plain AQUAL. The tables of --gamma replace the columns of the same name
  in aqual-refit-gamma-0.csv and -1.csv; columns they lack keep plain AQUAL's values. The correction of mode
  'corrected' (the error of the uniform-scaling reading against direct solutions at other mass-to-light ratios) is
  computed from plain AQUAL and carried to the coarse-grained tables unchanged.
  Codes: B100 and B200 are the coarse graining of form B (scale-selective coupling, coarse_grained_solve.py) at
  catchment lengths of 100 and 200 pc; T11 is the empirical function of McGaugh, Lelli and Schombert, with a0 fitted;
  T3 is the boost law of Paper I, Eq. (8), with a0 and the free volume fitted (the two-constant fit).

INPUTS
  Those of aqual-refit-fit.py, and the coarse-grained tables rotation-curves/coarse-gamma-*.csv
  (coarse_grained_solve.py; the T3 table at 100 pc is merged by coarse_grained_merge.py first). A missing table stops
  the run.

OUTPUTS AND COMMANDS, from the repository root (500 bootstrap draws, against 2,000 in aqual-refit-fit.py)
  python rotation-curves/coarse_grained_fit.py --law T11 --gamma rotation-curves/coarse-gamma-B100-T11-0.csv,rotation-curves/coarse-gamma-B100-T11-1.csv,rotation-curves/coarse-gamma-B100-T11-2.csv --tag B100-T11 --nboot 500 --out rotation-curves/coarse_grained_fit-B100-T11-results.txt
      (about 6 min)
  python rotation-curves/coarse_grained_fit.py --law T11 --gamma rotation-curves/coarse-gamma-B200-T11-0.csv,rotation-curves/coarse-gamma-B200-T11-1.csv,rotation-curves/coarse-gamma-B200-T11-2.csv,rotation-curves/coarse-gamma-B200-T11-3.csv --tag B200-T11 --nboot 500 --out rotation-curves/coarse_grained_fit-B200-T11-results.txt
      (about 4 min)
  python rotation-curves/coarse_grained_merge.py --in rotation-curves/coarse-gamma-B100-T3r-0.csv,rotation-curves/coarse-gamma-B100-T3r-1.csv --out rotation-curves/coarse-gamma-B100-T3-merged.csv
  python rotation-curves/coarse_grained_fit.py --law T3 --gamma rotation-curves/coarse-gamma-B100-T11-0.csv,rotation-curves/coarse-gamma-B100-T11-1.csv,rotation-curves/coarse-gamma-B100-T11-2.csv,rotation-curves/coarse-gamma-B100-T3-merged.csv --tag B100-T3 --nboot 500 --nomock --out rotation-curves/coarse_grained_fit-B100-T3-results.txt
      (about 34 min; --nomock skips the mocks, the slope tests and the point tables)
  --tag names the run in the output. Needs numpy.
"""
import csv, importlib.util, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_spec = importlib.util.spec_from_file_location("aqual_refit_fit", os.path.join(HERE, "aqual-refit-fit.py"))
AF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AF)
RC, np, out = AF.RC, AF.np, AF.out

CATCH_FILES = [f for f in (AF.arg("--gamma") or "").split(",") if f]
TAG = AF.arg("--tag", "test")


def swap_to_catch(ds, gals):
    """Replace the plain AQUAL Gamma tables in every group by the coarse-grained tables (columns present in
    CATCH_FILES), after the correction has been attached from plain AQUAL. The points mask gok is recomputed."""
    if CATCH_FILES:
        plain = RC.read_gamma(AF.GAMMA_FILES)
        catch = {}
        for p in CATCH_FILES:                    # files may hold different columns of the same points: merge per point
            for r in csv.DictReader(open(p, encoding="utf-8-sig")):
                catch.setdefault((r["galaxy"], round(float(r["R"]), 4)), {}).update(r)
        merged = {}
        ncol = set()
        for k, r in plain.items():
            rr = dict(r)
            if k in catch:
                for c, v in catch[k].items():
                    if c.startswith("G_") and v != "":
                        rr[c] = v; ncol.add(c)
            merged[k] = rr
        missing = [k for k in plain if k not in catch]
        out(f"Coarse-grained tables: {len(catch)} points from {', '.join(os.path.basename(f) for f in CATCH_FILES)}; "
            f"{len(ncol)} columns replaced; {len(missing)} points without coarse-grained rows keep plain AQUAL's Gamma"
            + (" (" + ", ".join(sorted({k[0] for k in missing})[:8]) + ")" if missing else ""))
        PTS = RC.read_points("rotation-curves/aqual-disk-points.csv")
        RC.attach(gals, merged, PTS, strict=True)
        for grp in ds.groups:
            if grp.G == 0:
                continue
            cat = lambda key: np.concatenate([g[key][s] for g, s in zip(grp.gl, grp.sel)])
            g11, g3 = cat("gam11"), cat("gam3")
            grp.gok = (grp.gN0 > 0) & np.all(np.isfinite(g11), axis=1) & np.all(np.isfinite(g3), axis=(1, 2))
            grp.gam11 = np.where(np.isfinite(g11), g11, 0.0)
            grp.gam3 = np.where(np.isfinite(g3), g3, 0.0)
    out(f"Bootstrap draws: {AF.NBOOT} (aqual-refit-fit.py: 2000). Tag: {TAG}")


AF.PREPARE = swap_to_catch

if __name__ == "__main__":
    AF.part_main()
    out(f"\n[run time {time.time() - AF.T00:.0f} s]")
    if AF.arg("--out"):
        with open(AF.arg("--out"), "w", newline="\n", encoding="utf-8") as fh:
            fh.write("\n".join(AF._lines) + "\n")
