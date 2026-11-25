"""The saved ladders of the catalog-distance check (Paper I section 4.2, Paper II section 6.1; the cases are defined in
catalog_distances_common.py). For one case, the boost law of Paper I, Eq. (8) (code T3 in refit_core.py), at each
fixed delta, a0 refitted by refit_core.Ladder,
every galaxy's mass-to-light ratios fitted under 0.10 dex priors, and q profiled (--q fit) or held at 1 (--q 1,
catalog distances and inclinations). One JSON file per delta in rotation-curves/catalog-distances-nodes/, holding the
per-galaxy objectives along the a0 profile for the galaxy bootstrap of catalog_distances.py. A delta whose file exists
is skipped. The folder is not deposited and git ignores it.

    python rotation-curves/catalog_distances_nodes.py --input sparc|model --model full|rel --q fit|1 --deltas 0.003,0.005,... [--la0 -10.1]

Each ladder starts from the a0 of the one before it in the same run, so catalog_distances-results.txt is reproduced
by the same runs with the same deltas in the same order. Every case has two runs from the repository root, the scan
and then the added ladders:
  the scan, for each of the eight cases:   --deltas 0.003,0.005,0.01,0.03,0.1,0.2,0.3,0.4
  the added ladders:
    --input sparc --model full --q fit   --deltas 0.1377,0.149,0.1661,0.1797
    --input sparc --model rel --q fit    --deltas 0.1498,0.1621,0.1806,0.1953,0.2972
    --input model --model full --q fit   --deltas 0.13,0.1406,0.1567,0.1695
    --input model --model rel --q fit    --deltas 0.0267,0.1418,0.1534,0.1709,0.1849
    --input sparc --model full --q 1     --deltas 0.05,0.075,0.1174,0.1273,0.1422,0.1542,0.175
    --input sparc --model rel --q 1      --deltas 0.1162,0.126,0.1408,0.1526,0.2577
    --input model --model full --q 1     --deltas 0.02,0.0259,0.05,0.07,0.1162,0.126,0.1407,0.1525
    --input model --model rel --q 1      --deltas 0.1156,0.1253,0.14,0.1517,0.1926
The added ladders are the minima and the points where kappa or kappa_dS equals 1, which catalog_distances.py asks for
in its TODO lines, with a few more about the minima. Run time, with the eight cases run in parallel, one process
each: 9 to 10 minutes for the scan of a full-solution case and 7 to 9.5 minutes for its added ladders, 1.5 to
2.5 minutes for each run of a relation case; about 90 minutes of processor time in all.
"""
import json, math, os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import catalog_distances_common as C
RC = C.RC

A = sys.argv
arg = lambda n, d=None: A[A.index(n) + 1] if n in A else d
inp, model, qmode = arg("--input"), arg("--model"), arg("--q")
deltas = [float(x) for x in arg("--deltas").split(",")]
tag = C.case_tag(inp, model, qmode)
la = float(arg("--la0")) if arg("--la0") else math.log10(0.95e-10 * (0.8 if qmode == "1" else 1.0))
os.makedirs(C.NODEDIR, exist_ok=True)

T0 = time.time()
T1, gals = C.load()
ds = RC.Dataset(gals, frozen=(qmode == "1"))
if model == "full":
    ds.attach_eps(C.upsvar_rows())
nbad = sum(int((~g.gok).sum()) for g in ds.groups if g.G)
print(f"{tag}: {len(ds.names)} galaxies, excluded points {nbad}, setup {time.time() - T0:.0f} s", flush=True)
for d in deltas:
    path = C.node_path(tag, d)
    if os.path.exists(path):
        r = json.load(open(path, encoding="utf-8")); la = r["la"]
        print(f"   delta {d:g}: exists (chi2 {r['chi2']:.2f}, a0 {10 ** la * 1e10:.4f}e-10)", flush=True)
        continue
    t0 = time.time()
    L = RC.Ladder(ds, C.ctx(inp, model == "full", d), la)
    la = L.la
    b = L.best
    rec = dict(tag=tag, input=inp, model=model, q=qmode, delta=d, la=L.la, sig_la=L.sig_la, chi2=L.chi2,
               a0=10 ** L.la, per={f"{k:.6f}": [float(x) for x in v] for k, v in L.per.items()},
               names=list(ds.names), u=[float(x) for x in b["u"]],
               v=[float(x) if np.isfinite(x) else None for x in b["v"]], q_fit=[float(x) for x in b["q"]],
               F=[float(x) for x in b["F"]])
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh)
    print(f"   delta {d:g}: a0 {10 ** L.la * 1e10:.4f}e-10, chi2 {L.chi2:.2f}, {len(L.v)} ladder nodes [{time.time() - t0:.0f} s]", flush=True)
print(f"[run time {time.time() - T0:.0f} s]", flush=True)
