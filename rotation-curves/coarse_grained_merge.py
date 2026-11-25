"""Complete the boost-law (T3) columns of a coarse-grained Gamma table to the full grid of aqual-refit-solve.py (delta
0.05, 0.2, 0.4, 0.7, 1.2 by log10 a0 = -10.7 to -9.3 in steps of 0.2) from coarse-grained solutions at fewer a0 nodes:
    Gamma_coarse(delta, la) = Gamma_plain(delta, la) + dGamma(delta, la),
with dGamma = Gamma_coarse - Gamma_plain known at the computed nodes (all five delta, the a0 nodes of --t3la in
coarse_grained_solve.py) and interpolated in log10 a0 by the cubic through the four nodes (Lagrange), extrapolated by
the same cubic to the outermost nodes of the grid. Gamma_plain is the plain AQUAL table, aqual-refit-gamma-0.csv and
-1.csv. Writes galaxy, R and the forty G_T3_ columns (T3: the boost law of Paper I, Eq. (8)), and prints how well the
merged table reproduces the computed nodes.

Supports Paper II, Sect. 6.1, through the two-constant fit coarse-grained at 100 pc (coarse_grained_fit.py, tag
B100-T3). The merged table is not deposited: it is rebuilt in seconds. Command, from the repository root:
    python rotation-curves/coarse_grained_merge.py --in rotation-curves/coarse-gamma-B100-T3r-0.csv,rotation-curves/coarse-gamma-B100-T3r-1.csv --out rotation-curves/coarse-gamma-B100-T3-merged.csv
"""
import csv, sys
import numpy as np

ARGS = sys.argv
arg = lambda n, d=None: ARGS[ARGS.index(n) + 1] if n in ARGS else d
LA_T3 = [round(-10.7 + 0.2 * k, 1) for k in range(8)]
DEL = [0.05, 0.2, 0.4, 0.7, 1.2]
plain = {}
for f in ("rotation-curves/aqual-refit-gamma-0.csv", "rotation-curves/aqual-refit-gamma-1.csv"):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        plain[(r["galaxy"], round(float(r["R"]), 4))] = r
rows = {}
for f in arg("--in").split(","):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        rows.setdefault((r["galaxy"], round(float(r["R"]), 4)), {}).update(r)
k0 = next(iter(rows.values()))
las = sorted({float(c.split("_")[3]) for c in k0 if c.startswith("G_T3_")})
print(f"computed a0 nodes: {las}; points {len(rows)}")


def lag(xs, x):
    w = np.ones(len(xs))
    for i, xi in enumerate(xs):
        for j, xj in enumerate(xs):
            if j != i:
                w[i] *= (x - xj) / (xi - xj)
    return w


W = {la: lag(las, la) for la in LA_T3}
cols = [f"G_T3_{d:g}_{la:.1f}" for d in DEL for la in LA_T3]
mx = 0.0
with open(arg("--out"), "w", newline="\n", encoding="utf-8") as fh:
    fh.write("galaxy,R," + ",".join(cols) + "\n")
    for k, r in rows.items():
        p = plain[k]
        vals = []
        for d in DEL:
            dG = np.array([float(r[f"G_T3_{d:g}_{la}"]) - float(p[f"G_T3_{d:g}_{la:.1f}"]) for la in las])
            for la in LA_T3:
                v = float(p[f"G_T3_{d:g}_{la:.1f}"]) + float(W[la] @ dG)
                if la in las:
                    mx = max(mx, abs(v - float(r[f"G_T3_{d:g}_{la}"])) if np.isfinite(v) else 0.0)
                vals.append(f"{v:.6f}")
        fh.write(f"{r['galaxy']},{r['R']}," + ",".join(vals) + "\n")
print(f"written {arg('--out')}; reproduces the computed nodes within {mx:.1e}")
