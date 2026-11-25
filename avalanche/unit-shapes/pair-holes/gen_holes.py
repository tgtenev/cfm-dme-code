"""Hole matrices for the simulator with pair-specific turning holes (Paper II, Appendix C), and the choice of
the shapes worth running with it.

Command (from the repository root; about 5 s; after catalogue/build_results.py and
catalogue/rebuild/rebuild_catalogue.py, before lattice_sim_pair.pl):

    py -3 -X utf8 -I avalanche/unit-shapes/pair-holes/gen_holes.py

For every shape of the catalogue of unit shapes, the matrix H[a][b] is the fewest fresh sites a unit in
orientation a needs to re-lay itself in orientation b, over any placement that shares a site, divided by the
site count. A pair-specific hole lies between the shape's cheapest and largest entries, and the divisor falls
as the hole grows, so a shape can land in the target divisor window (5.70 to 6.19, Paper II, Sec. 5.2) only if
its divisor at the cheapest hole is at least 5.70 and its divisor at the largest hole is at most 6.19. The
divisors at the two ends come from a fit of ln D, quadratic in ln d0, ln c_a and ln q, to the crossings in
catalogue/rebuild/rebuild-crossings.tsv. Shapes whose holes are all equal are left out, since the pair-specific
holes change nothing for them, except the rod of four sites (rod4), which is kept as the reference: with its
equal holes lattice_sim_pair.pl reproduces lattice_sim.pl bit for bit.

Inputs: catalogue/polycubes.py (the shapes and their turning moves), catalogue/pieces2.json and pieces3.json,
catalogue/catalogue-results.tsv (each shape's raw packing error delta0) and catalogue/rebuild/rebuild-crossings.tsv.
Outputs, beside this script: holes-<shape>.txt for each chosen shape (the matrix, one row per orientation a),
selection.tsv (every shape with q, delta0, its cheapest, mean and largest turning hole c_min, c_mean, c_max, the
fitted divisors at the two ends, whether its holes are all equal and whether it is chosen), and a summary table
on standard output.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAT = os.path.join(HERE, "..", "catalogue")
SR = os.path.join(CAT, "rebuild")
sys.path.insert(0, CAT)
from polycubes import PIECES, orientations, turning_moves  # noqa: E402

for extra in ("pieces2.json", "pieces3.json"):
    with open(os.path.join(CAT, extra), encoding="utf-8") as f:
        for name, cells in json.load(f).items():
            PIECES[name] = [tuple(c) for c in cells]
for L in (5, 6, 8):
    PIECES.setdefault(f"rod{L}", [(i, 0, 0) for i in range(L)])

# ---- the fit of the divisor to the catalogue's crossings ----
with open(os.path.join(SR, "rebuild-crossings.tsv"), encoding="utf-8") as f:
    cr = list(csv.DictReader(f, delimiter="\t"))


def design(u, v, w):
    return np.column_stack([np.ones_like(u), u, v, w, u * u, v * v, u * v, w * w, u * w, v * w])


d0 = np.array([float(r["d0_run"]) for r in cr]); ca = np.array([float(r["ca"]) for r in cr])
qq = np.array([float(r["q"]) for r in cr]); D = np.array([float(r["D"]) for r in cr])
X = design(np.log(d0), np.log(ca), np.log(qq))
bD, *_ = np.linalg.lstsq(X, np.log(D), rcond=None)


def fit(b, d, c, q):
    return float(np.exp(design(np.log(np.atleast_1d(d)), np.log(np.atleast_1d(c)), np.log(np.atleast_1d(q))) @ b)[0])


# ---- the raw packing errors of the catalogue (the avalanche runs' inputs) ----
d0tab = {}
with open(os.path.join(CAT, "catalogue-results.tsv"), encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        d0tab[r["shape"]] = float(r["delta0"])

rows, chosen = [], []
for name, d in sorted(d0tab.items()):
    if name not in PIECES or name in ("plate23", "p6-03"):     # the two duplicates of the 1 x 2 x 3 brick
        continue
    cells = PIECES[name]
    oris = orientations(cells)
    q = len(oris)
    if q < 2:
        continue
    moves, _ = turning_moves(oris)
    n = len(cells)
    H = np.zeros((q, q))
    for a in range(q):
        best = {}
        for b, t, fresh in moves[a]:
            best[b] = min(best.get(b, n), fresh)
        for b in range(q):
            if b != a:
                H[a, b] = best[b] / n
    off = H[~np.eye(q, dtype=bool)]
    cmin, cmean, cmax = off.min(), off.mean(), off.max()
    Dmin, Dmax = fit(bD, d, cmin, q), fit(bD, d, cmax, q)
    uniform = abs(cmax - cmin) < 1e-12
    pick = (not uniform and Dmin >= 5.70 and Dmax <= 6.19) or name == "rod4"
    rows.append(dict(shape=name, n=n, q=q, d0=round(d, 4), c_min=round(cmin, 4), c_mean=round(cmean, 4),
                     c_max=round(cmax, 4), D_at_min=round(Dmin, 3), D_at_max=round(Dmax, 3),
                     uniform=uniform, chosen=pick))
    if pick:
        chosen.append(name)
        with open(os.path.join(HERE, f"holes-{name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"# {name}: fresh sites over {n} for a re-laying from orientation a (row) to b (column)\n")
            for a in range(q):
                f.write(" ".join(f"{H[a, b]:.6f}" for b in range(q)) + "\n")

with open(os.path.join(HERE, "selection.tsv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
    w.writeheader(); w.writerows(rows)

print("shape      q   d0      c_min  c_mean c_max   D@min  D@max  chosen")
for r in rows:
    if r["chosen"] or (not r["uniform"] and r["D_at_min"] >= 5.5):
        print(f"{r['shape']:9s} {r['q']:3d}  {r['d0']:.4f}  {r['c_min']:.3f}  {r['c_mean']:.3f}  {r['c_max']:.3f}  "
              f"{r['D_at_min']:.3f}  {r['D_at_max']:.3f}  {r['chosen']}")
print(f"{len(chosen)} shapes chosen")
