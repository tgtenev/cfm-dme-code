"""The results table of the catalogue of unit shapes (Paper II, Sec. 5.2 and Appendix C): one row per shape.

Command (from the repository root; seconds; after the jamming and avalanche runs, before
rebuild/rebuild_catalogue.py and pair-holes/gen_holes.py):

    py -3 -X utf8 -I avalanche/unit-shapes/catalogue/build_results.py avalanche/unit-shapes/catalogue/catalogue-results.tsv \\
        avalanche/unit-shapes/catalogue avalanche/unit-shapes avalanche/unit-shapes/rod

Usage: py -3 build_results.py <out.tsv> <folder> [<folder> ...]. It reads, in each folder, the jamming runs
rsa-<shape>-s<seed>.tsv of polycubes.py and lattice-L<L>-s<seed>.tsv of latticebricks.py (the rods), the
avalanche runs <shape>-eta<eta>.tsv and L<L>-eta<eta>.tsv (the rods) of lattice_sim.pl, and the shape tables
pieces2-info.tsv and pieces3-info.tsv. For each shape it takes the mean raw packing error delta0 over the
seeds and the turning hole ca, and locates the crossing of A d_emit^2 = 2 by the rule of lattice_analyze.py
(ln(A d_emit^2) linear in ln eta between the bracketing runs, d_emit at the same fraction, A = 2/d_emit^2,
D = 1/ln(1 + d_emit)). It writes the table to <out.tsv> and prints it as Markdown, sorted by divisor.

The column standing places D against the target divisor window 5.70 to 6.19 of Paper II, Sec. 5.2 (target)
and the empirical function's divisors 4.52 to 5.61 of App. C (empirical); above, between and below are the
other places. Paper II results it supports: the rod of four sites (rod4) at delta0 0.2818, ca 0.75, eta
0.964 percent, d_emit 0.1879, A 56.6 and D 5.807 (Sec. 5.1 and 5.2); the four other shapes in the target
window under the cheapest hole, brick124 5.926, box145 5.938, p6-05 5.739 and box234 5.712 (App. C); the
largest shape, of 36 sites. Shape names: rodL is the 1 x 1 x L rod, pN-MM the generated pieces of N sites
(gen_catalogue2.py), boxLMN the L x M x N blocks (gen_boxes.py); the others are named in NAMES below.
"""
import csv, glob, math, os, re, sys
from collections import defaultdict

TARGET = (5.70, 6.19)      # the target divisor window of Paper II, Sec. 5.2
EMPIRICAL = (4.52, 5.61)   # the empirical function's divisors, Paper II, App. C

NAMES = {  # display names for the built-in and generated shapes
    "rod2": "rod of 2 (dimer)", "rod3": "rod of 3", "rod4": "rod of 4", "rod5": "rod of 5",
    "rod6": "rod of 6", "rod8": "rod of 8", "plate22": "1 x 2 x 2 plate", "plate23": "1 x 2 x 3 brick (second entry)",
    "plate33": "1 x 3 x 3 plate", "brick123": "1 x 2 x 3 brick", "brick124": "1 x 2 x 4 brick",
    "brick223": "2 x 2 x 3 block", "Ltro": "L tromino", "Ltet": "L tetromino", "Ttet": "T tetromino",
    "Stet": "S tetromino", "corner": "corner (tripod)", "screw": "screw (chiral tetracube)",
}


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from polycubes import PIECES
    import numpy as np
    BOXES = {name: "x".join(map(str, sorted((np.array(c).max(axis=0) - np.array(c).min(axis=0) + 1).tolist(), reverse=True)))
             for name, c in PIECES.items()}
    for L in (2, 3, 4, 5, 6, 8):
        BOXES[f"rod{L}"] = f"{L}x1x1"
except Exception:
    BOXES = {}


def read_tsv(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [r for r in csv.DictReader(fh, delimiter="\t")]


def main():
    out = sys.argv[1]
    folders = sys.argv[2:]
    jam = defaultdict(list)      # shape -> list of jamming rows
    av = defaultdict(list)       # shape -> list of (eta, A, d_emit, Ad2)
    info = {}
    for f in folders:
        for p in glob.glob(os.path.join(f, "rsa-*.tsv")):
            for r in read_tsv(p):
                jam[r["piece"]].append(r)
        for p in glob.glob(os.path.join(f, "lattice-L*-s*.tsv")):   # the rods, from latticebricks.py
            for r in read_tsv(p):
                r2 = dict(piece=f"rod{r['L']}", cells=r["L"], q="3", delta0=r["delta0"],
                          vstar_over_v=r["vstar_over_v"])
                jam[r2["piece"]].append(r2)
        for p in glob.glob(os.path.join(f, "*-eta*.tsv")):
            m = re.match(r"(.+)-eta([\d.]+)\.tsv", os.path.basename(p))
            if not m:
                continue
            name = m.group(1)
            if name.startswith("L") and name[1:].isdigit():          # the rod series: L4-eta...
                name = "rod" + name[1:]
            for r in read_tsv(p):
                av[name].append((float(r["eta"]), float(r["A"]), float(r["d_emit"]), float(r["Ad2"])))
        for ip in glob.glob(os.path.join(f, "pieces*-info.tsv")):
            for r in read_tsv(ip):
                info[r["name"]] = r
    rows = []
    for shape, jr in jam.items():
        d0 = sum(float(r["delta0"]) for r in jr) / len(jr)
        ca = float(jr[0]["vstar_over_v"])
        q = int(jr[0]["q"])
        cells = int(jr[0]["cells"])
        box = info.get(shape, {}).get("box", BOXES.get(shape, ""))
        pts = sorted(av.get(shape, []))
        cross = None
        for a, b in zip(pts, pts[1:]):
            if (a[3] - 2) * (b[3] - 2) <= 0 and a[3] != b[3]:
                t = (math.log(2) - math.log(a[3])) / (math.log(b[3]) - math.log(a[3]))
                eta = math.exp(math.log(a[0]) + t * (math.log(b[0]) - math.log(a[0])))
                dem = a[2] + t * (b[2] - a[2])
                cross = (eta, dem, 2 / dem ** 2, 1 / math.log(1 + dem))
                break
        if cross:
            D = cross[3]
            standing = ("target" if TARGET[0] <= D <= TARGET[1] else
                        "empirical" if EMPIRICAL[0] <= D <= EMPIRICAL[1] else
                        "above" if D > TARGET[1] else "between" if D > EMPIRICAL[1] else "below")
            rows.append(dict(shape=shape, name=NAMES.get(shape, shape), cells=cells, q=q, box=box,
                             delta0=round(d0, 4), ca=round(ca, 4), eta=round(cross[0], 5),
                             d_emit=round(cross[1], 4), A=round(cross[2], 1), D=round(D, 3), standing=standing))
        else:
            mx = max((p[1] for p in pts), default=float("nan"))
            rows.append(dict(shape=shape, name=NAMES.get(shape, shape), cells=cells, q=q, box=box,
                             delta0=round(d0, 4), ca=round(ca, 4), eta="", d_emit="", A=f"<= {mx:.0f}" if pts else "",
                             D="", standing="no crossing" if pts else "not run"))
    rows.sort(key=lambda r: (r["D"] == "", -(r["D"] or 0)))
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print("| Shape | Sites | Labels | Box | Raw error | Turning hole | Efficiency | Emitted error | Amplification | Divisor | Standing |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        eta = f"{100 * r['eta']:.2f} %" if r["eta"] != "" else "-"
        print(f"| {r['name']} | {r['cells']} | {r['q']} | {r['box']} | {r['delta0']:.3f} | {r['ca']:.3f} | {eta} | "
              f"{r['d_emit'] if r['d_emit'] != '' else '-'} | {r['A']} | {r['D'] if r['D'] != '' else '-'} | {r['standing']} |")
    n = sum(1 for r in rows if r["D"] != "")
    print(f"\n{len(rows)} shapes; {n} with a crossing; target window: {sum(r['standing'] == 'target' for r in rows)}; "
          f"empirical: {sum(r['standing'] == 'empirical' for r in rows)}")


if __name__ == "__main__":
    main()
