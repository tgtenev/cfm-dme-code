"""Generate the catalogue's pieces of five and six sites: every five-site piece (29 under rotations) and every six-site
piece with six or fewer labels, each with its tiling test, label count, turning hole, bounding box
and a systematic name pN-MM (N sites, MM the running number). Writes pieces2.json beside this script
(name -> cells, the pieces that tile by translation) and pieces2-info.tsv (every piece: sites, labels,
bounding box, turning hole as fresh sites over sites, the tiling test and the cells), and prints the table.
These pieces enter the catalogue of unit shapes of Paper II, Sec. 5.2.

Usage (from the repository root; about ten seconds; optional, since its two outputs are committed, and run
before polycubes.py): py -3 -X utf8 -I avalanche/unit-shapes/catalogue/gen_catalogue2.py
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling modules, also under -I
from enumerate_polycubes import canon, grow, n_labels  # noqa: E402
from polycubes import orientations, turning_moves  # noqa: E402
from tilingtest import tiles_by_translation  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def bbox(p):
    pts = np.array(p)
    return tuple(sorted((pts.max(axis=0) - pts.min(axis=0) + 1).tolist(), reverse=True))


def main():
    polys = {canon([(0, 0, 0)])}
    bysize = {1: polys}
    for size in range(2, 7):
        polys = grow(polys)
        bysize[size] = polys
    chosen = []
    for size in (5, 6):
        for p in sorted(bysize[size]):
            q = n_labels(p)
            if size == 6 and q > 6:
                continue
            chosen.append((size, q, bbox(p), p))
    chosen.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    out = {}
    counters = Counter()
    info = ["name\tsites\tlabels\tbox\thole\ttiles\tcells"]
    print("name      sites labels box      hole    tiles  cells")
    for size, q, bb, p in chosen:
        counters[size] += 1
        name = f"p{size}-{counters[size]:02d}"
        cells = [[int(v) for v in c] for c in p]
        oris = orientations(cells)
        _, vmin = turning_moves(oris)
        n = next(k for k in (4, 6, 8, 10, 12) if (k ** 3) % size == 0)
        tiles = tiles_by_translation(cells, n)
        cstr = ";".join(",".join(map(str, c)) for c in cells)
        if tiles:
            out[name] = cells
        info.append(f"{name}\t{size}\t{q}\t{'x'.join(map(str, bb))}\t{vmin}/{size}\t{tiles}\t{cstr}")
        print(f"{name:9s} {size:5d} {q:6d} {'x'.join(map(str, bb)):8s} {vmin}/{size}={vmin/size:.3f} {str(tiles):6s} {cstr}")
    with open(os.path.join(HERE, "pieces2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(HERE, "pieces2-info.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(info) + "\n")
    print(f"{len(out)} tiling pieces written to pieces2.json; {len(info) - 1} rows in pieces2-info.tsv")


if __name__ == "__main__":
    main()
