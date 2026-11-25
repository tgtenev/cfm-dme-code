"""The box family: every rectangular block of sites l x m x n with l <= m <= n, n <= 6, l, m <= 4 and
at most 36 sites, except the cubes (no label) and the boxes the catalogue already holds (rods of 2 to 6 and 8, the 1 x 2 x 2, 1 x 3 x 3,
1 x 2 x 3, 1 x 2 x 4 and 2 x 2 x 3). Writes pieces3.json beside this script, with the label count, the
turning hole and the tiling test (every box tiles by translation) in pieces3-info.tsv. The boxes enter the
catalogue of unit shapes of Paper II, Sec. 5.2; the largest have 36 sites.

Usage (from the repository root; seconds; optional, since its two outputs are committed, and run before
polycubes.py): py -3 -X utf8 -I avalanche/unit-shapes/catalogue/gen_boxes.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling modules, also under -I
from polycubes import orientations, turning_moves  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DONE = {(1, 1, 2), (1, 1, 3), (1, 1, 4), (1, 1, 5), (1, 1, 6), (1, 1, 8), (1, 2, 2), (1, 3, 3), (1, 2, 3),
        (1, 2, 4), (2, 2, 3)}


def main():
    out, info = {}, ["name\tsites\tlabels\tbox\thole\ttiles\tcells"]
    print("name       sites labels hole")
    for n in range(2, 7):
        for m in range(1, min(n, 4) + 1):
            for l in range(1, m + 1):
                if (l, m, n) in DONE or l == m == n or l * m * n > 36:
                    continue
                cells = [[x, y, z] for x in range(l) for y in range(m) for z in range(n)]
                q = len(orientations(cells))
                _, vmin = turning_moves(orientations(cells))
                name = f"box{l}{m}{n}"
                out[name] = cells
                cstr = ";".join(",".join(map(str, c)) for c in cells)
                info.append(f"{name}\t{len(cells)}\t{q}\t{n}x{m}x{l}\t{vmin}/{len(cells)}\tTrue\t{cstr}")
                print(f"{name:10s} {len(cells):5d} {q:6d} {vmin}/{len(cells)}={vmin/len(cells):.3f}")
    with open(os.path.join(HERE, "pieces3.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(HERE, "pieces3-info.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(info) + "\n")
    print(f"{len(out)} boxes written to pieces3.json")


if __name__ == "__main__":
    main()
