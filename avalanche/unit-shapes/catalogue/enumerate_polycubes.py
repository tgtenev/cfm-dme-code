"""Enumerate the polycubes up to a given size, distinct under proper rotations only (a piece and its
mirror image count separately, since a physical unit cannot be reflected), and report for each size
the count, the label counts (24 / order of the piece's rotation group) and how many of them the
catalogue has run. Grows each polycube by one adjacent site and keeps a canonical form. gen_catalogue2.py
uses its functions to generate the pieces of five and six sites.

Usage (from the repository root; seconds): py -3 -X utf8 -I avalanche/unit-shapes/catalogue/enumerate_polycubes.py [maxsize]
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling modules, also under -I
from polycubes import PIECES, rotations  # noqa: E402

ROTS = rotations()


def canon(cells):
    pts = np.array(sorted(cells), dtype=int)
    best = None
    for m in ROTS:
        img = pts @ m.T
        img = img - img.min(axis=0)
        key = tuple(sorted(map(tuple, img)))
        if best is None or key < best:
            best = key
    return best


def grow(polys):
    out = set()
    for p in polys:
        cells = set(p)
        for c in cells:
            for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                n = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
                if n not in cells:
                    out.add(canon(cells | {n}))
    return out


def n_labels(p):
    pts = np.array(p, dtype=int)
    seen = set()
    for m in ROTS:
        img = pts @ m.T
        img = img - img.min(axis=0)
        seen.add(tuple(sorted(map(tuple, img))))
    return len(seen)


def main():
    maxsize = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    ran = {canon(c): name for name, c in PIECES.items()}
    polys = {canon([(0, 0, 0)])}
    for size in range(1, maxsize + 1):
        if size > 1:
            polys = grow(polys)
        labels = sorted(n_labels(p) for p in polys)
        from collections import Counter
        cnt = Counter(labels)
        done = [ran[p] for p in polys if p in ran]
        print(f"size {size}: {len(polys)} pieces; label counts {dict(sorted(cnt.items()))}; "
              f"run so far: {len(done)} ({', '.join(sorted(done))})")


if __name__ == "__main__":
    main()
