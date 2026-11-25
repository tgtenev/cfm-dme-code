"""Does a piece tile the periodic lattice by translation alone, with every copy in one orientation?
That is what the aligned tiling of a candidate unit shape needs: all units sharing one label and covering
every site. Backtracking exact cover on a periodic box whose site count the piece divides: the first
empty site (lexicographic) must be covered, by any one of the piece's cells. gen_catalogue2.py keeps only
the pieces that pass.

Usage (from the repository root; seconds): py -3 -X utf8 -I avalanche/unit-shapes/catalogue/tilingtest.py [piece ...]
(default: every piece in polycubes.PIECES)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling modules, also under -I
from polycubes import PIECES, orientations  # noqa: E402


def tiles_by_translation(cells, n):
    cells = [tuple(c) for c in cells]
    m = len(cells)
    N = n ** 3
    if N % m:
        return None
    occ = np.zeros((n, n, n), dtype=bool)
    sys.setrecursionlimit(10000)

    def first_empty():
        idx = np.argmin(occ)          # first False in C order; if none, occ.all()
        if occ.flat[idx]:
            return None
        return np.unravel_index(idx, occ.shape)

    def rec():
        s = first_empty()
        if s is None:
            return True
        for c in cells:
            off = (s[0] - c[0], s[1] - c[1], s[2] - c[2])
            sites = [((off[0] + d[0]) % n, (off[1] + d[1]) % n, (off[2] + d[2]) % n) for d in cells]
            if any(occ[t] for t in sites):
                continue
            for t in sites:
                occ[t] = True
            if rec():
                return True
            for t in sites:
                occ[t] = False
        return False

    return rec()


def main():
    names = sys.argv[1:] or list(PIECES)
    for name in names:
        cells = PIECES[name]
        m = len(cells)
        n = next(k for k in (4, 6, 8, 10, 12) if (k ** 3) % m == 0)
        q = len(orientations(cells))
        res = tiles_by_translation(cells, n)
        print(f"{name:10s} sites {m:2d} labels {q:2d}  box {n}^3  tiles by translation: {res}")


if __name__ == "__main__":
    main()
