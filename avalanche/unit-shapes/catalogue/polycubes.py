"""The raw packing error and the turning hole of any polycube unit shape (Paper II, Sec. 5.2 and Appendix C).

latticebricks.py generalized from rods to any polycube (a three-dimensional Tetris piece): a connected set
of lattice cells laid on the sites of a periodic simple cubic lattice. The piece's labels are its distinct
orientations under the 24 rotations of the cube (translations removed), q of them. The aligned tiling by
one orientation covers every site and is the densest state. A roughening event lays each unit at a random
site in a random orientation, and it never moves again: random sequential addition until jamming, sampled
exactly (rejection sampling, then enumeration of the remaining feasible placements until none remains).

Outputs per piece: q, theta_J, delta_0 = (1 - theta_J)/theta_J (the raw packing error), and the turning hole
v*/v (the activation constant c_a), the smallest number of fresh cells a piece needs to re-lay itself in a
different orientation that shares at least one cell with its present footprint, divided by its cell count
(for the rod this is (L - 1)/L). Also the cage test: the fraction of pieces that could re-lay in another
orientation into existing vacancies (f_turn), and the mean number of vacant sites touching a piece
(vac_touch). The row's columns are piece, cells, q, size, seed, pieces, theta_J, delta0, vstar_over_v,
f_turn, vac_touch and t_s (run time, s).

Paper II results it supports: the raw packing errors and turning holes of the catalogue of unit shapes
screened in Sec. 5.2 (59 distinct shapes with a crossing, of up to 36 sites), read by build_results.py.

Commands (from the repository root; 120 runs, 3.6 h of recorded run time, 14 s to 6 min each; after
gen_catalogue2.py and gen_boxes.py, before the avalanche runs, which take delta_0 and c_a from these rows):
    c=avalanche/unit-shapes/catalogue
    for f in $c/rsa-*-s1.tsv; do p=${f#$c/rsa-}; p=${p%-s1.tsv}
      case $p in p5-*|p6-*) cat="--catalog $c/pieces2.json";; box*) cat="--catalog $c/pieces3.json";; *) cat="";; esac
      for s in 1 2; do rm -f $c/rsa-$p-s$s.tsv
        py -3 -X utf8 -I $c/polycubes.py --piece $p --size 48 --seed $s $cat --out $c/rsa-$p-s$s.tsv --header; done; done
(--out appends, hence the rm). The rods come from latticebricks.py instead.

Usage: py -3 polycubes.py --piece rod4 --size 48 --seed 1 [--catalog FILE.json] [--out file.tsv] [--header]
       py -3 polycubes.py --list [--catalog FILE.json]
"""
import argparse
import itertools
import time

import numpy as np

PIECES = {
    # name: cells (x, y, z)
    "rod2": [(0, 0, 0), (1, 0, 0)],
    "rod3": [(0, 0, 0), (1, 0, 0), (2, 0, 0)],
    "rod4": [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
    "rod5": [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)],
    "plate22": [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
    "plate23": [(x, y, 0) for x in range(2) for y in range(3)],
    "plate33": [(x, y, 0) for x in range(3) for y in range(3)],
    "brick123": [(0, x, y) for x in range(2) for y in range(3)],
    "brick124": [(0, x, y) for x in range(2) for y in range(4)],
    "brick223": [(x, y, z) for x in range(2) for y in range(2) for z in range(3)],
    "cube222": [(x, y, z) for x in range(2) for y in range(2) for z in range(2)],
    "Ltro": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
    "Ltet": [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0)],
    "Ttet": [(0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0)],
    "Stet": [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],
    "corner": [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
    "screw": [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)],
}


def rotations():
    """The 24 proper rotation matrices of the cube."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            m = np.zeros((3, 3), dtype=int)
            for i, p in enumerate(perm):
                m[i, p] = signs[i]
            if round(np.linalg.det(m)) == 1:
                mats.append(m)
    return mats


def orientations(cells):
    """Distinct orientations of a piece under proper rotations, each normalized so that its
    lexicographically smallest cell is the origin; returned as a list of offset arrays."""
    seen = {}
    pts = np.array(cells, dtype=int)
    for m in rotations():
        img = pts @ m.T
        img = img - img.min(axis=0)
        key = tuple(sorted(map(tuple, img)))
        if key not in seen:
            seen[key] = np.array(key, dtype=int)
    return list(seen.values())


def free_placements(occ, oris):
    """free[o] (q, n, n, n): placement of orientation o with its origin cell at site s is
    feasible iff every cell s + offset is free (periodic)."""
    free = np.empty((len(oris),) + occ.shape, dtype=bool)
    for o, offs in enumerate(oris):
        f = np.ones_like(occ)
        for d in offs:
            f &= ~np.roll(occ, shift=tuple(-d), axis=(0, 1, 2))
        free[o] = f
    return free


def place(occ, heads, o, s, oris):
    n = occ.shape[0]
    for d in oris[o]:
        occ[tuple((s + d) % n)] = True
    heads[tuple(s)] = o + 1


def rsa_jam(n, oris, rng, streak=3000):
    occ = np.zeros((n, n, n), dtype=bool)
    heads = np.zeros((n, n, n), dtype=np.int16)
    q = len(oris)
    placed = 0
    fails = 0
    while fails < streak:
        o = int(rng.integers(q))
        s = rng.integers(0, n, 3)
        ok = True
        for d in oris[o]:
            if occ[tuple((s + d) % n)]:
                ok = False
                break
        if ok:
            place(occ, heads, o, s, oris)
            placed += 1
            fails = 0
        else:
            fails += 1
    while True:
        free = free_placements(occ, oris)
        cand = np.argwhere(free)
        if len(cand) == 0:
            break
        c = cand[rng.integers(len(cand))]
        place(occ, heads, int(c[0]), c[1:], oris)
        placed += 1
    return occ, heads, placed


def turning_moves(oris):
    """For each orientation o: the list of (o2, t) with o2 != o and translation t such that the
    re-laid piece shares at least one cell with the present one, with the number of fresh cells
    each needs. Also the minimum fresh-cell count over all of them."""
    sets = [set(map(tuple, offs)) for offs in oris]
    moves = []
    vmin = None
    for o, A in enumerate(sets):
        lst = []
        for o2, B in enumerate(sets):
            if o2 == o:
                continue
            ts = set()
            for a in A:
                for b in B:
                    ts.add(tuple(np.array(a) - np.array(b)))
            for t in ts:
                Bt = {tuple(np.array(b) + np.array(t)) for b in B}
                fresh = len(Bt - A)
                lst.append((o2, t, fresh))
                vmin = fresh if vmin is None else min(vmin, fresh)
        moves.append(lst)
    return moves, vmin


def cage_stats(occ, heads, oris, moves):
    n = occ.shape[0]
    hs = np.argwhere(heads > 0)
    can_turn = 0
    vac_touch = 0.0
    for h in hs:
        o = int(heads[tuple(h)]) - 1
        own = {tuple((h + d) % n) for d in oris[o]}
        touch = set()
        for site in own:
            for e in range(3):
                for sgn in (-1, 1):
                    t = list(site)
                    t[e] = (t[e] + sgn) % n
                    t = tuple(t)
                    if t not in own and not occ[t]:
                        touch.add(t)
        vac_touch += len(touch)
        turned = False
        for o2, t, fresh in moves[o]:
            ok = True
            for d in oris[o2]:
                c = tuple((h + np.array(t) + d) % n)
                if c not in own and occ[c]:
                    ok = False
                    break
            if ok:
                turned = True
                break
        can_turn += turned
    m = len(hs)
    return can_turn / m, vac_touch / m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piece", default="rod4")
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--header", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--catalog", default="", help="JSON file of extra pieces, name -> list of cells")
    a = ap.parse_args()
    if a.catalog:
        import json
        with open(a.catalog, encoding="utf-8") as f:
            for name, cells in json.load(f).items():
                PIECES[name] = [tuple(c) for c in cells]
    if a.list:
        for name, cells in PIECES.items():
            oris = orientations(cells)
            _, vmin = turning_moves(oris)
            hole = "no label" if vmin is None else f"{vmin}/{len(cells)} = {vmin / len(cells):.3f}"
            print(f"{name:10s} cells {len(cells):2d}  orientations {len(oris):2d}  turning hole {hole}")
        return
    cells = PIECES[a.piece]
    oris = orientations(cells)
    moves, vmin = turning_moves(oris)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    occ, heads, placed = rsa_jam(a.size, oris, rng)
    theta = occ.mean()
    f_turn, vac = cage_stats(occ, heads, oris, moves)
    t1 = time.time()
    row = {
        "piece": a.piece, "cells": len(cells), "q": len(oris), "size": a.size, "seed": a.seed,
        "pieces": placed, "theta_J": round(float(theta), 5), "delta0": round((1 - theta) / theta, 4),
        "vstar_over_v": round(vmin / len(cells), 4), "f_turn": round(f_turn, 4),
        "vac_touch": round(vac, 3), "t_s": round(t1 - t0, 1),
    }
    header = "\t".join(row.keys())
    line = "\t".join(str(v) for v in row.values())
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            if a.header:
                f.write(header + "\n")
            f.write(line + "\n")
    print(header)
    print(line)


if __name__ == "__main__":
    main()
