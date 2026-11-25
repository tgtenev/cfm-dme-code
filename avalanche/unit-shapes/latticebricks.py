"""The raw packing error and the turning hole of the rod-shaped units (Paper II, Sec. 5.1 and Appendix C).

Template units as 1 x 1 x L rods laid on the sites of a periodic simple cubic lattice, with three labels
(the rod's axis along x, y or z). The aligned tiling, every rod parallel, covers every site and is the
densest state. A roughening event lays its units at random: each takes a random site and a random label
and never moves again. That is random sequential addition until jamming, and the vacancies it leaves are
the free volume. The raw packing error of the unit, empty sites per occupied site, is therefore

    delta_0 = (1 - theta_J)/theta_J,

with theta_J the jamming coverage. The hole a rod needs to turn to another label is the volume it sweeps
beyond its own body, (L - 1) sites of its L, so v*/v_unit = (L - 1)/L, and the activation constant of the
Cohen-Turnbull form is c_a = (L - 1)/L (column vstar_over_v).

The sampling is exact random sequential addition: a placement (site, label) is drawn uniformly among all
placements and accepted if its L sites are free; once a long run of draws fails, the remaining feasible
placements are enumerated and placed one by one, re-enumerating after each, until none remains (true
jamming, not an attempt budget). --protocol changes the laying rule (the check of Paper II's Table 1 row
"laying rule of a roughening event"): rsa is the bulk random addition above; front fills the lattice layer
by layer, each layer to jamming; front1 also goes layer by layer but gives each site of a layer one draw.

Also measured in the jammed state: the fraction of rods that could turn to another label into the existing
vacancies with no expansion (f_turn), and the mean number of vacant sites touching a rod (vac_touch).

Paper II results it supports: for the rod of four sites, delta_0 = 0.282 with 22 percent of the sites vacant
(theta_J = 0.780) and c_a = 3/4 (lattice-L4-s{1,2,3}.tsv: 0.2817, 0.2809, 0.2828); and bulk random addition
as the laying rule that keeps the divisor in the target window (checks/lattice-protocols.tsv: the bulk rule
gives 0.2820 and 0.2818 on 32^3 and 64^3, a filling front 0.015, a one-draw front 0.49, whose avalanche runs
give a divisor of 4.236).

Commands (from the repository root; --out appends, so delete the old file first; 25 to 50 s per packing at
48^3; run before the avalanche runs, which take delta_0 from these files):
    for L in 2 3 4 5 6 8; do for s in 1 2 3; do
      py -3 -X utf8 -I avalanche/unit-shapes/latticebricks.py --L $L --size 48 --seed $s \\
        --out avalanche/unit-shapes/lattice-L$L-s$s.tsv --header; done; done
and the laying-rule check, --header on its first row only (about 4 min in all):
    f=avalanche/unit-shapes/checks/lattice-protocols.tsv; h=--header
    for a in "4 48 1 front1" "3 48 1 front1" "5 48 1 front1" "6 48 1 front1" "4 48 2 front1" \\
             "4 48 1 front" "4 32 1 rsa" "4 64 1 rsa"; do set -- $a
      py -3 -X utf8 -I avalanche/unit-shapes/latticebricks.py --L $1 --size $2 --seed $3 --protocol $4 \\
        --out $f $h; h=; done
Each run prints its row; the row's columns are L, size, seed, protocol, rods, theta_J, delta0, vstar_over_v,
f_turn, vac_touch and t_s (run time, s). The committed lattice-L*-s*.tsv, all of protocol rsa, carry no protocol
column; a rerun adds it and reproduces every other column but t_s.
"""
import argparse
import time

import numpy as np


def free_placements(occ, L):
    """Boolean arrays free[d] (3, n, n, n): placement of a rod with axis d at site s is
    feasible iff the L sites s, s+e_d, ..., s+(L-1)e_d are all free (periodic)."""
    free = np.empty((3,) + occ.shape, dtype=bool)
    for d in range(3):
        f = ~occ
        for k in range(1, L):
            f &= ~np.roll(occ, -k, axis=d)
        free[d] = f
    return free


def place(occ, labels, d, s, L):
    n = occ.shape[0]
    idx = [list(s) for _ in range(L)]
    for k in range(L):
        idx[k][d] = (s[d] + k) % n
    for k in range(L):
        occ[tuple(idx[k])] = True
    labels[tuple(s)] = d + 1          # the rod's head site records its label (1..3)


def rsa_jam(n, L, rng, streak=3000):
    occ = np.zeros((n, n, n), dtype=bool)
    labels = np.zeros((n, n, n), dtype=np.int8)
    placed = 0
    fails = 0
    N = n ** 3
    # phase 1: rejection sampling, exact RSA
    while fails < streak:
        d = int(rng.integers(3))
        s = rng.integers(0, n, 3)
        ok = True
        for k in range(L):
            t = s.copy()
            t[d] = (s[d] + k) % n
            if occ[tuple(t)]:
                ok = False
                break
        if ok:
            place(occ, labels, d, s, L)
            placed += 1
            fails = 0
        else:
            fails += 1
    # phase 2: enumerate the remaining feasible placements and fill them one by one
    while True:
        free = free_placements(occ, L)
        cand = np.argwhere(free)
        if len(cand) == 0:
            break
        c = cand[rng.integers(len(cand))]
        place(occ, labels, int(c[0]), c[1:], L)
        placed += 1
    return occ, labels, placed


def front_jam(n, L, rng):
    """A growth front instead of bulk laying: the lattice fills layer by layer along z. At each
    layer every feasible placement whose lowest site lies in that layer (in-plane rods in the
    layer, vertical rods standing on it) is enumerated and filled at random until none is left,
    then the front moves up one layer. Vacancies a layer leaves cannot be filled later from
    above except by vertical rods that happen to stand on them, so the front traps less than
    bulk random sequential addition does only where in-plane rods block the layer."""
    occ = np.zeros((n, n, n), dtype=bool)
    labels = np.zeros((n, n, n), dtype=np.int8)
    placed = 0
    for z in range(n):
        while True:
            free = free_placements(occ, L)
            layer = free[:, :, :, z]                     # placements with base site in layer z
            cand = np.argwhere(layer)                    # (d, x, y)
            if len(cand) == 0:
                break
            c = cand[rng.integers(len(cand))]
            place(occ, labels, int(c[0]), np.array([c[1], c[2], z]), L)
            placed += 1
    return occ, labels, placed


def front_one_draw(n, L, rng):
    """One draw per front site: the lattice fills layer by layer along z; the free sites of a layer
    are visited in random order, each draws one label at random, and the rod is laid if its L sites
    are free. A draw that does not fit leaves the site empty for good (a dead site: the unit was
    laid once with its label frozen, and a misfit is a vacancy, not a retry). A vertical draw always
    fits, since nothing has been laid above the front yet; an in-plane draw needs L - 1 free
    sites along its axis in the layer."""
    occ = np.zeros((n, n, n), dtype=bool)
    dead = np.zeros((n, n, n), dtype=bool)
    labels = np.zeros((n, n, n), dtype=np.int8)
    placed = 0
    for z in range(n):
        order = rng.permutation(n * n)
        for o in order:
            x, y = divmod(int(o), n)
            if occ[x, y, z] or dead[x, y, z]:
                continue
            d = int(rng.integers(3))
            s = np.array([x, y, z])
            ok = True
            for k in range(L):
                t = s.copy()
                t[d] = (s[d] + k) % n
                if occ[tuple(t)] or dead[tuple(t)]:
                    ok = False
                    break
            if ok:
                place(occ, labels, d, s, L)
                placed += 1
            else:
                dead[x, y, z] = True
    return occ, labels, placed


def cage_stats(occ, labels, L):
    """Fraction of rods that could turn to another label into existing vacancies, and the
    mean number of vacant sites in contact with a rod."""
    n = occ.shape[0]
    heads = np.argwhere(labels > 0)
    can_turn = 0
    vac_touch = 0.0
    for h in heads:
        d = int(labels[tuple(h)]) - 1
        # the rod's own sites
        own = []
        for k in range(L):
            t = h.copy()
            t[d] = (h[d] + k) % n
            own.append(tuple(t))
        own_set = set(own)
        # vacancies touching the rod
        touch = set()
        for site in own:
            for e in range(3):
                for sgn in (-1, 1):
                    t = list(site)
                    t[e] = (t[e] + sgn) % n
                    t = tuple(t)
                    if t not in own_set and not occ[t]:
                        touch.add(t)
        vac_touch += len(touch)
        # could the rod turn? try every other axis and every shift along it that keeps the
        # head site inside the rod's current footprint
        turned = False
        for d2 in range(3):
            if d2 == d or turned:
                continue
            for shift in range(-(L - 1), 1):
                ok = True
                for k in range(L):
                    t = h.copy()
                    t[d2] = (h[d2] + shift + k) % n
                    tt = tuple(t)
                    if tt not in own_set and occ[tt]:
                        ok = False
                        break
                if ok:
                    turned = True
                    break
        can_turn += turned
    m = len(heads)
    return can_turn / m, vac_touch / m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--protocol", choices=["rsa", "front", "front1"], default="rsa")
    ap.add_argument("--out", default="")
    ap.add_argument("--header", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    if a.protocol == "front":
        occ, labels, placed = front_jam(a.size, a.L, rng)
    elif a.protocol == "front1":
        occ, labels, placed = front_one_draw(a.size, a.L, rng)
    else:
        occ, labels, placed = rsa_jam(a.size, a.L, rng)
    theta = occ.mean()
    f_turn, vac = cage_stats(occ, labels, a.L)
    t1 = time.time()
    row = {
        "L": a.L, "size": a.size, "seed": a.seed, "protocol": a.protocol, "rods": placed,
        "theta_J": round(float(theta), 5), "delta0": round((1 - theta) / theta, 4),
        "vstar_over_v": round((a.L - 1) / a.L, 4),
        "f_turn": round(f_turn, 4), "vac_touch": round(vac, 3), "t_s": round(t1 - t0, 1),
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
