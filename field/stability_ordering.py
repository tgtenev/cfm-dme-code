"""The ordering point of renewal domains that are layers, for the stability check of the renewal's noise.

A compact domain has twelve neighbours, and its folds order at 9.75 misfit energies (field/stability_threshold_mc.py
with the fcc lattice). A domain that is a layer, a square tile much wider than it is thick, has four neighbours across
its edges and, on each face, one neighbour where the layers stack in register or four where they stack out of
register. This script locates the ordering point of Ising folds on those two arrangements, with every pair coupled by
one well J, the per-pair convention. It then repeats the out-of-register case with each coupling in proportion to the
area that the two tiles share: J/4 for each of the eight face neighbours and J/a across an edge, with a = 100 the
width of a tile over its thickness. For tiles in register with couplings by area, which are chains coupled weakly, it
gives the chain mean-field estimate, 4 (J/a) chi(T) = 1 with chi = exp(2J/T)/T.

Algorithm: Wolff single-cluster updates with a bond probability 1 - exp(-2 J_ij / T), and the crossing of the Binder
cumulant U4 = 1 - <m^4>/(3 <m^2>^2) for three sizes. Each run has a fixed seed, so the output repeats exactly.

Supports Paper II section 5.3 and Table 1: 6 and 12 neighbours; ordering points of 4.50 and 9.79 per pair and 1.63
with couplings by shared area; margins of 5.7 and 19.5 against 9.75, from 55 to 190 misfit energies per domain.

Inputs: none. Output: field/stability_ordering-results.txt.
Run from the repository root (about 4 minutes; the script writes its output itself):
    python field/stability_ordering.py
"""
import math
import pathlib
import random
import time


def lattice(L, offset, j_face, j_edge):
    """Tiles (x, y, z) on an L x L x L periodic block. Returns for each tile its neighbours and their couplings."""
    def site(x, y, z):
        return ((x % L) * L + (y % L)) * L + (z % L)
    nb, jj = [], []
    for x in range(L):
        for y in range(L):
            for z in range(L):
                n = [site(x + 1, y, z), site(x - 1, y, z), site(x, y + 1, z), site(x, y - 1, z)]
                j = [j_edge] * 4
                for dz in (1, -1):
                    if offset:
                        # layers alternate between two positions half a tile apart, so a tile faces four tiles
                        s = 1 if z % 2 == 0 else -1
                        n += [site(x, y, z + dz), site(x - s, y, z + dz), site(x, y - s, z + dz), site(x - s, y - s, z + dz)]
                        j += [j_face] * 4
                    else:
                        n.append(site(x, y, z + dz))
                        j.append(j_face)
                nb.append(tuple(n))
                jj.append(tuple(j))
    return nb, jj


def wolff(nb, jj, T, n_therm, n_meas, seed):
    rng = random.Random(seed)
    N = len(nb)
    s = [1] * N
    padd = [tuple(1.0 - math.exp(-2.0 * j / T) for j in row) for row in jj]
    M = N
    rand = rng.random
    blocks2, blocks4 = [], []
    per = n_meas // 10
    acc2 = acc4 = 0.0
    for it in range(n_therm + n_meas):
        i0 = int(rand() * N)
        sp = s[i0]
        s[i0] = -sp
        todo = [i0]
        c = 1
        while todo:
            i = todo.pop()
            pi = padd[i]
            for k, jn in enumerate(nb[i]):
                if s[jn] == sp and rand() < pi[k]:
                    s[jn] = -sp
                    todo.append(jn)
                    c += 1
        M -= 2 * sp * c
        if it >= n_therm:
            m2 = (M / N) ** 2
            acc2 += m2
            acc4 += m2 * m2
            if (it - n_therm + 1) % per == 0:
                blocks2.append(acc2 / per)
                blocks4.append(acc4 / per)
                acc2 = acc4 = 0.0
    m2 = sum(blocks2) / len(blocks2)
    m4 = sum(blocks4) / len(blocks4)
    ub = [1.0 - b4 / (3.0 * b2 * b2) for b2, b4 in zip(blocks2, blocks4)]
    mu = sum(ub) / len(ub)
    err = math.sqrt(sum((u - mu) ** 2 for u in ub) / (len(ub) - 1) / len(ub))
    return 1.0 - m4 / (3.0 * m2 * m2), err


def crossing(Ts, ua, ub):
    d = [b - a for a, b in zip(ua, ub)]
    for i in range(len(Ts) - 1):
        if d[i] * d[i + 1] < 0:
            return Ts[i] + (Ts[i + 1] - Ts[i]) * d[i] / (d[i] - d[i + 1])
    return float('nan')


CASES = [
    ("in register, every pair one well: 4 edge and 2 face neighbours, the simple cubic lattice (published 4.5115)",
     False, 1.0, 1.0, [4.45, 4.49, 4.53, 4.57]),
    ("out of register, every pair one well: 4 edge and 8 face neighbours, the face-centred cubic lattice (published 9.794)",
     True, 1.0, 1.0, [9.65, 9.75, 9.85, 9.95]),
    ("out of register, couplings by shared area: 8 face neighbours at J/4 and 4 edge neighbours at J/100",
     True, 0.25, 0.01, [1.54, 1.58, 1.62, 1.66]),
]
SIZES = [8, 12, 16]
N_MEAS = {8: 60000, 12: 40000, 16: 30000}

out = []


def say(s=""):
    print(s, flush=True)
    out.append(s)


say("stability_ordering.py: ordering point of Ising folds on tiles stacked as layers, in units of the pair well J")
say(f"sizes {SIZES} tiles on a side; thermalisation 3000 clusters; measured clusters per size {N_MEAS}")
t0 = time.time()
for name, offset, j_face, j_edge, Ts in CASES:
    say()
    say(name)
    table = {}
    for L in SIZES:
        nb, jj = lattice(L, offset, j_face, j_edge)
        assert all(len(set(row)) == len(row) for row in nb)
        table[L] = [wolff(nb, jj, T, 3000, N_MEAS[L], seed=1000 * L + int(1000 * T)) for T in Ts]
        print(f"   size {L} done, t = {time.time() - t0:.0f} s", flush=True)
    say("   T       " + "  ".join(f"U4(size {L})       " for L in SIZES))
    for i, T in enumerate(Ts):
        say(f"   {T:6.3f}  " + "  ".join(f"{table[L][i][0]:.4f} +- {table[L][i][1]:.4f}" for L in SIZES))
    cs = [crossing(Ts, [r[0] for r in table[a]], [r[0] for r in table[b]]) for a, b in ((8, 12), (12, 16), (8, 16))]
    good = [x for x in cs if x == x]
    say("   crossings 8/12, 12/16, 8/16: " + ", ".join(f"{x:.4f}" for x in cs))
    if good:
        say(f"   ordering point: {sum(good) / len(good):.3f} J, spread {max(good) - min(good):.3f}")

say()
say("in register, couplings by shared area: chains along the stack coupled by J/a across four edges; chain mean field, "
    "4 (J/a) exp(2J/T) / T = 1")
for a in (100, 300, 1000):
    lo, hi = 0.05, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 4.0 / a * math.exp(2.0 / mid) / mid > 1.0:
            lo = mid
        else:
            hi = mid
    say(f"   a = {a}: ordering point about {0.5 * (lo + hi):.2f} J")

say()
say("the check of the renewal's noise: standing agitation of 55 to 190 misfit energies against each ordering point")
for label, t in (("12 neighbours, one well each", 9.75), ("6 neighbours, one well each", 4.49)):
    say(f"   {label}: threshold {t}: margin {55.4 / t:.1f} at the slow edge and {190.0 / t:.1f} at the fast edge")
say(f"run time {time.time() - t0:.0f} s")
pathlib.Path(__file__).with_name("stability_ordering-results.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
