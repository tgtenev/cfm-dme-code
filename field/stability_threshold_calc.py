"""The band of the ordering point of the folds of renewal domains, for the stability check of the renewal's noise.

The ordering point T* is the noise temperature, in units of the pair well J, below which the folds of neighbouring
domains order. Its central value is the Monte Carlo crossing for Ising folds on the face-centred cubic lattice, twelve
neighbours, read from field/stability_threshold_mc-fcc0.txt; the simple cubic crossing, six neighbours, is read from
field/stability_threshold_mc-sc0.txt. The script then varies the fold's symmetry (Ising, XY, Heisenberg), the number
of neighbours, the fluctuation corrections (three-dimensional values against mean field) and a tail of the coupling
(lattice sums of 1/r^p, validated against the published Lennard-Jones sums), and gives the band that these span.

Supports Paper II section 5.3 and Table 1: the ordering point of about 9.75 for twelve neighbours and about 4.5 for
six, and the band of 4.40 to 15.50 across the fold's symmetry, the number of neighbours and the fluctuation
corrections, against which kinetics/kinetic_windows.py gives the margins of 5.7 and 3.6.

Inputs: field/stability_threshold_mc-sc0.txt and field/stability_threshold_mc-fcc0.txt (deposited; written by
field/stability_threshold_mc.py). If one is missing, the script says so and uses the published value instead.
Output: field/stability_threshold_calc-results.txt.
Run from the repository root (a few seconds):
    python field/stability_threshold_calc.py > field/stability_threshold_calc-results.txt
"""
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

out = []


def p(s=''):
    out.append(s)


p('stability_threshold_calc.py results')
p('=' * 78)

# ---------------------------------------------------------------- 1. lattice sums
p('1. Lattice sums S_p = sum over j of (r_nn / r_j)^p, distances in nearest-neighbour units')


def lattice_points(kind, R):
    rng = np.arange(-R, R + 1)
    g = np.array(np.meshgrid(rng, rng, rng, indexing='ij')).reshape(3, -1).T
    if kind == 'sc':
        pts = g.astype(float)
        dens = 1.0
    elif kind == 'fcc':
        pts = g[(g.sum(1) % 2) == 0].astype(float) / math.sqrt(2.0)   # nn distance sqrt(2) on the grid
        dens = math.sqrt(2.0)
    elif kind == 'bcc':
        allsame = ((g % 2) == (g[:, :1] % 2)).all(1)
        pts = g[allsame].astype(float) / math.sqrt(3.0)                # nn distance sqrt(3) on the grid
        dens = 3.0 * math.sqrt(3.0) / 4.0
    r = np.sqrt((pts ** 2).sum(1))
    return r[r > 0], dens


def lattice_sum(kind, pw, R=60):
    r, dens = lattice_points(kind, R)
    rcut = R / (math.sqrt(2.0) if kind == 'fcc' else (math.sqrt(3.0) if kind == 'bcc' else 1.0)) * 0.95
    inner = r[r <= rcut]
    tail = 4.0 * math.pi * dens * rcut ** (3 - pw) / (pw - 3)
    return float(np.sum(inner ** (-pw))) + tail


published = {('sc', 6): 8.40192, ('bcc', 6): 12.2533, ('fcc', 6): 14.45392,
             ('sc', 12): 6.20215, ('bcc', 12): 9.11418, ('fcc', 12): 12.13188}
S = {}
for kind in ('sc', 'bcc', 'fcc'):
    for pw in (12, 6, 5, 4):
        S[(kind, pw)] = lattice_sum(kind, pw)
for kind, zc in (('sc', 6), ('bcc', 8), ('fcc', 12)):
    row = '  '.join(f'p={pw}: {S[(kind, pw)]:8.4f}' for pw in (12, 6, 5, 4))
    p(f'   {kind} (z = {zc}): {row}')
p('   validation against the published Lennard-Jones sums (Jones and Ingham 1925):')
for (kind, pw), val in published.items():
    p(f'      {kind} p={pw}: computed {S[(kind, pw)]:.5f}, published {val:.5f}, difference {S[(kind, pw)] - val:+.5f}')
p()

# ---------------------------------------------------------------- 2. Monte Carlo crossings
p('2. Ordering temperatures in units of the pair coupling J (kT_c / J)')


def read_mc(name):
    """Return (T_c, spread) from a results file of stability_threshold_mc.py, or (None, None) if it is missing."""
    fp = os.path.join(HERE, name)
    if not os.path.exists(fp):
        return None, None
    txt = open(fp, encoding='utf-8').read()
    m = re.search(r'T_c estimate \(mean of crossings\): ([0-9.]+), spread ([0-9.]+)', txt)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


lit = {  # three-dimensional values from the literature
    ('ising', 'sc'): 4.5115, ('ising', 'bcc'): 6.3548, ('ising', 'fcc'): 9.7944,
    ('xy', 'sc'): 2.2016, ('heis', 'sc'): 1.4430, ('heis', 'bcc'): 2.0542, ('heis', 'fcc'): 3.1784,
    ('nematic', 'sc'): 1.1232,
}
mc_sc, mc_sc_sp = read_mc('stability_threshold_mc-sc0.txt')
mc_fcc, mc_fcc_sp = read_mc('stability_threshold_mc-fcc0.txt')
p(f'   Monte Carlo (stability_threshold_mc.py), Ising: sc {mc_sc} (spread {mc_sc_sp}), fcc {mc_fcc} (spread {mc_fcc_sp})')
p(f'   published Ising: sc {lit[("ising", "sc")]}, fcc {lit[("ising", "fcc")]}')
if mc_sc:
    p(f'      Monte Carlo minus published: sc {mc_sc - lit[("ising", "sc")]:+.4f} ({100 * (mc_sc / lit[("ising", "sc")] - 1):+.2f}%), fcc {mc_fcc - lit[("ising", "fcc")]:+.4f} ({100 * (mc_fcc / lit[("ising", "fcc")] - 1):+.2f}%)' if mc_fcc else '')
TC_FCC = mc_fcc if mc_fcc else lit[('ising', 'fcc')]
p(f'   value used for the face-centred Ising threshold: {TC_FCC:.3f}')
p()
p('   Mean-field values kT_c/J = z_eff / m, m = number of components (Ising 1, XY 2, Heisenberg 3),')
p('   and the ratio of the three-dimensional value to the mean-field value where it is known:')
zlist = {'sc': 6, 'bcc': 8, 'fcc': 12}
for sym, mcomp in (('ising', 1), ('xy', 2), ('heis', 3)):
    for kind in ('sc', 'bcc', 'fcc'):
        mf = zlist[kind] / mcomp
        ex = lit.get((sym, kind))
        if sym == 'ising' and kind == 'fcc' and mc_fcc:
            ex = mc_fcc
        if sym == 'ising' and kind == 'sc' and mc_sc:
            ex = mc_sc
        ratio = f'{ex / mf:.3f}' if ex else '  -  '
        exs = f'{ex:.3f}' if ex else '  -  '
        p(f'      {sym:5s} {kind}: mean field {mf:6.3f}, three-dimensional {exs}, ratio {ratio}')
p(f'      nematic (Lebwohl-Lasher, sc): mean field {0.22015 * 6:.3f}, three-dimensional {lit[("nematic", "sc")]}, ratio {lit[("nematic", "sc")] / (0.22015 * 6):.3f}')
p()

# ---------------------------------------------------------------- 3. the threshold and its band
p('3. The threshold T* on Theta/J for staying disordered, J the pair well (J <= u_mis)')
T_central = TC_FCC
variants = [
    ('central: Ising, z = 12 (close packed), three-dimensional fluctuations, per-pair bound', T_central),
    ('XY symmetry (sign and phase), z = 12, fluctuation ratio of sc XY', 6.0 * lit[('xy', 'sc')] / 3.0),
    ('Heisenberg symmetry, z = 12', lit[('heis', 'fcc')]),
    ('Ising, z = 6 (sc)', mc_sc if mc_sc else lit[('ising', 'sc')]),
    ('Ising, z = 8 (bcc)', lit[('ising', 'bcc')]),
    ('Ising, z = 14 (bcc first and second shells), mean field x fcc ratio', 14 * T_central / 12),
    ('Ising, z = 15.5 (Poisson-Voronoi faces), mean field x fcc ratio', 15.5 * T_central / 12),
    ('Ising, z = 15.5, mean field (top of the plausible band)', 15.5),
    ('Ising, z = 12, mean field (no fluctuation reduction)', 12.0),
    ('Ising, fcc with a 1/r^6 tail, mean field', S[('fcc', 6)]),
    ('Ising, fcc with a 1/r^5 tail, mean field', S[('fcc', 5)]),
    ('well defined against anti-alignment (J = well/2)', T_central / 2),
]
for lab, val in variants:
    p(f'   {val:7.3f}   {lab}')
p()
plaus = [T_central, 6.0 * lit[('xy', 'sc')] / 3.0, 15.5 * T_central / 12, 12.0, S[('fcc', 6)], 15.5]
p(f'   plausible band (symmetry XY to Ising, z = 12 to 15.5, fluctuations on or off, 1/r^6 tail): {min(plaus):.2f} to {max(plaus):.2f}')
p(f'   band relative to the central value: x{min(plaus) / T_central:.2f} to x{max(plaus) / T_central:.2f}')
print('\n'.join(out))
