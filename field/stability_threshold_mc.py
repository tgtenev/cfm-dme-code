"""Monte Carlo check of the ordering point of the folds of renewal domains.

Model. Ising orientational variables s = +1 or -1 on a lattice, pair energy -J s_i s_j between nearest neighbours
(J = 1; the pair well, aligned against uncorrelated, is J). Heat-bath updates at noise temperature T (in units of J),
on sublattices that hold no two neighbours.

What it measures. The Binder cumulant U4 = 1 - <m^4> / (3 <m^2>^2) for three lattice sizes. The temperature where the
curves cross is the ordering temperature T_c (in units of J).

Validation. The crossing must reproduce the published three-dimensional Ising values, T_c = 4.5115 J on the simple
cubic lattice and T_c = 9.794 J on the face-centred cubic lattice.

Supports Paper II section 5.3 and Table 1: compact domains with 12 neighbours order at about 9.75 misfit energies
(job fcc0, 9.7494), and with 6 neighbours at about 4.5 (job sc0, 4.4915). field/stability_threshold_calc.py reads both.

Inputs: none. Outputs: field/stability_threshold_mc-sc0.txt and field/stability_threshold_mc-fcc0.txt.
Run from the repository root, one job at a time (about 2 minutes each); seeds are fixed per (job, L, T):
    python field/stability_threshold_mc.py sc0 field/stability_threshold_mc-sc0.txt
    python field/stability_threshold_mc.py fcc0 field/stability_threshold_mc-fcc0.txt
Each job writes its own results file and prints its progress.
"""
import sys

import numpy as np


def build_lattice(kind, L):
    """Return (neighbour table, list of sublattice index arrays, number of sites).

    The integer grid is L x L x L with periodic boundaries (L even).
    sc : every grid point is a site; 6 neighbours; 2 sublattices by parity.
    fcc: grid points with i+j+k even; 12 neighbours (+-1,+-1,0 and permutations); 4 sublattices.
    """
    grid = np.indices((L, L, L)).reshape(3, -1).T
    if kind == 'sc':
        keep = np.ones(len(grid), dtype=bool)
        vecs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    elif kind == 'fcc':
        keep = (grid.sum(1) % 2 == 0)
        vecs = []
        for a in (1, -1):
            for b in (1, -1):
                vecs += [(a, b, 0), (a, 0, b), (0, a, b)]
    else:
        raise ValueError(kind)
    sites = grid[keep]
    index = -np.ones((L, L, L), dtype=np.int64)
    index[sites[:, 0], sites[:, 1], sites[:, 2]] = np.arange(len(sites))
    nbr = np.empty((len(sites), len(vecs)), dtype=np.int64)
    for c, v in enumerate(vecs):
        q = (sites + np.array(v)) % L
        nbr[:, c] = index[q[:, 0], q[:, 1], q[:, 2]]
    assert (nbr >= 0).all()
    if kind == 'sc':
        colour = sites.sum(1) % 2
    else:
        colour = (sites[:, 0] % 2) * 2 + (sites[:, 1] % 2)  # 0,3,2,1 for the four FCC sublattices
    subs = [np.where(colour == c)[0] for c in np.unique(colour)]
    # independence check: no site shares a sublattice with any of its neighbours
    for s in subs:
        cs = colour[s[0]]
        assert (colour[nbr[s]] != cs).all()
    return nbr, subs, len(sites)


def run(kind, L, T, n_eq, n_meas, seed):
    rng = np.random.default_rng(seed)
    nbr, subs, N = build_lattice(kind, L)
    s = np.ones(N, dtype=np.int8)  # ordered start; the noise disorders it if T > T_c
    beta = 1.0 / T
    z = nbr.shape[1]
    hvals = np.arange(-z, z + 1)
    p_table = 1.0 / (1.0 + np.exp(-2.0 * beta * hvals))  # heat-bath probability of +1 for each local field
    series = np.empty(n_meas)
    for sweep in range(n_eq + n_meas):
        for sub in subs:
            h = s[nbr[sub]].sum(1, dtype=np.int32)
            p_up = p_table[h + z]
            new = np.where(rng.random(len(sub)) < p_up, 1, -1).astype(np.int8)
            s[sub] = new
        if sweep >= n_eq:
            m = s.mean(dtype=np.float64)
            series[sweep - n_eq] = m
    m2 = np.mean(series ** 2)
    m4 = np.mean(series ** 4)
    mabs = np.mean(np.abs(series))
    u4 = 1.0 - m4 / (3.0 * m2 ** 2)
    # blocking error of U4 (10 blocks)
    blocks = np.array_split(series, 10)
    u4b = []
    for b in blocks:
        b2 = np.mean(b ** 2)
        b4 = np.mean(b ** 4)
        u4b.append(1.0 - b4 / (3.0 * b2 ** 2))
    u4err = np.std(u4b, ddof=1) / np.sqrt(len(u4b))
    return N, u4, u4err, mabs


def crossing(Ts, u_small, u_large):
    """Linear interpolation of the first sign change of u_large - u_small (the crossing)."""
    d = np.array(u_large) - np.array(u_small)
    for i in range(len(Ts) - 1):
        if d[i] == 0:
            return Ts[i]
        if d[i] * d[i + 1] < 0:
            t0, t1 = Ts[i], Ts[i + 1]
            return t0 + (t1 - t0) * d[i] / (d[i] - d[i + 1])
    return float('nan')


JOBS = {
    # name: (lattice, sizes, temperatures, equilibration sweeps, measured sweeps)
    'sc0': ('sc', [8, 12, 16], [4.40, 4.45, 4.50, 4.55, 4.60, 4.65], 2000, 16000),
    'fcc0': ('fcc', [8, 12, 16], [9.55, 9.65, 9.75, 9.85, 9.95, 10.05], 2000, 16000),
}


def main():
    job, out = sys.argv[1], sys.argv[2]
    kind, sizes, Ts, n_eq, n_meas = JOBS[job]
    lines = []
    lines.append(f'job {job}: lattice {kind}')
    lines.append(f'sweeps: {n_eq} equilibration + {n_meas} measured per point; heat-bath, sublattice updates')
    table = {}
    for L in sizes:
        row = []
        for T in Ts:
            seed = 100000 * sizes.index(L) + int(1000 * T)
            N, u4, err, mabs = run(kind, L, T, n_eq, n_meas, seed)
            row.append((u4, err, mabs))
            print(job, L, T, N, f'{u4:.4f} +- {err:.4f}', flush=True)
        table[L] = row
    lines.append('')
    lines.append('T        ' + '  '.join(f'U4(L={L})          <|m|>' for L in sizes))
    for i, T in enumerate(Ts):
        cells = '  '.join(f'{table[L][i][0]:.4f} +- {table[L][i][1]:.4f}  {table[L][i][2]:.3f}' for L in sizes)
        lines.append(f'{T:7.3f}  {cells}')
    lines.append('')
    c1 = crossing(Ts, [r[0] for r in table[sizes[0]]], [r[0] for r in table[sizes[1]]])
    c2 = crossing(Ts, [r[0] for r in table[sizes[1]]], [r[0] for r in table[sizes[2]]])
    c3 = crossing(Ts, [r[0] for r in table[sizes[0]]], [r[0] for r in table[sizes[2]]])
    lines.append(f'crossing L={sizes[0]}/{sizes[1]}: {c1:.4f}')
    lines.append(f'crossing L={sizes[1]}/{sizes[2]}: {c2:.4f}')
    lines.append(f'crossing L={sizes[0]}/{sizes[2]}: {c3:.4f}')
    vals = [c for c in (c1, c2, c3) if np.isfinite(c)]
    if vals:
        lines.append(f'T_c estimate (mean of crossings): {np.mean(vals):.4f}, spread {np.ptp(vals):.4f}')
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
