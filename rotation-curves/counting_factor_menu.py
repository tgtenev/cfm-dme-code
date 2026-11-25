"""The counting-factor menu on the rotation curves (Paper I §4.2 and Table 2, Paper II §6.1): the scale relation
a0 = c H_dS ln(1 + delta) / kappa of Paper I, Eq. (10), at kappa = 1/3, 2/3, 1 and 2, in the four setups of
constrained_fit.py, at both Hubble constants.

The menu is that of Paper I §3.2. kappa = 1 compares the contrast across one catchment length with the per-axis share
of the set point; 1/3 compares it with the whole set point; 2 takes the contrast across a window that reaches one
catchment length to each side, two catchment lengths long; 2/3 does both. Along each constrained curve every galaxy's
mass-to-light ratios, distance and inclination are refitted at every node, as in constrained_fit.py part main, and the
differences against kappa = 1 are clustered by galaxy over the same 2,000 bootstrap draws (seed 20261114).

The grids: kappa = 1 and 1/3 on the grids of constrained_fit.py (0.05 to 0.45 in steps of 0.01; 0.02 to 0.12 in steps
of 0.0025), so their minima and their difference reproduce constrained_fit-results.txt; kappa = 2/3 from 0.08 to 0.17
in steps of 0.0025 and kappa = 2 from 0.28 to 0.60 in steps of 0.01, which span scales of 0.75 to 1.25e-10 m/s^2 at
both Hubble constants. On the full field solution the correction for other mass-to-light ratios is linear in delta
between its nodes 0.03 and 0.3 and is extrapolated beyond 0.3, as on the constrained curves of constrained_fit.py up to
0.45, so the full-field numbers at kappa = 2 carry that extrapolation.

At each minimum the script also gives the onset of the hardening, X = xi*/ln(1 + delta) with e^xi* - 1 = sqrt(2) xi*,
and the dilute slope s = ln(1 + delta)/delta (Paper I §3.5).

Inputs: those of constrained_fit.py. Run from the repository root, after constrained_solve.py parts gamma and upsvar:
    python rotation-curves/counting_factor_menu.py [--setups sparc-relation,model-relation,sparc-full,model-full]
It writes rotation-curves/counting_factor_menu-results.txt. The whole run took 19 minutes, most of it on the two
setups with the full field solution.
"""
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import constrained_core as CC
import constrained_fit as CF

RC = CC.RC
ARGV = sys.argv[1:]
T0 = time.time()
LINES = []


def out(s=''):
    print(s, flush=True)
    LINES.append(s)


def grid(a, b, step):
    n = int(round((b - a) / step))
    return [round(a + step * k, 4) for k in range(n + 1)]


MENU = [(1.0, '1', CF.DELTAS), (1.0 / 3.0, '1/3', CF.DELTAS_K3), (2.0 / 3.0, '2/3', grid(0.08, 0.17, 0.0025)),
        (2.0, '2', grid(0.28, 0.60, 0.01))]


def xi_star():
    lo, hi = 0.1, 2.0
    for _ in range(200):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if math.expm1(m) < math.sqrt(2.0) * m else (lo, m)
    return 0.5 * (lo + hi)


XI = xi_star()
SETUPS = [('sparc', False), ('model', False), ('sparc', True), ('model', True)]
NAMES = {('sparc', False): "SPARC's tabulated field, the algebraic relation (the fit of record)",
         ('model', False): "the model's own Newtonian field of disks with thickness, the algebraic relation",
         ('sparc', True): "SPARC's tabulated field, the full field solution",
         ('model', True): "the model's own Newtonian field of disks with thickness, the full field solution"}

T1, gals, ds, nnew, dmax, ref = CF.load()
W = RC.boot_weights(len(ds.names), CC.NBOOT, CC.SEED)
out('THE COUNTING-FACTOR MENU ON THE ROTATION CURVES: THE SCALE RELATION a0 = c H_dS ln(1 + delta) / kappa AT kappa = 1/3, 2/3, 1 AND 2')
out(f'{len(ds.names)} galaxies; {CC.NBOOT} bootstrap draws by galaxy, seed {CC.SEED}; xi* = {XI:.4f}')
want = ARGV[ARGV.index('--setups') + 1].split(',') if '--setups' in ARGV else None
for inp, full in SETUPS:
    tag = f'{inp}-{"full" if full else "relation"}'
    if want and tag not in want:
        continue
    out('')
    out(f'== {NAMES[(inp, full)]}')
    for h0 in CC.HUBBLE:
        res = {}
        for kappa, lab, deltas in MENU:
            T = CC.ConstrainedLine(ds, inp, full, h0, deltas, kappa=kappa)
            draws = np.array([T.minimum(w)[0] for w in W])
            res[lab] = (T, draws)
        T1_, b1 = res['1']
        out(f'   H0 = {h0} km/s/Mpc (c H_dS = {CC.c_hds(h0) * 1e10:.4f}e-10 m/s^2)')
        for kappa, lab, deltas in MENU:
            T, b = res[lab]
            d = T.dmin
            lam = math.log1p(d)
            a0 = CC.c_hds(h0) * lam / kappa
            line = (f'      kappa = {lab:>3}: delta = {d:.4f}{" AT AN END OF THE GRID" if T.edge else ""}, '
                    f'a0 = {a0 * 1e10:.4f}e-10, chi^2 = {T.chi2:.2f}, onset X = {XI / lam:.2f}, dilute slope {lam / d:.3f}')
            if lab != '1':
                diff = T.chi2 - T1_.chi2
                sd = CC.sd_of(b - b1)
                line += (f'; minus kappa = 1: {diff:+.1f} +/- {sd:.1f} ({diff / sd:+.2f} SD), '
                         f'share of draws in which it fits better {np.mean(b < b1):.3f}')
            out(line)
    out(f'   [{tag}: {time.time() - T0:.0f} s]')

with open(os.path.join(HERE, 'counting_factor_menu-results.txt'), 'w', newline='\n', encoding='utf-8') as fh:
    fh.write('\n'.join(LINES) + '\n')
print('wrote rotation-curves/counting_factor_menu-results.txt')
