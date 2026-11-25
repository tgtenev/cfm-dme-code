"""The data of Paper I's Figure 4: the radial-acceleration relation of the 153 SPARC galaxies at the minimum of the
constrained fit and at the fitted scale of MOND's simple function, with the binned residuals of both.

It imports the fit machinery through constrained_core.py (which imports refit_core.py and aqual-refit-fit.py of this
folder) and fits each galaxy's mass-to-light ratios, distance and inclination at one global node for each law:

  - the constrained fit, the boost law of Paper I, Eq. (8), at the constrained minimum on SPARC's field with the
    algebraic relation, delta_fv = 0.1966 with a0 = c H_dS ln(1 + delta_fv) at H0 = 67.4 km/s/Mpc, 0.9725e-10 m/s^2
    at that rounded delta_fv (constrained_fit-results.txt, the fit of record at H0 = 67.4, where the objective
    evaluated directly at the minimum is 4323.12);
  - MOND's simple function at its fitted scale, 1.0796e-10 m/s^2 (constrained_fit-results.txt, the same setup's table
    at equal freedom; objective 4350.30).
In the code, T3 is the boost law and SIM the simple function, as in constrained_core.py.

It writes two files into rotation-curves/, one row per valid point, sorted by galaxy:
  rar-law.tsv     lg g_bar, lg g_obs', lg (g_bar/a0), residual (dex), for the constrained fit
  rar-simple.tsv  lg g_bar, lg g_obs', lg (g_bar/a0), residual (dex), for the simple function
where g_obs' = g_obs/q^2 is the observed acceleration corrected by the fitted distance and inclination, g_bar is
invariant under them, and the residual is lg g_obs' - lg(nu g_bar). It prints the objectives, the scatter and the
binned mean residuals in 0.25-dex bins of lg g_bar from -12.6, kept where both laws have 8 points or more, as the lower
panel of Figure 4 shows them. The figure and the script that draws it are not part of this repository.

Inputs: those of constrained_fit.py (data/sparc/ and the Gamma tables in rotation-curves/, which the loader reads).
Run from the repository root (about 4 s):
    python rotation-curves/rar_figure_data.py > rotation-curves/rar_figure_data-results.txt
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import constrained_core as CC

RC, AF = CC.RC, CC.AF
OUT = HERE
DELTA_FV = 0.1966            # the constrained minimum on SPARC's field with the relation, H0 = 67.4 (constrained_fit-results.txt)
A0_SIM = 1.0796e-10          # the fitted scale of the simple function in the same setup (constrained_fit-results.txt)


def load():
    """The dataset as constrained_fit.py loads it."""
    T1, gals = AF.load()
    ds = CC.XDataset(gals)
    ds.attach_eps(AF.upsvar_rows())
    CC.attach_x(ds, gals)
    CC.attach_eps_std(ds, CC.upsvar_x_rows())
    return ds


def dump(ds, c, a0, path):
    res = ds.fit(c)
    P = ds.points(c, res)
    rows = []
    for n in ds.names:
        p = P[n]
        k = p['valid'] & (p['gbar'] > 0)
        lgb = np.log10(p['gbar'][k])
        lgo = p['r'][k] + p['lg'][k]
        for a, b, r in zip(lgb, lgo, p['r'][k]):
            rows.append((a, b, a - math.log10(a0), r))
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write('%.5f\t%.5f\t%.5f\t%.5f\n' % row)
    arr = np.array(rows)
    return float(res['F'].sum()), arr


def main():
    ds = load()
    a0t = CC.a0_of_delta(DELTA_FV, 67.4)
    cases = [('the constrained fit', CC.ctx('sparc', 'T3', delta=DELTA_FV, a0=a0t), a0t, 'rar-law.tsv',
              'constrained_fit-results.txt, the fit of record evaluated directly at the minimum: 4323.12'),
             ("MOND's simple function", CC.ctx('sparc', 'SIM', a0=A0_SIM), A0_SIM, 'rar-simple.tsv',
              'constrained_fit-results.txt, the fit of record at equal freedom: 4350.30')]
    print('FIGURE DATA FOR PAPER I, FIGURE 4 (SPARC\'s field, the algebraic relation, H0 = 67.4 km/s/Mpc)')
    store = {}
    for lab, c, a0, fn, rec in cases:
        chi, arr = dump(ds, c, a0, os.path.join(OUT, fn))
        store[lab] = arr
        print(f'  {lab}: a0 = {a0 * 1e10:.4f}e-10 m/s^2, chi^2 = {chi:.2f} ({rec}); {len(arr)} points; '
              f'scatter {np.std(arr[:, 3]):.4f} dex; wrote rotation-curves/{fn}')
    print('  binned mean residual (dex) in 0.25-dex bins of lg g_bar from -12.6, bins with 8 points or more for both laws:')
    bt = np.floor((store['the constrained fit'][:, 0] + 12.6) / 0.25).astype(int)
    bs = np.floor((store["MOND's simple function"][:, 0] + 12.6) / 0.25).astype(int)
    big = {'t': (0.0, ''), 's': (0.0, '')}
    for b in sorted(set(bt) | set(bs)):
        nt, ns = int(np.sum(bt == b)), int(np.sum(bs == b))
        if nt < 8 or ns < 8:
            continue
        mt = float(np.mean(store['the constrained fit'][bt == b, 3]))
        ms = float(np.mean(store["MOND's simple function"][bs == b, 3]))
        lo = -12.6 + 0.25 * b
        print(f'    [{lo:6.2f},{lo + 0.25:6.2f})  constrained {nt:4d} {mt:+.4f}   simple {ns:4d} {ms:+.4f}')
        if abs(mt) > abs(big['t'][0]):
            big['t'] = (mt, f'{lo:.2f}')
        if abs(ms) > abs(big['s'][0]):
            big['s'] = (ms, f'{lo:.2f}')
    print(f'  largest |bin offset|: the constrained fit {abs(big["t"][0]):.4f} dex (bin from {big["t"][1]}), '
          f'MOND\'s simple function {abs(big["s"][0]):.4f} dex (bin from {big["s"][1]})')
    lam = math.log1p(DELTA_FV)
    print(f'  the constrained fit: delta_fv {DELTA_FV}, lambda {lam:.4f}, dilute slope s {lam / DELTA_FV:.4f}, '
          f's^-1/2 {(lam / DELTA_FV) ** -0.5:.4f}')


if __name__ == '__main__':
    main()
