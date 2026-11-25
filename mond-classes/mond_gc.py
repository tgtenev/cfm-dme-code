r"""MOND's boosts of the globular clusters' dynamical mass-to-light ratios, against the measured ratios.

The law is MOND's simple interpolation function at a0 = 1.2e-10 m/s^2 (mond_law.py), with the Galaxy's field as
the external field through the angle-averaged QUMOND field. For each of the 133 globular clusters with rich
kinematics, MOND predicts that the Newtonian-inferred dynamical mass-to-light ratio is Y_* B, with Y_* the
stellar ratio and B the boost of the internal field at the 3D half-mass radius. The script computes B for five
treatments of the external field, the slope of the predicted ratios against log P, and the likelihood of
MOND's prediction against flat Newtonian ratios. P is the ambient rescaling factor of Paper II's Section 6.3,
P = nu(Xbar(l_c))/nu_sun, at the adopted floor of the catchment length, 58 pc (58.254 pc).

It supports:
  Paper I, Table 1 (small systems) and Section 3.7: MOND predicts boosts of 1.02 to 2.13 in the globular
      clusters (the flat-curve external field below, B from 1.016 to 2.127);
  Paper II, Section 6.6: the boosts of 1.02 to 2.13 across the 133 rich clusters; the positive log-slope of
      +0.09 to +0.27 against log P (the slope fitted to Y_* B, over the five treatments); Delta lnL = 14 for
      flat Newtonian ratios over MOND (14.17, the flat-curve external field).
The validation lines reproduce, from the figure data of globular-clusters/gc_dynamical.py, the fit of Paper II's
Section 6.3 and Figure 3b at 58 pc: the slope -0.20 +- 0.08, Delta lnL = 60, Y_* = 1.90 and the line 2.26 P.

Inputs:
    <data>/baumgardt-gc/combined_table.txt    the globular-cluster database of Baumgardt and Hilker, not
                                              redistributed (columns R_GC, N_RV, N_PM, Mass, DM, M/L_V,
                                              DM/L, rh,m)
    globular-clusters/gc_dynamical-figdata.tsv  written by globular-clusters/gc_dynamical.py (3 to 5 min),
                                              which must run first; it gives NGC 2419's per-star Jeans ratio
                                              and the fit that the validation lines check
Output: mond-classes/mond_gc-results.txt, also printed.
Command, from the repository root, after globular-clusters/gc_dynamical.py (a few seconds):
    python -I -B mond-classes/mond_gc.py data
Needs numpy.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8')
from mond_law import mond_g, mond_monopole, mond_newtonian_of_total  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else 'data'
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, 'mond_gc-results.txt')
GC_TABLE = os.path.join(DATA, 'baumgardt-gc', 'combined_table.txt')
GC_FIG_NAME = 'globular-clusters/gc_dynamical-figdata.tsv'
GC_FIG = os.path.join(REPO, *GC_FIG_NAME.split('/'))
LC = 58.254                         # the adopted floor of the catchment length, pc
LCS = '%g' % LC                     # its label in the figure data

# the constants of globular-clusters/gc_dynamical.py, which computes P
G_GC, A0_GC, MSUN_GC, PC_GC = 6.674e-11, 1.2e-10, 1.989e30, 3.0857e16
MB_GC = 6.5e10 * MSUN_GC            # the Galaxy as a point mass, kg
VF = 230e3                          # the Galaxy's flat rotation speed, m/s
LN10 = math.log(10.0)
SG_C = np.linspace(0.005, 0.5, 496)       # the grid of intrinsic scatter, dex
BS_C = np.linspace(-2.0, 2.0, 4001)       # the grid of slopes

LINES = []


def P(s=''):
    LINES.append(s)
    print(s, flush=True)


# ----------------------------------------------------------------------------------------------
# the ambient rescaling factor P, as globular-clusters/gc_dynamical.py computes it
# ----------------------------------------------------------------------------------------------
def amplify(y):
    """The total field X from the Newtonian field y on the simple function, X = y nu(X), both in units of a0."""
    return 0.5 * (y + math.sqrt(y * y + 4.0 * y))


def nu_x(X):
    """The simple function in total-field form."""
    return 1.0 + 1.0 / X


NUSUN = nu_x(amplify(G_GC * MB_GC / ((8.2e3 * PC_GC) ** 2) / A0_GC))


def P_lc(mass, rgc, lc):
    """P = nu(Xbar(l_c))/nu_sun: Xbar sums the cluster's own field at l_c (pc) and the Galaxy's point-mass field
    at R_GC (kpc)."""
    yamb = G_GC * MB_GC / ((rgc * 1e3 * PC_GC) ** 2) / A0_GC
    yown = G_GC * mass * MSUN_GC / ((lc * PC_GC) ** 2) / A0_GC
    return nu_x(amplify(yown + yamb)) / NUSUN


# ----------------------------------------------------------------------------------------------
# the fits, in the convention of globular-clusters/gc_dynamical.py
# ----------------------------------------------------------------------------------------------
def wls_scan(y, X, e, j, sgrid):
    """For each intrinsic scatter s, weighted least squares of y on the columns of X with weights 1/(s^2 + e^2):
    the log-likelihood at its maximum, the coefficient j, its variance and all coefficients."""
    llm, bh, vb, cf = [], [], [], []
    for s in sgrid:
        w = 1.0 / (s * s + e * e)
        A = X * np.sqrt(w)[:, None]
        M = A.T @ A
        c = np.linalg.solve(M, A.T @ (y * np.sqrt(w)))
        res = y - X @ c
        llm.append(-0.5 * (w * res * res).sum() + 0.5 * np.log(w).sum())
        bh.append(c[j])
        vb.append(np.linalg.inv(M)[j, j])
        cf.append(c)
    return np.array(llm), np.array(bh), np.array(vb), cf


def slope_fit(x, y, e):
    """log10 Y = a + b log10 P, scatter free: b, its 68 percent range, Delta lnL (b = 0 against b = 1), the
    scatter at b = 0 and the population values at b = 0 and 1."""
    X = np.vstack([np.ones_like(x), x]).T
    llm, bh, vb, _ = wls_scan(y, X, e, 1, SG_C)
    prof = (llm[:, None] - 0.5 * (BS_C[None, :] - bh[:, None]) ** 2 / vb[:, None]).max(axis=0)
    i = int(np.argmax(prof))
    ok = BS_C[prof >= prof[i] - 0.5]

    def at_b(bv):
        v = llm - 0.5 * (bv - bh) ** 2 / vb
        k = int(np.argmax(v))
        return v[k], SG_C[k]
    l0, s0 = at_b(0.0)
    l1, s1 = at_b(1.0)
    res = dict(b=float(BS_C[i]), blo=float(ok.min()), bhi=float(ok.max()), d=float(l0 - l1), s0=float(s0))
    for bv, key, s in ((0.0, 'Y0', s0), (1.0, 'Y1', s1)):
        w = 1.0 / (s * s + e * e)
        res[key] = float(10 ** ((w * (y - bv * x)).sum() / w.sum()))
    return res


def offset_fit(y, e):
    """log10 Y = a, scatter free: the maximum log-likelihood, the population value and the scatter."""
    X = np.ones((len(y), 1))
    llm, bh, vb, cf = wls_scan(y, X, e, 0, SG_C)
    k = int(np.argmax(llm))
    return float(llm[k]), float(10 ** cf[k][0]), float(SG_C[k])


# ----------------------------------------------------------------------------------------------
def read_clusters():
    raw = open(GC_TABLE, encoding='utf-8').read().splitlines()
    hdr = raw[0].lstrip('#').split()
    col = {n: i for i, n in enumerate(hdr)}
    rows = []
    for ln in raw:
        if ln.startswith('#') or not ln.strip():
            continue
        c = ln.split()
        rows.append(dict(name=c[0], rgc=float(c[col['R_GC']]), nrv=int(c[col['N_RV']]), npm=int(c[col['N_PM']]),
                         mass=float(c[col['Mass']]), dmass=float(c[col['DM']]), ml=float(c[col['M/L_V']]),
                         dml=float(c[col['DM/L']]), rh=float(c[col['rh,m']])))
    return rows


def read_figdata():
    """From gc_dynamical.py's figure data at LC: NGC 2419's Jeans ratio and interval, each rich cluster's P and
    ratio, and the fitted slope line."""
    fig_rich, jeans = {}, None
    with open(GC_FIG, encoding='utf-8') as f:
        for ln in f:
            if ln.startswith('#'):
                continue
            t = ln.rstrip('\n').split('\t')
            if t[0] == 'jeans' and t[1] == LCS:
                jeans = (float(t[4]), float(t[5]), float(t[6]))
            if t[0] == 'rich' and t[1] == LCS:
                fig_rich[t[2]] = (float(t[3]), float(t[4]), float(t[6]))
            if t[0] == 'line' and t[1] == LCS and t[2] in ('rich_slope', 'rich_Y0', 'rich_Y1'):
                fig_rich['_' + t[2]] = (float(t[4]), float(t[5]), float(t[6]), float(t[8]))
    return fig_rich, jeans


def main():
    P('MOND\'S BOOSTS OF THE GLOBULAR CLUSTERS\' DYNAMICAL MASS-TO-LIGHT RATIOS')
    P('=' * 110)
    P(f'data folder: {DATA}; the ambient rescaling factor P at the catchment length {LC:g} pc, NGC 2419\'s Jeans ratio and')
    P(f'the fit of the validation from {GC_FIG_NAME}')
    P('')
    good = [r for r in read_clusters() if r['mass'] > 0 and r['ml'] > 0 and r['rh'] > 0]
    for r in good:
        n = r['nrv'] + 2 * r['npm'] if r['npm'] > 0 else r['nrv']
        r['floor'] = math.sqrt(2.0 / n) if n > 0 else float('inf')
        r['L'] = r['mass'] / r['ml']
        r['P'] = P_lc(r['mass'], r['rgc'], LC)
    rich = [dict(r) for r in good if r['dml'] > 0 and (r['npm'] > 0 or r['nrv'] >= 50)]
    fig_rich, jeans = read_figdata()
    for r in rich:
        if r['name'] == 'NGC_2419':                  # NGC 2419 at its per-star Jeans ratio and interval
            r['ml'] = jeans[0]
            r['e'] = 0.5 * (math.log10(jeans[2]) - math.log10(jeans[1]))
        else:                                        # the quoted error, raised to the sampling floor sqrt(2/N)
            r['e'] = max(r['dml'] / r['ml'], r['floor']) / LN10
    dP = max(abs(r['P'] - fig_rich[r['name']][0]) for r in rich)
    dY = max(abs(r['ml'] - fig_rich[r['name']][1]) for r in rich)
    x = np.array([math.log10(r['P']) for r in rich])
    y = np.array([math.log10(r['ml']) for r in rich])
    e = np.array([r['e'] for r in rich])
    fz = slope_fit(x, y, e)
    rs = fig_rich['_rich_slope']
    P(f'THE {len(rich)} GLOBULAR CLUSTERS WITH RICH KINEMATICS')
    P('-' * 110)
    P(f'  sample: {len(rich)} clusters (N_PM > 0 or N_RV >= 50, nonzero quoted error), NGC 2419 at its Jeans ratio {jeans[0]:.3f};'
      f' errors raised to the floor sqrt(2/N)')
    P(f'  validation: P at {LC:g} pc against gc_dynamical.py\'s figure data: largest difference {dP:.1e}; the ratios: {dY:.1e}')
    P(f'  validation, the slope of log Y on log P({LC:g} pc), scatter fitted: {fz["b"]:+.3f} (68% {fz["blo"]:+.3f} to {fz["bhi"]:+.3f}),'
      f' Delta lnL (slope 0 against 1) {fz["d"]:+.2f}; population values {fz["Y0"]:.4f} and {fz["Y1"]:.4f}')
    P(f'  gc_dynamical.py at {LC:g} pc: {rs[0]:+.3f} ({rs[1]:+.3f} to {rs[2]:+.3f}), {rs[3]:+.2f};'
      f' {fig_rich["_rich_Y0"][0]:.4f} and {fig_rich["_rich_Y1"][0]:.4f}')
    Ystar = fz['Y0']
    # MOND's boost of the Newtonian-inferred mass-to-light ratio at the 3D half-mass radius
    for r in rich:
        Mb = Ystar * r['L']
        gN = G_GC * 0.5 * Mb * MSUN_GC / (r['rh'] * PC_GC) ** 2
        gNc = G_GC * 0.5 * r['mass'] * MSUN_GC / (r['rh'] * PC_GC) ** 2
        Rm = r['rgc'] * 1e3 * PC_GC
        xeA = VF ** 2 / (max(r['rgc'], 3.0) * 1e3 * PC_GC)                # field A: the total field of a flat curve
        eNA = float(mond_newtonian_of_total(xeA))
        eNB = G_GC * MB_GC / Rm ** 2                                       # field B: the Newtonian field of the point mass
        r['B_iso'] = float(mond_g(gN)) / gN
        r['B_A'] = float(mond_monopole(gN, eNA)[0]) / gN
        r['B_B'] = float(mond_monopole(gN, eNB)[0]) / gN
        r['B_A_1D'] = (float(mond_g(gN + eNA)) - xeA) / gN                  # Famaey and McGaugh 2012, Eq. 60
        r['B_A_cat'] = float(mond_monopole(gNc, eNA)[0]) / gNc               # the internal field from the catalog mass
    variants = {'A': ('field A, the Galaxy\'s field for a flat rotation curve of 230 km/s, QUMOND monopole', 'B_A'),
                'B': ('field B, the Newtonian field of the Galaxy as a point mass of 6.5e10 Msun, QUMOND monopole', 'B_B'),
                'A1D': ('field A, the one-dimensional QUMOND formula of Famaey and McGaugh 2012', 'B_A_1D'),
                'Acat': ('field A, QUMOND monopole, the internal field from the catalog mass', 'B_A_cat'),
                'iso': ('no external field', 'B_iso')}
    lnl_N, YN, sN = offset_fit(y, e)
    P('')
    P('  MOND\'s prediction: the Newtonian-inferred ratio is Y_* B, with B the boost of the field at the 3D half-mass radius')
    P(f'  (catalog rh,m) for baryons Y_* L, Y_* = {Ystar:.3f} (the flat fit), the simple function at 1.2e-10 m/s^2 and the')
    P('  Galaxy\'s field as the external field, at least 3 kpc out for field A. For each treatment: the range and median of B;')
    P(f'  the slope of the noise-free prediction Y_* B against log P({LC:g} pc), from the same fit that gives the measured')
    P(f'  slope of {fz["b"]:+.2f}; and Delta lnL, Newton (flat) minus MOND (log Y = a + log B), with the population value Y')
    P('  and the scatter s fitted under each.')
    slopes = []
    for k, (lab, key) in variants.items():
        lB = np.array([math.log10(r[key]) for r in rich])
        fs = slope_fit(x, math.log10(Ystar) + lB, e)
        lnl_M, YM, sM = offset_fit(y - lB, e)
        Bv = np.array([r[key] for r in rich])
        slopes.append(fs['b'])
        P(f'  {lab}')
        P(f"     B {Bv.min():.3f} to {Bv.max():.3f}, median {np.median(Bv):.3f}; the slope fitted to Y_* B"
          f" {fs['b']:+.3f} ({fs['blo']:+.3f} to {fs['bhi']:+.3f}, the scatter of the prediction about the line {fs['s0']:.3f} dex)")
        P(f"     Delta lnL Newton minus MOND {lnl_N - lnl_M:+.2f} (Y {YN:.3f}, s {sN:.3f}; Y {YM:.3f}, s {sM:.3f})")
    P(f'  The slopes fitted to Y_* B over the five treatments: {min(slopes):+.3f} to {max(slopes):+.3f}')
    P('')
    with open(RESULTS, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LINES) + '\n')


if __name__ == '__main__':
    main()
