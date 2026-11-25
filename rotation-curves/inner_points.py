"""Paper I §4.1: the points of the SPARC sample that lie inside 1 kpc and 0.5 kpc of a galaxy's centre, the galaxies
that hold them, and how far the field averaged over a window one catchment length long departs there from the local
field.

The script first reproduces Paper I's sample (153 galaxies, 3,168 points) with the cuts of the fit code, as
refit_core.py (read_table1, read_rotmod) applies them in the loader that constrained_fit.py uses through
aqual-refit-fit.py:
  - galaxy listed in SPARC Table 1 (data/sparc/SPARC_Lelli2016c.mrt), quality flag Q != 3, inclination >= 30 deg;
  - rotmod rows of data/sparc/rotmod_ltg/<name>_rotmod.dat that are not comments, have >= 6 columns, start with a
    digit or '.', and have R > 0;
  - at least 3 such rows.
In the chi^2 a point counts only while g_bar = Vg|Vg|/R + Yd Vd^2/R + Yb Vb^2/R > 0 at the trial ratios.
It parses the files independently (fixed-width Table 1, its own rotmod reader) and reads them only.

It then counts the points inside 1 kpc and 0.5 kpc, and gives at every point the departure |<g>/g - 1| of the field
averaged over a window one catchment length long (200 pc, centred on the point, along the radius) from its local
value, on the observed curve and on a smooth model field. Paper I §3.2 takes the contrast over such a window.

Run from the repository root (a few seconds):
    python rotation-curves/inner_points.py data/sparc > rotation-curves/inner_points-results.txt
"""
import math
import os
import sys

SP = sys.argv[1] if len(sys.argv) > 1 else 'data/sparc'
T1 = os.path.join(SP, 'SPARC_Lelli2016c.mrt')
ROT = os.path.join(SP, 'rotmod_ltg')


def table1(path):
    """Rows after the last dashed rule, split on blanks (the file's columns are shifted by one byte against its
    byte-by-byte header): name, T, D, e_D, f_D, Inc, e_Inc, L, e_L, Reff, SBeff, Rdisk, SBdisk, MHI, RHI, Vflat,
    e_Vflat, Q, Ref."""
    lines = open(path, encoding='ascii', errors='replace').read().splitlines()
    last_rule = max(i for i, l in enumerate(lines) if l.strip() and set(l.strip()) == {'-'})
    T = {}
    for l in lines[last_rule + 1:]:
        f = l.split()
        if len(f) < 18:
            continue
        T[f[0]] = dict(D=float(f[2]), eD=float(f[3]), inc=float(f[5]), einc=float(f[6]), Q=int(f[17]))
    return T


def rotmod(path):
    pts = []
    for line in open(path, encoding='ascii', errors='replace'):
        if line.startswith('#'):
            continue
        c = line.split()
        if len(c) < 6 or not (c[0][0].isdigit() or c[0][0] == '.'):
            continue
        if float(c[0]) <= 0:
            continue
        pts.append([float(x) for x in c[:6]])
    return pts


T = table1(T1)
files = sorted(f for f in os.listdir(ROT) if f.endswith('_rotmod.dat'))
allg = {f[:-len('_rotmod.dat')]: rotmod(os.path.join(ROT, f)) for f in files}
print('Table 1 rows: %d; rotmod files: %d; files without a Table 1 row: %s'
      % (len(T), len(allg), sorted(set(allg) - set(T)) or 'none'))
print('all points in all rotmod files: %d' % sum(len(v) for v in allg.values()))

q3 = {n for n in allg if T[n]['Q'] == 3}
lowi = {n for n in allg if T[n]['inc'] < 30}
print('Q = 3: %d galaxies; inc < 30: %d; both: %d; removed by either: %d'
      % (len(q3), len(lowi), len(q3 & lowi), len(q3 | lowi)))
keep = {n: p for n, p in allg.items() if n not in q3 | lowi and len(p) >= 3}
short = [n for n, p in allg.items() if n not in q3 | lowi and len(p) < 3]
print('galaxies with fewer than 3 points after the two cuts: %s' % (short or 'none'))
npts = sum(len(p) for p in keep.values())
print('SAMPLE: %d galaxies, %d points' % (len(keep), npts))

CONV = 1.0e6 / 3.0856775814913673e19          # (km/s)^2 / kpc -> m/s^2


def gbar(p, yd=0.5, yb=0.7):
    R, Vo, eV, Vg, Vd, Vb = p
    return (Vg * abs(Vg) + yd * Vd * Vd + yb * Vb * Vb) / R * CONV


pos = sum(1 for ps in keep.values() for p in ps if gbar(p) > 0)
print('points with g_bar > 0 at the prior centres (0.5, 0.7): %d' % pos)


def weight(p):
    R, Vo, eV = p[0], p[1], p[2]
    return 1.0 / (eV * eV + (0.05 * Vo) ** 2)


wtot = sum(weight(p) for ps in keep.values() for p in ps)
print()
for cut in (1.0, 0.5):
    inner = {n: [p for p in ps if p[0] < cut] for n, ps in keep.items()}
    inner = {n: v for n, v in inner.items() if v}
    k = sum(len(v) for v in inner.values())
    kpos = sum(1 for v in inner.values() for p in v if gbar(p) > 0)
    w = sum(weight(p) for v in inner.values() for p in v)
    only = sum(1 for n, v in inner.items() if len(v) == len(keep[n]))
    print('R < %.1f kpc: %4d points (%.1f%% of %d), %3d galaxies hold them, %d of these galaxies lie wholly inside; '
          'g_bar > 0: %d; share of the summed fit weight 1/(eV^2+(0.05V)^2): %.1f%%'
          % (cut, k, 100.0 * k / npts, npts, len(inner), only, kpos, 100.0 * w / wtot))


# --------------------------------------------------------------- the window departure on each galaxy's own field
A0 = 0.97e-10                                  # m/s^2, the scale of the constrained fit (Paper I, Eq. (16))
ELL = 0.2                                      # kpc, the catchment length


def g_obs(ps):
    return [(p[0], p[1] * p[1] / p[0] * CONV) for p in ps]


def g_model(ps, a0=A0):
    """The law-like total field nu_simple(g_bar/a0) g_bar from SPARC's mass models at the prior-centre ratios, a
    smoother profile than the observed curve; points with g_bar <= 0 are skipped."""
    out = []
    for p in ps:
        gb = gbar(p)
        if gb > 0:
            y = gb / a0
            out.append((p[0], gb * 0.5 * (1.0 + math.sqrt(1.0 + 4.0 / y))))
    return out


def g_at(prof, r):
    """Total field from the observed curve: linear interpolation in R, g proportional to R inside the first point,
    g proportional to 1/R beyond the last; the radial component changes sign through the centre."""
    if r < 0:
        return -g_at(prof, -r)
    Rf, gf = prof[0]
    if r <= Rf:
        return gf * r / Rf
    for (Ra, ga), (Rb, gb) in zip(prof, prof[1:]):
        if r <= Rb:
            return ga + (gb - ga) * (r - Ra) / (Rb - Ra)
    Rn, gn = prof[-1]
    return gn * Rn / r


def window_mean(prof, r, half, n=400):
    a, b = r - half, r + half
    h = (b - a) / n
    s = 0.5 * (g_at(prof, a) + g_at(prof, b)) + sum(g_at(prof, a + k * h) for k in range(1, n))
    return s * h / (b - a)


for label, build in (('observed curve V_obs^2/R', g_obs), ('model total field nu_simple(g_bar/a0) g_bar', g_model)):
    print()
    print('Departure of the window mean of the total field from its local value, |<g>/g - 1|, at every point, on the')
    print('%s (linear interpolation; a window one catchment length long, centred on the point, along the radius):' % label)
    dep = []
    for n, ps in keep.items():
        prof = build(ps)
        if len(prof) < 2:
            continue
        for Rp, gp in prof:
            dep.append((Rp, abs(window_mean(prof, Rp, ELL / 2) / gp - 1.0)))
    for lo, hi in ((0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e9)):
        sel = [d for R, d in dep if lo <= R < hi]
        if not sel:
            continue
        sel.sort()
        print('  catchment length %3.0f pc, R in [%.1f, %s) kpc: n = %4d, >1%%: %4d, >10%%: %4d, median %.4f, max %.3f'
              % (ELL * 1000, lo, ('%.1f' % hi) if hi < 1e8 else 'inf', len(sel),
                 sum(1 for d in sel if d > 0.01), sum(1 for d in sel if d > 0.1), sel[len(sel) // 2], sel[-1]))
