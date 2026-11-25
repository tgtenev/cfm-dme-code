"""Shared pieces of the constrained fit (constrained_fit.py) and of the counting-factor menu (counting_factor_menu.py).

The fit machinery is imported unchanged from two modules of this folder: refit_core.py (RC: the SPARC data, the
per-galaxy fits with the mass-to-light ratios, distance and inclination refitted, the ladder in log10 a0, the boost
law's scan in the free volume, the bootstrap weights) and aqual-refit-fit.py (AF: the loader, the Gamma tables of the
full field solution and their correction, the two-constant fit LawFit). This module adds, by the same interface:

  - MOND's simple function, nu(y) = 1/2 + sqrt(1/4 + 1/y), evaluated as two_constant_fit.pl evaluates it
    (nu_simple), and MOND's standard function, mu(x) = x/sqrt(1 + x^2), tabulated as two_constant_fit.pl
    tabulates it (build_table and tab_nu; RC.NuTable here);
  - for the full field solution, the departure Gamma of those two functions from the direct AQUAL solutions of
    constrained_solve.py part gamma, read by the rule of refit_core.py (a Catmull-Rom cubic in log10 a0 at a0/s, s the
    local ratio of the model's Newtonian field to its value at the prior-centre ratios), plus that module's correction
    for the remaining error at other mass-to-light ratios: measured directly for the standard function
    (constrained_solve.py part upsvar), and for the simple function borrowed from the boost law at delta = 0.03, the
    lowest node of that correction;
  - the constrained curve: chi^2 along the scale relation a0 = c H_dS ln(1 + delta) / kappa (Paper I, Eq. (10); at
    kappa = 1 it is Eq. (11)), with H_dS = sqrt(Omega_Lambda) H0, evaluated node by node with every galaxy's
    nuisances refitted, its minimum by a parabola through the lowest node and its two neighbours, and the same rule
    in every bootstrap draw;
  - the sixteen treatments of Paper I §4.1 on SPARC's field.

Law codes: T3, the boost law of Paper I, Eq. (8); T11, the empirical function of McGaugh, Lelli and Schombert,
nu(y) = 1/(1 - exp(-sqrt(y))); SIM and STD, MOND's simple and standard functions.
"""
import csv
import importlib.util
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
_saved_argv = sys.argv
sys.argv = [sys.argv[0]]                 # the refit module reads its options from sys.argv at import
import numpy as np
import refit_core as RC

_spec = importlib.util.spec_from_file_location('aqual_refit_fit', os.path.join(HERE, 'aqual-refit-fit.py'))
AF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AF)
sys.argv = _saved_argv

C_LIGHT = 299792458.0
MPC = 3.0856775814913673e22
OMEGA_L = 0.6847
KAPPA = 1.0
HUBBLE = (67.4, 73.0)
NBOOT = 2000
SEED = 20261114
LA_X = np.array([round(-10.7 + 0.2 * k, 1) for k in range(8)])       # the grid of constrained_solve.py part gamma
GAMMA_X = ['rotation-curves/constrained_solve-gamma-0.csv', 'rotation-curves/constrained_solve-gamma-1.csv']
UPSVAR_X = ['rotation-curves/constrained_solve-upsvar-0.csv', 'rotation-curves/constrained_solve-upsvar-1.csv']


def c_hds(h0):
    """c H_dS in m/s^2 for H0 in km/s/Mpc."""
    return C_LIGHT * math.sqrt(OMEGA_L) * h0 * 1.0e3 / MPC


def a0_of_delta(delta, h0):
    """The scale that the scale relation gives at free volume delta (kappa = 1)."""
    return c_hds(h0) * math.log1p(delta) / KAPPA


def delta_of_a0(a0, h0):
    return math.expm1(KAPPA * a0 / c_hds(h0))


# ------------------------------------------------------------------ the laws in the relation
def nu_simple(y):
    y = np.maximum(y, 1e-300)
    return 0.5 * (1.0 + np.sqrt(1.0 + 4.0 / y))


def mu_standard(x):
    return x / np.sqrt(1.0 + x * x)


NU_STD = RC.NuTable(mu_standard)
LAW_NAMES = {'TIED': 'the constrained fit', 'SIM': "MOND's simple function", 'STD': "MOND's standard function",
             'T11': 'the empirical function', 'T3': 'the two-constant fit'}


def ctx(inp, law, full=False, delta=None, a0=None, gmode='corrected', use_gok=True):
    """A fit context in the machinery's format. law: 'T3' (the boost law at delta; the constrained fit is T3 with a0
    set by the scale relation), 'T11', 'SIM', 'STD'."""
    c = dict(input=inp, full=full, gmode=gmode if full else 'scaled', use_gok=use_gok, _cache={})
    if law == 'T3':
        c['law'] = 'T3'
        if delta is not None:
            c['delta'] = delta
            c['nu'] = RC.law_nu('T3', delta)
    elif law == 'T11':
        c['law'] = 'T11'; c['nu'] = RC.nu_t11
    elif law == 'SIM':
        c['law'] = 'SIM'; c['nu'] = nu_simple
    elif law == 'STD':
        c['law'] = 'STD'; c['nu'] = NU_STD
    else:
        raise ValueError(law)
    if a0 is not None:
        c['a0'] = a0
    return c


# ------------------------------------------------------------------ the dataset with the two added laws
class XDataset(RC.Dataset):
    """RC.Dataset with Gamma tables and corrections for SIM and STD. ctx['sim_gamma'] = 'T3lim' reads SIM's Gamma from
    the boost law's table extrapolated to delta = 0 instead (a cross-check)."""

    def gtab(self, grp, ctx_):
        law = ctx_['law']
        if law in ('SIM', 'STD'):
            src = 'T3lim' if (law == 'SIM' and ctx_.get('sim_gamma') == 'T3lim') else 'own'
            key = ('gtab', id(grp), law, src)
            cache = ctx_.setdefault('_cache', {})
            if key not in cache:
                if src == 'T3lim':
                    cache[key] = (np.tensordot(grp.gam3, RC.delta_weights(0.0), axes=([1], [0])), RC.LA_T3[0], 0.2)
                else:
                    cache[key] = (grp.gamX[law], LA_X[0], 0.2)
            return cache[key]
        return super().gtab(grp, ctx_)

    def eps_corr(self, grp, Yd, Yb, ctx_):
        if ctx_['law'] == 'STD':
            return self._eps_tag(grp, Yd, Yb, grp.eps['STD'])
        if ctx_['law'] == 'SIM':
            return self._eps_tag(grp, Yd, Yb, grp.eps['T3d0.03'])
        return super().eps_corr(grp, Yd, Yb, ctx_)


def attach_x(ds, gals, paths=GAMMA_X):
    """Attach the Gamma tables of SIM and STD (constrained_solve.py part gamma) to the groups. Returns the number of points
    at which a new table is undefined where the tables of aqual-refit-fit.py are defined (excluded from every model, so
    that all models keep seeing the same points), and the largest relative difference of the model's Newtonian field
    between the two solver runs."""
    rows = {}
    for p in paths:
        for r in csv.DictReader(open(p, encoding='utf-8-sig')):
            rows[(r['galaxy'], round(float(r['R']), 4))] = r
    nnew, dmax = 0, 0.0
    for grp in ds.groups:
        if grp.G == 0:
            continue
        keys = [(g['name'], round(float(R), 4)) for g, s in zip(grp.gl, grp.sel) for R in g['R'][s]]
        grp.gamX = {}
        ok = np.ones(len(keys), bool)
        for law in ('SIM', 'STD'):
            T = np.array([[float(rows[k][f'G_{law}_{la:.1f}']) if k in rows else np.nan for la in LA_X] for k in keys])
            ok &= np.all(np.isfinite(T), axis=1)
            grp.gamX[law] = np.where(np.isfinite(T), T, 0.0)
        gN = np.array([float(rows[k]['gN']) * RC.CONV if k in rows else np.nan for k in keys])
        both = grp.gok & np.isfinite(gN) & (grp.gN0 > 0)
        if np.any(both):
            dmax = max(dmax, float(np.max(np.abs(gN[both] / grp.gN0[both] - 1.0))))
        nnew += int(np.sum(grp.gok & ~ok))
        grp.gok = grp.gok & ok
    return nnew, dmax


def upsvar_x_rows(paths=UPSVAR_X):
    out = []
    for p in paths:
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p, encoding='utf-8-sig')):
            for k, v in r.items():
                if not k.startswith('STD_a') or v == '':
                    continue
                a0s, rest = k[5:].split('_d')
                sd, sb = rest.split('_b')
                out.append(dict(galaxy=r['galaxy'], R=r['R'], a0=a0s, dlogYd=sd, dlogYb=sb, Gamma=v))
    return out


def attach_eps_std(ds, rows):
    """The correction of refit_core.py (RC.Dataset.attach_eps, repeated here for the tag STD): the error of the scaled
    reading against the direct solutions at other ratios, per point, on the disk grid and, for bulge galaxies, on the
    grid of disk by bulge ratios."""
    D = {}
    ref = None
    for r in rows:
        D[(r['galaxy'], round(float(r['R']), 4), round(float(r['dlogYd']), 2), round(float(r['dlogYb']), 2))] = float(r['Gamma'])
        ref = float(r['a0']) if ref is None else ref
    for grp in ds.groups:
        if grp.G == 0:
            continue
        keys = [(g['name'], round(float(R), 4)) for g, s in zip(grp.gl, grp.sel) for R in g['R'][s]]
        c = dict(input='model', law='STD', a0=ref, gmode='scaled', full=True, _cache={})

        def err_at(du, dv):
            U = np.full((grp.G, 1), RC.LGY0 + du); V = np.full((grp.G, 1), RC.LGYB0 + dv)
            Yd = 10.0 ** U[grp.gid]; Yb = 10.0 ** V[grp.gid] if grp.two else 0.0
            Gd = np.array([D.get((k[0], k[1], round(du, 2), round(dv, 2)), np.nan) for k in keys])
            if not np.any(np.isfinite(Gd)):
                return None
            Gs = ds.gamma(grp, Yd, Yb, c)[:, 0]
            return np.where(np.isfinite(Gd) & grp.gok, Gd - Gs, 0.0)
        E = {}
        col = {0.0: np.zeros(len(keys))}
        for du in RC.EPS_D:
            if abs(du) > 1e-9:
                e = err_at(float(du), 0.0)
                if e is not None:
                    col[round(float(du), 1)] = e
        lo = 0.0
        while round(lo - 0.2, 1) in col:
            lo = round(lo - 0.2, 1)
        hi = 0.0
        while round(hi + 0.2, 1) in col:
            hi = round(hi + 0.2, 1)
        E['drange'] = (lo, hi)
        E['d1'] = np.stack([col[round(lo + 0.2 * k, 1)] for k in range(int(round((hi - lo) / 0.2)) + 1)], axis=1)
        if grp.two:
            for tg, dv in (('bp', 0.2), ('bm', -0.2)):
                e = err_at(0.0, dv)
                E[tg] = e if e is not None else np.zeros(len(keys))
            T2, ok = np.zeros((len(keys), len(RC.EPS_D2), len(RC.EPS_B))), True
            for i, du in enumerate(RC.EPS_D2):
                for j, dv in enumerate(RC.EPS_B):
                    if abs(dv) < 1e-9:
                        if round(float(du), 1) in col:
                            T2[:, i, j] = col[round(float(du), 1)]
                        else:
                            ok = False
                        continue
                    e = err_at(float(du), float(dv))
                    if e is None:
                        ok = False
                    else:
                        T2[:, i, j] = e
            if ok:
                E['d2'] = T2
        grp.eps['STD'] = E
    return ref


# ------------------------------------------------------------------ the constrained curve
class ConstrainedLine:
    """chi^2 along a0 = c H_dS ln(1 + delta) / kappa for the delta nodes given, every galaxy's nuisances refitted at
    each node. The per-galaxy objectives are stored for the bootstrap."""

    def __init__(self, ds, inp, full, h0, deltas, gmode='corrected', kappa=KAPPA):
        self.ds, self.inp, self.full, self.h0, self.kappa = ds, inp, full, h0, kappa
        self.d = np.array([round(float(x), 4) for x in deltas])
        self.fits = []
        for d in self.d:
            a0 = c_hds(h0) * math.log1p(float(d)) / kappa
            self.fits.append(ds.fit(ctx(inp, 'T3', full=full, delta=float(d), a0=a0, gmode=gmode)))
        self.M = np.array([f['F'] for f in self.fits])        # (nodes, galaxies)
        self.curve = self.M.sum(axis=1)
        self.chi2, self.dmin, self.edge, self.sig = self.minimum()

    def minimum(self, w=None):
        """(chi^2 at the minimum, delta there, minimum on an end of the grid, the parabola's Delta chi^2 = 1 half-width)."""
        y = self.curve if w is None else self.M @ w
        i = int(np.argmin(y))
        if i == 0 or i == len(y) - 1:
            return float(y[i]), float(self.d[i]), True, float('nan')
        r = RC.parab3([(self.d[i - 1], y[i - 1]), (self.d[i], y[i]), (self.d[i + 1], y[i + 1])])
        if r is None:
            return float(y[i]), float(self.d[i]), False, float('nan')
        return float(r[2]), float(r[0]), False, float(r[1])

    def at(self, d):
        k = int(np.argmin(np.abs(self.d - d)))
        if abs(self.d[k] - d) > 1e-9:
            raise KeyError(d)
        return float(self.curve[k]), self.fits[k]


def ladder(ds, inp, law, full, la_start, gmode='corrected', sim_gamma=None):
    base = ctx(inp, law, full=full, gmode=gmode)
    if sim_gamma:
        base['sim_gamma'] = sim_gamma
    return RC.Ladder(ds, base, la_start)


def two_constant(ds, inp, full, la_start):
    base = AF.ctx_of(inp, 'T3', full=True, gmode='corrected') if full else AF.ctx_of(inp, 'T3')
    return AF.LawFit(ds, base, la_start)


def sd_of(x):
    return float(np.std(np.asarray(x), ddof=1))


def pct(x, p):
    return float(np.percentile(np.asarray(x), p))


# ------------------------------------------------------------------ the sixteen treatments of Paper I §4.1
# (cell, mass-to-light treatment, distances and inclinations, sample)
TREATMENTS = [
    ('c01-baseline', 'standard', 'marginalized', 'all'), ('c02-locked', 'locked', 'marginalized', 'all'),
    ('c03-pinned', 'pinned', 'marginalized', 'all'), ('c04-catalog', 'standard', 'catalog', 'all'),
    ('c05-i45', 'standard', 'marginalized', 'i45'), ('c06-wide', 'wide', 'marginalized', 'all'),
    ('c07-pinned-catalog', 'pinned', 'catalog', 'all'), ('c08-locked-catalog', 'locked', 'catalog', 'all'),
    ('c09-wide-catalog', 'wide', 'catalog', 'all'), ('c10-locked-i45', 'locked', 'marginalized', 'i45'),
    ('c11-pinned-i45', 'pinned', 'marginalized', 'i45'), ('c12-catalog-i45', 'standard', 'catalog', 'i45'),
    ('c13-wide-i45', 'wide', 'marginalized', 'i45'), ('c14-pinned-catalog-i45', 'pinned', 'catalog', 'i45'),
    ('c15-locked-catalog-i45', 'locked', 'catalog', 'i45'), ('c16-wide-catalog-i45', 'wide', 'catalog', 'i45'),
]
PRIOR_OF = {'standard': 0.10, 'locked': 0.10, 'pinned': 0.01, 'wide': 0.25}
BRAT = 1.4                               # the locked bulge, Yb = 1.4 Yd (two_constant_fit.pl --bulge=fixed)


def treatment_dataset(gals, ml, nuis, sample):
    """The dataset of one treatment on SPARC's field. ml: standard (both ratios free under 0.10 dex priors), locked (the
    bulge at 1.4 times the disk, as two_constant_fit.pl --bulge=fixed, with the disk prior only), pinned (both priors
    0.01 dex), wide (both priors 0.25 dex); nuis: marginalized (distance and inclination fitted under their priors) or
    catalog (both at the catalog values, q = 1); sample: all (the 153 galaxies) or i45 (inclination >= 45 deg).
    Sets RC.PRIOR, which the caller restores."""
    RC.PRIOR = PRIOR_OF[ml]
    if ml == 'locked':
        g2 = {}
        for n, g in gals.items():
            h = dict(g)
            if g['hasb']:
                h['Vd'] = np.sqrt(g['Vd'] ** 2 + BRAT * g['Vb'] ** 2)
                h['Vb'] = np.zeros_like(g['Vb'])
                h['hasb'] = False
            g2[n] = h
        gals = g2
    names = sorted(gals) if sample == 'all' else [n for n in sorted(gals) if gals[n]['ic'] >= 45.0]
    return XDataset(gals, names=names, frozen=(nuis == 'catalog'))
