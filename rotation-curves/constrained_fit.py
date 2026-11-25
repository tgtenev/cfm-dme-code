"""The constrained fit: the boost law of Paper I, Eq. (8), with its scale set by its free volume through the scale
relation, fitted to the 153 SPARC rotation curves and set beside MOND's simple and standard functions and the empirical
function at equal freedom, and beside the two-constant fit.

The scale relation is Paper I's Eq. (10), a0 = c H_dS ln(1 + delta) / kappa, at kappa = 1 (Eq. (11)), with
H_dS = sqrt(Omega_Lambda) H0, Omega_Lambda = 0.6847, at H0 = 67.4 and 73 km/s/Mpc. The law then has one fitted
constant, the free volume delta, or equivalently the scale a0. Law codes: T3, the boost law of Paper I, Eq. (8); T11,
the empirical function of McGaugh, Lelli and Schombert, nu(y) = 1/(1 - exp(-sqrt(y))); SIM and STD, MOND's simple and
standard functions; BCF, the sharp knee of Burton, Cabrera and Frank; T4, the square-root onset form of Paper I §4.2.

It supports Paper I §4.1 (the scale of the empirical function under other objectives), §4.2 with Eq. (16) and
Table 2, §4.3 (the weak-field normalization) and §5.3 (Verlinde's scale against the fitted scales), and Paper II §2,
§5.2 and §6.1 (the constrained fit on the full field solution).

Parts (all by default, in this order; --parts repro,main,treat,scales,exact selects):
  repro    the numbers of two_constant_fit.pl (two_constant_fit-results.txt) that this machinery must reproduce.
  main     four setups: SPARC's tabulated field with the algebraic relation (the fit of record), the model's own
           Newtonian field of disks with thickness with the relation, and both fields with the full field solution.
           For each: chi^2 along the constrained curve at both Hubble constants (delta 0.05 to 0.45 in steps of 0.01,
           every galaxy's nuisances refitted at every node), its minimum, the galaxy-clustered 68 percent range of the
           minimum over 2,000 bootstrap draws by galaxy (seed 20261114), the scale and the divisor there, chi^2 at
           delta = 0.19 and 0.17; the scale relation at kappa = 1/3; the comparison with the simple, standard and
           empirical functions, each with its scale fitted by the ladder in log10 a0 of two_constant_fit.pl, and with
           the two-constant fit, with paired differences and their galaxy-clustered standard deviations. On the fit of
           record it adds the square-root onset form, the scatter, the binned residuals, the match to the empirical
           function, Verlinde's scale and the weak-field normalization. It then sets the full field solution against
           the relation. Writes constrained_fit-optimum.csv (the fitted nuisances at the optima of the full field
           solution, for constrained_solve.py part atopt) and constrained_fit-valley.csv (the constrained curves).
  treat    the sixteen treatments of Paper I §4.1 on SPARC's field with the relation.
  scales   the scale of the empirical function under other objectives and cuts (Paper I §4.1), and the product
           D ln(1 + delta) and the divisor of the two-constant fit (Paper I §4.2).
  exact    the exact re-solve at the optima of the full field solution (needs constrained_solve-atopt-*.csv).

Paper I's Figures 5 and 6 are drawn from constrained_fit-valley.csv (chi^2 along the constrained curves at kappa = 1
and 1/3, both Hubble constants, the four setups) and from the node table of the two-constant fit in
two_constant_fit-results.txt (at each free-volume node, a0 with its statistical error and chi^2). The figures and the
script that draws them are not part of this repository. In constrained_fit-optimum.csv and constrained_solve-atopt-*.csv
the rows of the constrained fit at H0 = 67.4 carry the tag constrained, those of the other laws their codes.

Inputs: data/sparc/ and rotation-curves/aqual-disk-points.csv (through aqual-refit-fit.py), aqual-refit-gamma-*.csv
and aqual-refit-upsvar*.csv (the boost law and the empirical function on the full field solution),
constrained_solve-gamma-*.csv and constrained_solve-upsvar-*.csv (the simple and standard functions), and for part
exact constrained_fit-optimum.csv and constrained_solve-atopt-*.csv, all in rotation-curves/.

Run from the repository root, in this order: constrained_solve.py parts gamma and upsvar; this script with
--parts main and --out pointing to a scratch file, which writes constrained_fit-optimum.csv; constrained_solve.py part
atopt, which reads it; then the whole run,
    python rotation-curves/constrained_fit.py --out rotation-curves/constrained_fit-results.txt
The fits are deterministic, so the whole run writes the same optimum file again. The whole run took 2,290 s: 4 s for
repro, about 1,600 s for main (about 130 s for each setup with the relation and 650 to 700 s for each with the full
field solution), about 670 s for treat, about 15 s for scales and under 1 s for exact. Run alone, --parts scales first
runs the setup of the fit of record and takes about 150 s; --parts repro,exact takes about 10 s. The options
--setups sparc-relation,model-full and --cells c01-baseline,c04-catalog restrict a run, for testing only.
"""
import csv
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import constrained_core as CC

RC, AF = CC.RC, CC.AF
ARGV = sys.argv[1:]
T00 = time.time()
_lines = []


def arg(name, default=None):
    return ARGV[ARGV.index(name) + 1] if name in ARGV else default


def out(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def stamp(lab):
    out(f'   [{lab}: {time.time() - T00:.0f} s]')


DELTAS = [round(0.05 + 0.01 * k, 2) for k in range(41)]           # 0.05 to 0.45
DELTAS_K3 = [round(0.02 + 0.0025 * k, 4) for k in range(41)]      # kappa = 1/3: 0.02 to 0.12
DELTAS_T = [round(0.03 + 0.01 * k, 2) for k in range(48)]         # the treatments: 0.03 to 0.50
OPT_PATH = 'rotation-curves/constrained_fit-optimum.csv'
VALLEY_PATH = 'rotation-curves/constrained_fit-valley.csv'
ATOPT = [f'rotation-curves/constrained_solve-atopt-{k}.csv' for k in range(3)]
TAG67 = 'constrained'                    # the tag of the constrained fit at H0 = 67.4 in the optimum and atopt files
SETUPS = [('sparc', False, "SPARC'S TABULATED FIELD, THE ALGEBRAIC RELATION (THE FIT OF RECORD)"),
          ('model', False, "THE MODEL'S OWN NEWTONIAN FIELD OF DISKS WITH THICKNESS, THE ALGEBRAIC RELATION"),
          ('sparc', True, "SPARC'S TABULATED FIELD, THE FULL FIELD SOLUTION"),
          ('model', True, "THE MODEL'S OWN NEWTONIAN FIELD OF DISKS WITH THICKNESS, THE FULL FIELD SOLUTION")]
LA_START = {(False, 'SIM'): 1.08e-10, (False, 'STD'): 1.21e-10, (False, 'T11'): 1.05e-10, (False, 'BCF'): 1.24e-10,
            (True, 'SIM'): 0.97e-10, (True, 'STD'): 1.10e-10, (True, 'T11'): 0.94e-10}
NU_BCF = RC.NuTable(lambda x: x * np.tanh(1.0 / np.maximum(x, 1e-300)))
RES = {}                                                          # results kept for the later parts
_NU_T4 = {}


def nu_t4(xs):
    """The square-root onset form of two_constant_fit.pl (mu_sqrt, its section E): mu = b/(1 + b),
    b = X sqrt(1 + (X/X_sat)^2), tabulated as two_constant_fit.pl tabulates it."""
    k = round(float(xs), 6)
    if k not in _NU_T4:
        _NU_T4[k] = RC.NuTable(lambda X: (lambda b: b / (1.0 + b))(X * np.sqrt(1.0 + (X / k) ** 2)))
    return _NU_T4[k]


class OnsetScan(RC.Scan):
    """Section E of two_constant_fit.pl: a ladder in log10 a0 at each X_sat node, the minimum by the parabola through
    the three lowest nodes."""

    def ctx(self, d):
        c = dict(self.base); c['law'] = 'T4'; c['xsat'] = d; c['nu'] = nu_t4(d); c['_cache'] = {}
        return c


def fmt_sd(d, sd):
    return f'{d:+8.1f} +/- {sd:5.1f} ({d / sd:+.2f} SD)' if sd > 0 else f'{d:+8.1f}'


# ====================================================================== loading
def load():
    T1, gals = AF.load()
    ds = CC.XDataset(gals)
    ds.attach_eps(AF.upsvar_rows())
    nnew, dmax = CC.attach_x(ds, gals)
    ref = CC.attach_eps_std(ds, CC.upsvar_x_rows())
    return T1, gals, ds, nnew, dmax, ref


# ====================================================================== part repro
def part_repro(ds):
    out("REPRODUCTION OF two_constant_fit.pl's NUMBERS (SPARC's field, the algebraic relation, the machinery of refit_core.py unchanged)")
    src1 = 'two_constant_fit-results.txt, the boost law refitted at the minimum of its scan in delta_fv'
    src2 = 'two_constant_fit-results.txt, the table of the one-constant fits with chi^2 to three decimals'
    rows = [('the boost law at delta = 0.2953, a0 = 0.92459e-10', CC.ctx('sparc', 'T3', delta=0.2953, a0=0.92459e-10), 4319.22, src1),
            ('the empirical function at a0 = 1.0475e-10', CC.ctx('sparc', 'T11', a0=1.0475e-10), 4326.672, src2),
            ("MOND's simple function at a0 = 1.0796e-10", CC.ctx('sparc', 'SIM', a0=1.0796e-10), 4350.303, src2),
            ("MOND's standard function at a0 = 1.2144e-10", CC.ctx('sparc', 'STD', a0=1.2144e-10), 4491.176, src2),
            ('the sharp knee of Burton, Cabrera and Frank at a0 = 1.2395e-10',
             dict(CC.ctx('sparc', 'T11', a0=1.2395e-10), law='BCF', nu=NU_BCF), 4594.748, src2)]
    for lab, c, rec, src in rows:
        out(f'   {lab}: chi^2 {ds.fit(c)["F"].sum():.3f}; two_constant_fit.pl {rec} ({src})')
    stamp('repro')


# ====================================================================== the pieces of one setup
def ladder_fit(ds, inp, full, law, sim_gamma=None):
    if law == 'BCF':
        base = dict(CC.ctx(inp, 'T11'), law='BCF', nu=NU_BCF)
        return RC.Ladder(ds, base, math.log10(LA_START[(False, 'BCF')]))
    return CC.ladder(ds, inp, law, full, math.log10(LA_START[(full, law)]), sim_gamma=sim_gamma)


def boot_line(T, W):
    r = [T.minimum(w) for w in W]
    return np.array([x[0] for x in r]), np.array([x[1] for x in r]), np.array([x[2] for x in r])


def describe_line(T, W, kappa, lab):
    ch_b, d_b, e_b = boot_line(T, W)
    a0 = CC.c_hds(T.h0) * math.log1p(T.dmin) / kappa
    a0_b = CC.c_hds(T.h0) * np.log1p(d_b) / kappa
    out(f'   {lab}: constrained minimum delta = {T.dmin:.4f} (Delta chi^2 = 1 half-width {T.sig:.4f}){" AT AN END OF THE GRID" if T.edge else ""}, '
        f'chi^2 = {T.chi2:.2f}; scale a0 = {a0 * 1e10:.4f}e-10 m/s^2; divisor c H_dS/a0 = {kappa / math.log1p(T.dmin):.3f}')
    out(f'      bootstrap ({len(W)} draws): delta 16-84 percent {CC.pct(d_b, 16):.4f} to {CC.pct(d_b, 84):.4f}, median {np.median(d_b):.4f}, '
        f'SD {CC.sd_of(d_b):.4f}, 2.5-97.5 percent {CC.pct(d_b, 2.5):.4f} to {CC.pct(d_b, 97.5):.4f}; draws at an end of the grid {int(e_b.sum())}')
    out(f'      the scale over the draws: 16-84 percent {CC.pct(a0_b, 16) * 1e10:.4f} to {CC.pct(a0_b, 84) * 1e10:.4f}e-10, SD {CC.sd_of(a0_b) * 1e10:.4f}e-10; '
        f'divisor 16-84 percent {kappa / math.log1p(CC.pct(d_b, 84)):.3f} to {kappa / math.log1p(CC.pct(d_b, 16)):.3f}')
    return ch_b, d_b, a0_b


def run_setup(ds, inp, full, title, W):
    out('')
    out('=' * 118)
    out(title)
    out('=' * 118)
    t0 = time.time()
    key = (inp, full)
    R = RES[key] = {}
    # 1. the constrained curve at both Hubble constants
    out('1. THE CONSTRAINED CURVE a0 = c H_dS ln(1 + delta), kappa = 1 (every galaxy\'s mass-to-light ratios, distance and inclination refitted at every node)')
    for h0 in CC.HUBBLE:
        T = CC.ConstrainedLine(ds, inp, full, h0, DELTAS)
        R[('line', h0)] = T
        ch_b, d_b, a0_b = describe_line(T, W, 1.0, f'H0 = {h0} km/s/Mpc (c H_dS = {CC.c_hds(h0) * 1e10:.4f}e-10 m/s^2)')
        R[('line_b', h0)] = (ch_b, d_b, a0_b)
        dstar = round(T.dmin, 4)
        c = CC.ctx(inp, 'T3', full=full, delta=dstar, a0=CC.a0_of_delta(dstar, h0))
        r = ds.fit(c)
        R[('line_at_min', h0)] = (r, c)
        out(f'      chi^2 evaluated directly at delta = {dstar:.4f}: {r["F"].sum():.2f} (the parabola through the three lowest nodes gives {T.chi2:.2f}; '
            f'difference {r["F"].sum() - T.chi2:+.3f})')
        out(f'      chi^2 at delta = 0.19: {T.at(0.19)[0]:.2f} (a0 = {CC.a0_of_delta(0.19, h0) * 1e10:.4f}e-10); at delta = 0.17: {T.at(0.17)[0]:.2f} '
            f'(a0 = {CC.a0_of_delta(0.17, h0) * 1e10:.4f}e-10)')
        out(f'      chi^2 minus the constrained minimum: at 0.19 {T.at(0.19)[0] - T.chi2:+.2f}, at 0.17 {T.at(0.17)[0] - T.chi2:+.2f}')
    stamp(f'{inp}, {"full" if full else "relation"}: constrained curve')
    # the curves, for the valley figure
    T67, T73 = R[('line', 67.4)], R[('line', 73.0)]
    out('   the constrained curves: delta | a0 at 67.4 (1e-10) | chi^2 at 67.4 | a0 at 73 (1e-10) | chi^2 at 73')
    for i, d in enumerate(T67.d):
        out(f'      {d:.2f} | {CC.a0_of_delta(d, 67.4) * 1e10:.4f} | {T67.curve[i]:8.2f} | {CC.a0_of_delta(d, 73.0) * 1e10:.4f} | {T73.curve[i]:8.2f}')
    # the scale relation at kappa = 1/3
    out('   the scale relation at kappa = 1/3, a0 = 3 c H_dS ln(1 + delta) (the whole set point as the mean threshold), delta 0.02 to 0.12 in steps of 0.0025:')
    for h0 in CC.HUBBLE:
        K = CC.ConstrainedLine(ds, inp, full, h0, DELTAS_K3, kappa=1.0 / 3.0)
        R[('k3', h0)] = K
        kb, db, ab = describe_line(K, W, 1.0 / 3.0, f'kappa = 1/3, H0 = {h0}')
        tb = R[('line_b', h0)][0]
        out(f'      kappa = 1/3 minus kappa = 1 at their minima: {fmt_sd(K.chi2 - R[("line", h0)].chi2, CC.sd_of(kb - tb))}; '
            f'share of draws in which kappa = 1/3 fits better {np.mean(kb < tb):.3f}')
    stamp(f'{inp}, {"full" if full else "relation"}: kappa 1/3')
    # 2. the competitors at equal freedom, and the two-constant fit
    out('2. AT EQUAL FREEDOM: ONE FITTED GLOBAL CONSTANT EACH (the scale by the ladder in log10 a0 of two_constant_fit.pl, every galaxy\'s nuisances refitted)')
    laws = ['SIM', 'STD', 'T11'] + ([] if full else ['BCF'])
    for law in laws:
        L = ladder_fit(ds, inp, full, law)
        R[law] = L
        R[(law, 'b')] = np.array([L.node_min(w)[0] for w in W])
    if full:
        L = ladder_fit(ds, inp, full, 'SIM', sim_gamma='T3lim')
        R['SIM_T3lim'] = L
        out(f'   cross-check: the simple function with Gamma from the boost law\'s table at delta = 0 (quartic extrapolated): '
            f'a0 = {10 ** L.la * 1e10:.4f}e-10, chi^2 = {L.chi2:.2f}, against {10 ** R["SIM"].la * 1e10:.4f}e-10 and {R["SIM"].chi2:.2f} with its own table')
    two = CC.two_constant(ds, inp, full, math.log10(0.95e-10))
    R['T3'] = two
    R[('T3', 'b')] = [two.minw(w) for w in W]
    if not full and inp == 'sparc':
        S4 = OnsetScan(ds, CC.ctx('sparc', 'T11'), [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0], math.log10(1.02e-10), extend=False)
        R['T4'] = S4
        R[('T4', 'b')] = [S4.scan_min(w) for w in W]
        xb = np.array([x[1] for x in R[('T4', 'b')]])
        t4b = np.array([x[0] for x in R[('T4', 'b')]])
        tb_ = R[('line_b', 67.4)][0]
        out(f'   the square-root onset form, X_sat and a0 free (two_constant_fit.pl section E): X_sat = {S4.dmin:.4f} (stat {S4.sig:.4f}; clustered SD '
            f'{CC.sd_of(xb):.4f}, 16-84 percent {CC.pct(xb, 16):.3f} to {CC.pct(xb, 84):.3f}), a0 = {10 ** S4.la * 1e10:.4f}e-10, chi^2 = {S4.chi2:.2f} '
            f'(two_constant_fit.pl 3.3024, 1.0308e-10, 4314.09; two_constant_fit-results.txt section E)')
        out(f'      the constrained fit (67.4) minus the onset form: {fmt_sd(R[("line", 67.4)].chi2 - S4.chi2, CC.sd_of(tb_ - t4b))}; the two-constant '
            f'fit minus the onset form: {fmt_sd(two.chi2 - S4.chi2, CC.sd_of(np.array([x[0] for x in R[("T3", "b")]]) - t4b))}')
        for d_ in (0.19, R[('line', 67.4)].dmin):
            on = 0.6573 / math.log1p(d_)
            out(f'      the constrained fit\'s onset at delta {d_:.4f}: X = {on:.3f}, against X_sat {S4.dmin:.3f}: {(on - S4.dmin) / CC.sd_of(xb):+.2f} clustered SD')
    stamp(f'{inp}, {"full" if full else "relation"}: competitors')
    out('   law | fitted constant | scale a0 (1e-10 m/s^2) | chi^2 | the constrained fit (H0 = 67.4) minus the law, galaxy-clustered SD | '
        'share of draws in which the constrained fit is better')
    tb67 = R[('line_b', 67.4)][0]
    tb73 = R[('line_b', 73.0)][0]
    for h0, Tt, tb in ((67.4, T67, tb67), (73.0, T73, tb73)):
        out(f'   the constrained fit, H0 = {h0} | delta = {Tt.dmin:.4f} | {CC.a0_of_delta(Tt.dmin, h0) * 1e10:.4f} | {Tt.chi2:.2f} | '
            + ('(itself)' if h0 == 67.4 else f'{fmt_sd(T67.chi2 - Tt.chi2, CC.sd_of(tb67 - tb))}') + ' |')
    for law in laws:
        L = R[law]
        lb = R[(law, 'b')]
        out(f'   {CC.LAW_NAMES.get(law, "the sharp knee of Burton, Cabrera and Frank")} | none | {10 ** L.la * 1e10:.4f} | {L.chi2:.2f} | '
            f'{fmt_sd(T67.chi2 - L.chi2, CC.sd_of(tb67 - lb))} | {np.mean(tb67 < lb):.3f}')
    t3b = np.array([x[0] for x in R[('T3', 'b')]])
    stop = " (the scan's lower stop)" if two.edge else ''
    out(f'   the two-constant fit (reference) | delta = {two.delta:.4f}{stop} | '
        f'{10 ** two.la * 1e10:.4f} | {two.chi2:.2f} | {fmt_sd(T67.chi2 - two.chi2, CC.sd_of(tb67 - t3b))} | {np.mean(tb67 < t3b):.3f}')
    out('   the same differences for the constrained fit at H0 = 73: '
        + '; '.join(f'minus {CC.LAW_NAMES.get(l, "the sharp knee")}: {fmt_sd(T73.chi2 - R[l].chi2, CC.sd_of(tb73 - R[(l, "b")]))}' for l in laws)
        + f'; minus the two-constant fit: {fmt_sd(T73.chi2 - two.chi2, CC.sd_of(tb73 - t3b))}')
    out('   pairwise among the competitors (first minus second): '
        + '; '.join(f'{a} - {b}: {fmt_sd(R[a].chi2 - R[b].chi2, CC.sd_of(R[(a, "b")] - R[(b, "b")]))}'
                    for a, b in (('STD', 'SIM'), ('SIM', 'T11'), ('STD', 'T11')) + ((('BCF', 'SIM'),) if not full else ()))
        + f'; STD - two-constant: {fmt_sd(R["STD"].chi2 - two.chi2, CC.sd_of(R[("STD", "b")] - t3b))}'
        + f'; SIM - two-constant: {fmt_sd(R["SIM"].chi2 - two.chi2, CC.sd_of(R[("SIM", "b")] - t3b))}')
    stamp(f'{inp}, {"full" if full else "relation"}')
    return R


# ====================================================================== the fit of record at the constrained minimum
def baseline_extras(ds, gals):
    """Scatter, binned residuals and the match to the empirical function at the constrained minimum, by the definitions
    of two_constant_fit.pl (scatter_dex, the bins of its section D2, the comparison of its section H); Verlinde's scale;
    the derived quantities of Paper I §§3.5 and 4.2; and the weak-field normalization of Paper I §4.3."""
    R = RES[('sparc', False)]
    out('')
    out('3. THE FIT OF RECORD AT THE CONSTRAINED MINIMUM: SCATTER, BINNED RESIDUALS, THE DEEP LIMB AND THE MATCH TO THE EMPIRICAL FUNCTION')

    def resid(fit_res, c):
        P = ds.points(c, fit_res)
        r = np.concatenate([P[n]['r'][P[n]['valid']] for n in ds.names])
        gb = np.concatenate([P[n]['gbar'][P[n]['valid']] for n in ds.names])
        return r, gb
    T = R[('line', 67.4)]
    rt, ct = R[('line_at_min', 67.4)]
    cases = [('the constrained fit (H0 = 67.4)', rt, ct)]
    two = R['T3']
    cases.append((f'the two-constant fit (delta {two.delta:.4f})', two.best, two.ctx()))
    for law in ('T11', 'SIM', 'STD'):
        L = R[law]
        cases.append((CC.LAW_NAMES[law], L.best, dict(L.base, a0=10 ** L.la, _cache={})))
    store = {}
    for lab, res, c in cases:
        r, gb = resid(res, c)
        store[lab] = (r, gb)
        out(f'   scatter about {lab}: {np.std(r):.4f} dex over {len(r)} points (two_constant_fit.pl: boost law 0.1101, empirical function 0.1103, simple 0.1105, '
            f'standard 0.1103; two_constant_fit-results.txt section J)')
    out('   binned mean residual log10(g_obs/q^2) - log10 g_model in 0.25-dex bins of log10 g_bar at each law\'s own fit (the bins of '
        'two_constant_fit.pl section D2):')
    rt_, gt_ = store['the constrained fit (H0 = 67.4)']
    rs_, gs_ = store[CC.LAW_NAMES['STD']]
    bt = np.floor((np.log10(gt_) + 12.6) / 0.25).astype(int)
    bs = np.floor((np.log10(gs_) + 12.6) / 0.25).astype(int)
    big_t, big_s = (0.0, ''), (0.0, '')
    for b in sorted(set(bt)):
        nt, ns = int(np.sum(bt == b)), int(np.sum(bs == b))
        if nt < 8 or ns < 8:
            continue
        mt, ms = float(np.mean(rt_[bt == b])), float(np.mean(rs_[bs == b]))
        lo = -12.6 + 0.25 * b
        out(f'      [{lo:6.2f},{lo + 0.25:6.2f})  constrained {nt:4d} {mt:+.4f}   standard {ns:4d} {ms:+.4f}')
        if abs(mt) > abs(big_t[0]):
            big_t = (mt, f'{lo:.2f}')
        if abs(ms) > abs(big_s[0]):
            big_s = (ms, f'{lo:.2f}')
    out(f'      largest |bin offset|: the constrained fit {abs(big_t[0]):.4f} dex (bin from {big_t[1]}), the standard function {abs(big_s[0]):.4f} dex '
        f'(bin from {big_s[1]}); two_constant_fit.pl at the two-constant fit 0.0593 and 0.0777 (two_constant_fit-results.txt section D2)')
    d = T.dmin
    lam = math.log1p(d)
    ly = np.arange(-2.0, math.log10(32.0) + 1e-9, 0.01)
    for dv in (round(d, 4), 0.19, 0.2953):
        dd = np.log10(RC.law_nu('T3', dv)(10 ** ly)) - np.log10(RC.nu_t11(10 ** ly))
        out(f'   the boost law at delta {dv:.4f} against the empirical function over 0.01 <= y <= 32 (same y): rms {np.sqrt(np.mean(dd ** 2)):.4f} dex, '
            f'largest {np.max(np.abs(dd)):.4f} dex (two_constant_fit.pl at 0.2953: 0.0135 and 0.0245, two_constant_fit-results.txt section H)')
    a0b = R[('line_b', 67.4)][2]
    for h0 in CC.HUBBLE:
        aV = CC.C_LIGHT * h0 * 1e3 / CC.MPC / 6.0
        Tt = R[('line', h0)]
        at_ = CC.a0_of_delta(Tt.dmin, h0)
        sd_ = CC.sd_of(R[('line_b', h0)][2])
        out(f'   Verlinde\'s c H0/6 at H0 = {h0}: {aV * 1e10:.4f}e-10 (divisor {CC.c_hds(h0) / aV:.3f} against c H_dS); against the constrained scale '
            f'{at_ * 1e10:.4f}e-10 (clustered SD {sd_ * 1e10:.4f}): {(aV - at_) / sd_:+.2f} SD; against the empirical function\'s '
            f'{10 ** R["T11"].la * 1e10:.4f}e-10 (clustered SD {CC.sd_of(10 ** np.array([R["T11"].node_min(w)[1] for w in W_ALL])) * 1e10:.4f}): '
            f'{(aV - 10 ** R["T11"].la) / CC.sd_of(10 ** np.array([R["T11"].node_min(w)[1] for w in W_ALL])):+.2f} SD')
    for dv in (0.19, 0.17, d, 0.2953):
        lm = math.log1p(dv)
        s = lm / dv
        out(f'   at delta {dv:.4f}: lambda {lm:.4f}, dilute slope s = {s:.4f}, deep-limb factor s^-1/2 = {s ** -0.5:.4f}, onset 0.6573/lambda = '
            f'{0.6573 / lm:.3f}, guided share at the crossover delta/(1 + delta) = {dv / (1 + dv):.3f}, amplification A = 2/delta^2 = {2 / dv ** 2:.1f}, '
            f'deep normalization a0/s at the constrained scale (67.4) = {CC.a0_of_delta(dv, 67.4) / s * 1e10:.4f}e-10')
    # the binned relation below 1e-11 m/s^2 at the prior-centre ratios and the catalog distances, in 0.2-dex bins of log10 g_bar
    gb_all, go_all = [], []
    for n in ds.names:
        g = gals[n]
        gbar = (g['Vg'] * np.abs(g['Vg']) + 0.5 * g['Vd'] ** 2 + 0.7 * g['Vb'] ** 2) / g['R'] * RC.CONV
        gobs = g['Vo'] ** 2 / g['R'] * RC.CONV
        k = (gbar > 0) & (gobs > 0)
        gb_all.append(gbar[k]); go_all.append(gobs[k])
    gb_all, go_all = np.concatenate(gb_all), np.concatenate(go_all)
    lg = np.log10(gb_all)
    edges = np.arange(-12.4, -8.39, 0.2)
    for dv, a0, lab in ((0.2953, 0.9246e-10, 'the two-constant fit (0.2953, 0.9246e-10)'),
                        (round(d, 4), CC.a0_of_delta(round(d, 4), 67.4), f'the constrained fit ({d:.4f}, {CC.a0_of_delta(d, 67.4) * 1e10:.4f}e-10)'),
                        (0.19, CC.a0_of_delta(0.19, 67.4), 'the scale relation at the fiducial free volume (0.19, 0.9426e-10)')):
        nuf = RC.law_nu('T3', dv)
        gl = nuf(gb_all / a0) * gb_all
        med = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (lg >= a) & (lg < b)
            if m.sum() >= 10 and b <= -11.0 + 1e-9:
                med.append(float(np.median(np.log10(go_all[m] / gl[m]))))
        out(f'   binned relation below 1e-11 m/s^2 (Upsilon 0.5/0.7, catalog distances, {len(gb_all)} points, 0.2-dex bins, median log10 g_obs/g_law): '
            f'{lab}: bins {", ".join(f"{x:+.3f}" for x in med)}; the law over the data, 10^-(mean) = {10 ** -np.mean(med):.3f}')
    stamp('the fit of record at the constrained minimum')


# ====================================================================== the full solution against the relation
def cross_setups():
    out('')
    out('THE FULL FIELD SOLUTION AGAINST THE ALGEBRAIC RELATION, EACH LAW AT ITS OWN BEST FIT (full minus relation; positive favours the relation;')
    out('galaxy-clustered SD over the same 2,000 draws; aqual-refit-main-T11-results.txt and aqual-refit-main-T3-results.txt give +233 to +279 for')
    out('the empirical function and +241 to +261 for the two-constant fit, with SDs of 140 to 144 on draws of their own)')
    for inp in ('sparc', 'model'):
        A, F = RES.get((inp, False)), RES.get((inp, True))
        if not (A and F):
            continue
        parts_ = []
        for h0 in CC.HUBBLE:
            parts_.append(f'the constrained fit (H0 = {h0}): {fmt_sd(F[("line", h0)].chi2 - A[("line", h0)].chi2, CC.sd_of(F[("line_b", h0)][0] - A[("line_b", h0)][0]))}')
        for law in ('SIM', 'STD', 'T11'):
            parts_.append(f'{CC.LAW_NAMES[law]}: {fmt_sd(F[law].chi2 - A[law].chi2, CC.sd_of(F[(law, "b")] - A[(law, "b")]))}')
        fb = np.array([x[0] for x in F[('T3', 'b')]]); ab = np.array([x[0] for x in A[('T3', 'b')]])
        parts_.append(f'the two-constant fit: {fmt_sd(F["T3"].chi2 - A["T3"].chi2, CC.sd_of(fb - ab))}')
        out(f'   {inp}: ' + '; '.join(parts_))


# ====================================================================== the optimum file for the exact re-solve
def write_optimum():
    if arg('--setups'):
        out(f'   --setups restricts the run: {OPT_PATH} and {VALLEY_PATH} are not written')
        return
    rows = []
    for inp in ('sparc', 'model'):
        R = RES.get((inp, True))
        if not R:
            continue
        T = R[('line', 67.4)]
        r, c = R[('line_at_min', 67.4)]
        sets = [(TAG67, 'T3', f'{T.dmin:.4f}', c['a0'], r)]
        for law in ('SIM', 'STD', 'T11'):
            L = R[law]
            sets.append((law, law, '', 10 ** L.la, L.best))
        for tag, law, dl, a0, res in sets:
            for gi, nm in enumerate(CC_DS.names):
                v = res['v'][gi]
                rows.append(f"{tag},{inp},full,{nm},{law},{dl},{a0:.6e},{10 ** res['u'][gi]:.8f},{(10 ** v) if np.isfinite(v) else 0.0:.8f},"
                            f"{res['q'][gi]:.8f},{res['F'][gi]:.6f}")
    with open(OPT_PATH, 'w', newline='\n', encoding='utf-8') as fh:
        fh.write('tag,input,model,galaxy,law,delta,a0,Yd,Yb,q,F\n' + '\n'.join(rows) + '\n')
    with open(VALLEY_PATH, 'w', newline='\n', encoding='utf-8') as fh:
        fh.write('setup,H0,kappa,delta,a0,chi2\n')
        for (inp, full), R in RES.items():
            if not isinstance(R, dict) or ('line', 67.4) not in R:
                continue
            lab = f'{inp}-{"full" if full else "relation"}'
            for kk, kap in (('line', 1.0), ('k3', 1.0 / 3.0)):
                for h0 in CC.HUBBLE:
                    T = R[(kk, h0)]
                    for d, ch in zip(T.d, T.curve):
                        fh.write(f'{lab},{h0},{kap:.4f},{d:.4f},{CC.c_hds(h0) * math.log1p(d) / kap:.6e},{ch:.4f}\n')
    out(f'   written: {OPT_PATH} (the fitted nuisances at the optima of the full field solution, for constrained_solve.py part atopt) and '
        f'{VALLEY_PATH} (the constrained curves)')


# ====================================================================== part treat
def part_treat(gals):
    out('')
    out('=' * 118)
    out("THE SIXTEEN TREATMENTS OF PAPER I §4.1, SPARC'S FIELD, THE ALGEBRAIC RELATION")
    out('=' * 118)
    out('   M/L: standard (both ratios free under 0.10 dex priors), locked (bulge at 1.4 x disk), pinned (0.01 dex), wide (0.25 dex); distances')
    out('   and inclinations marginalized or at the catalog values (q = 1); all 153 galaxies or those inclined by 45 deg or more. The')
    out('   constrained curve runs from delta 0.03 to 0.50 in steps of 0.01; the bootstrap uses 2,000 draws by galaxy with seed 20261114 in')
    out('   every cell.')
    out('   cell | galaxies | two-constant: delta, a0, chi^2 | constrained H0 67.4: delta [16-84], a0, chi^2 | constrained H0 73: delta [16-84] | '
        'simple: a0, chi^2 | constrained (67.4) minus simple | minus empirical | minus standard')
    rows = []
    for cell, ml, nuis, samp in CC.TREATMENTS:
        if arg('--cells') and cell not in arg('--cells').split(','):
            continue                     # testing only
        t0 = time.time()
        ds = CC.treatment_dataset(gals, ml, nuis, samp)
        n = len(ds.names)
        W = RC.boot_weights(n, CC.NBOOT, CC.SEED)
        T = {h0: CC.ConstrainedLine(ds, 'sparc', False, h0, DELTAS_T) for h0 in CC.HUBBLE}
        tb = {h0: boot_line(T[h0], W) for h0 in CC.HUBBLE}
        fr = 0.80 if nuis == 'catalog' else 1.0
        L = {}
        for law, a in (('SIM', 1.08e-10), ('T11', 1.05e-10), ('STD', 1.21e-10)):
            L[law] = RC.Ladder(ds, CC.ctx('sparc', law), math.log10(a * fr * (1.15 if ml == 'pinned' and nuis != 'catalog' else 1.0)))
        Lb = {law: np.array([L[law].node_min(w)[0] for w in W]) for law in L}
        two = CC.two_constant(ds, 'sparc', False, math.log10(0.95e-10 * fr))
        row = dict(cell=cell, n=n, two=(two.delta, 10 ** two.la, two.chi2, two.edge),
                   t67=(T[67.4].dmin, CC.pct(tb[67.4][1], 16), CC.pct(tb[67.4][1], 84), CC.a0_of_delta(T[67.4].dmin, 67.4), T[67.4].chi2, T[67.4].edge),
                   t73=(T[73.0].dmin, CC.pct(tb[73.0][1], 16), CC.pct(tb[73.0][1], 84), CC.a0_of_delta(T[73.0].dmin, 73.0), T[73.0].chi2),
                   sim=(10 ** L['SIM'].la, L['SIM'].chi2),
                   dsim=(T[67.4].chi2 - L['SIM'].chi2, CC.sd_of(tb[67.4][0] - Lb['SIM'])),
                   dt11=(T[67.4].chi2 - L['T11'].chi2, CC.sd_of(tb[67.4][0] - Lb['T11'])),
                   dstd=(T[67.4].chi2 - L['STD'].chi2, CC.sd_of(tb[67.4][0] - Lb['STD'])),
                   dsim73=(T[73.0].chi2 - L['SIM'].chi2, CC.sd_of(tb[73.0][0] - Lb['SIM'])),
                   chi19=T[67.4].at(0.19)[0] - T[67.4].chi2, chi17=T[73.0].at(0.17)[0] - T[73.0].chi2)
        rows.append(row)
        t2 = row['two']; a = row['t67']; b = row['t73']
        out(f'   {cell} | {n} | {t2[0]:.4f}{" (stop)" if t2[3] else ""}, {t2[1] * 1e10:.4f}, {t2[2]:.2f} | '
            f'{a[0]:.4f} [{a[1]:.3f}-{a[2]:.3f}]{" (end)" if a[5] else ""}, {a[3] * 1e10:.4f}, {a[4]:.2f} | {b[0]:.4f} [{b[1]:.3f}-{b[2]:.3f}] | '
            f'{row["sim"][0] * 1e10:.4f}, {row["sim"][1]:.2f} | {fmt_sd(*row["dsim"])} | {fmt_sd(*row["dt11"])} | {fmt_sd(*row["dstd"])}   [{time.time() - t0:.0f} s]')
        RC.PRIOR = 0.10
    RES['treat'] = rows
    f = lambda key, i: [r[key][i] for r in rows]
    out('   ACROSS THE SIXTEEN TREATMENTS:')
    out(f'      the constrained minimum at H0 = 67.4: delta {min(f("t67", 0)):.3f} to {max(f("t67", 0)):.3f}, a0 {min(f("t67", 3)) * 1e10:.3f} to {max(f("t67", 3)) * 1e10:.3f}e-10 m/s^2; '
        f'the lowest 16th and highest 84th percentiles {min(f("t67", 1)):.3f} and {max(f("t67", 2)):.3f}')
    out(f'      the constrained minimum at H0 = 73: delta {min(f("t73", 0)):.3f} to {max(f("t73", 0)):.3f}')
    out(f'      the constrained fit minus the simple function: {min(f("dsim", 0)):+.1f} to {max(f("dsim", 0)):+.1f}; in SD: '
        f'{min(r["dsim"][0] / r["dsim"][1] for r in rows):+.2f} to {max(r["dsim"][0] / r["dsim"][1] for r in rows):+.2f} (at H0 = 73: '
        f'{min(f("dsim73", 0)):+.1f} to {max(f("dsim73", 0)):+.1f})')
    out(f'      the constrained fit minus the empirical function: {min(f("dt11", 0)):+.1f} to {max(f("dt11", 0)):+.1f}; minus the standard function: '
        f'{min(f("dstd", 0)):+.1f} to {max(f("dstd", 0)):+.1f}')
    out(f'      the two-constant free volume: {min(f("two", 0)):.3f} to {max(f("two", 0)):.3f}; chi^2 at delta = 0.19 minus the constrained minimum (67.4): '
        f'{min(r["chi19"] for r in rows):+.2f} to {max(r["chi19"] for r in rows):+.2f}')
    stamp('treatments')


# ====================================================================== part scales
def acc_fit(gb, go, sig_o=None, sig_b=None):
    """The empirical function fitted in log acceleration, log10 g_obs = log10[nu(g_bar/a0) g_bar], over log10 a0:
    least squares unweighted (sig_o None) or with the effective variance sig_o^2 + (F' sig_b)^2, F' = dlog g/dlog g_bar."""
    lgo, lgb = np.log10(go), np.log10(gb)

    def cost(la):
        a0 = 10 ** la
        F = np.log10(RC.nu_t11(gb / a0) * gb)
        if sig_o is None:
            return float(np.sum((lgo - F) ** 2))
        h = 1e-4
        Fp = (np.log10(RC.nu_t11(gb * 10 ** h / a0) * gb * 10 ** h) - np.log10(RC.nu_t11(gb * 10 ** -h / a0) * gb * 10 ** -h)) / (2 * h)
        v = sig_o ** 2 + (Fp * sig_b) ** 2
        return float(np.sum((lgo - F) ** 2 / v))
    las = np.arange(-10.4, -9.6, 0.005)
    cs = np.array([cost(x) for x in las])
    i = int(np.argmin(cs))
    r = RC.parab3([(las[i - 1], cs[i - 1]), (las[i], cs[i]), (las[i + 1], cs[i + 1])])
    return 10 ** r[0], r[2]


def part_scales(gals):
    out('')
    out('=' * 118)
    out('THE SCALE OF THE EMPIRICAL FUNCTION UNDER OTHER OBJECTIVES, AND THE PRODUCT D ln(1 + delta) OF THE TWO-CONSTANT FIT')
    out('=' * 118)
    R = RES[('sparc', False)]
    # (a) the scale of the empirical function
    out('(a) THE SCALE OF THE EMPIRICAL FUNCTION OF McGAUGH, LELLI AND SCHOMBERT, 0.96e-10 HERE AGAINST THEIR PUBLISHED 1.20e-10 m/s^2: the scale refitted')
    steps = []
    RC.PRIOR = 0.01
    dsf = CC.XDataset(gals, frozen=True)
    L = RC.Ladder(dsf, CC.ctx('sparc', 'T11'), math.log10(0.95e-10))
    steps.append(('pinned ratios (0.01 dex about 0.5 and 0.7), catalog distances, all points, velocity objective with the 5 percent floor (Paper I)', L))
    dsm = CC.XDataset(gals, frozen=False)
    L2 = RC.Ladder(dsm, CC.ctx('sparc', 'T11'), math.log10(1.25e-10))
    steps.append(('pinned ratios, distances and inclinations fitted within their priors', L2))
    pm = {n: g['selA'] for n, g in gals.items()}
    namesA = [n for n in sorted(gals) if pm[n].any()]
    dsa = CC.XDataset(gals, names=namesA, pmask=pm, frozen=True)
    L3 = RC.Ladder(dsa, CC.ctx('sparc', 'T11'), math.log10(0.95e-10))
    steps.append(('pinned, catalog distances, the points with eV/V < 0.1 only (the published cut)', L3))
    RC.FLOOR = 0.0
    dsa0 = CC.XDataset(gals, names=namesA, pmask=pm, frozen=True)
    L4 = RC.Ladder(dsa0, CC.ctx('sparc', 'T11'), math.log10(0.95e-10))
    steps.append(('the same without the 5 percent floor (weights 1/eV^2)', L4))
    RC.FLOOR = 0.05
    RC.PRIOR = 0.10
    for lab, L_ in steps:
        npt = sum(int(np.sum(g.npt)) for g in L_.ds.groups if g.G)
        out(f'   {lab}: a0 = {10 ** L_.la * 1e10:.4f}e-10, chi^2 = {L_.chi2:.1f} ({len(L_.ds.names)} galaxies, {npt} points)')
    # the fit in log acceleration at the prior-centre ratios and catalog distances
    gbA, goA, eA, gbF, goF, eF, fs = [], [], [], [], [], [], []
    for n in sorted(gals):
        g = gals[n]
        gbar = (g['Vg'] * np.abs(g['Vg']) + 0.5 * g['Vd'] ** 2 + 0.7 * g['Vb'] ** 2) / g['R'] * RC.CONV
        gstar = (0.5 * g['Vd'] ** 2 + 0.7 * g['Vb'] ** 2) / g['R'] * RC.CONV
        gobs = g['Vo'] ** 2 / g['R'] * RC.CONV
        k = (gbar > 0) & (gobs > 0)
        gbF.append(gbar[k]); goF.append(gobs[k]); eF.append((g['eV'] / g['Vo'])[k])
        kA = k & g['selA']
        gbA.append(gbar[kA]); goA.append(gobs[kA]); eA.append((g['eV'] / g['Vo'])[kA])
        fs.append(np.clip(gstar[kA] / gbar[kA], 0, 1))
    gbA, goA, eA, gbF, goF, eF, fs = map(np.concatenate, (gbA, goA, eA, gbF, goF, eF, fs))
    a1, _ = acc_fit(gbF, goF)
    a2, _ = acc_fit(gbA, goA)
    so = 2.0 * eA / math.log(10.0)
    a3, _ = acc_fit(gbA, goA, sig_o=so, sig_b=np.zeros_like(so))
    a4, _ = acc_fit(gbA, goA, sig_o=so, sig_b=0.1 * fs)
    a5, _ = acc_fit(gbA, goA, sig_o=np.sqrt(so ** 2 + 0.05 ** 2), sig_b=0.1 * fs)
    out('   in log acceleration at the prior-centre ratios (0.5, 0.7) and catalog distances, the scale alone fitted:')
    out(f'      unweighted least squares, all {len(gbF)} points: a0 = {a1 * 1e10:.4f}e-10; the {len(gbA)} points with eV/V < 0.1: {a2 * 1e10:.4f}e-10')
    out(f'      weighted by the velocity errors only (sigma = 2 eV/V / ln 10), eV/V < 0.1: {a3 * 1e10:.4f}e-10')
    out(f'      errors in both variables (effective variance), with 0.1 dex on the stellar part of g_bar (an assumed error model of the kind the '
        f'published fit describes): {a4 * 1e10:.4f}e-10; adding 0.05 dex to the g_obs error (distance and inclination): {a5 * 1e10:.4f}e-10')
    RES['mls'] = dict(steps=[(lab, 10 ** L_.la) for lab, L_ in steps], acc=(a1, a2, a3, a4, a5))
    stamp('scales (a)')
    # (b) the product D ln(1 + delta) and the divisor, from the two-constant fit's own draws
    two = R['T3']
    out('(b) THE PRODUCT D ln(1 + delta) AND THE DIVISOR OF THE TWO-CONSTANT FIT (the rule of two_constant_fit.pl section M on this run\'s draws):')
    S = two.S
    nodes = sorted(S.nodes)
    kap = {h0: [] for h0 in CC.HUBBLE}
    for w in RC.boot_weights(len(CC_DS.names), CC.NBOOT, CC.SEED):
        nm = {d: S.nodes[d].node_min(w) for d in nodes}
        yb, xb = S.scan_min(w)
        n3 = sorted(sorted(nodes, key=lambda d: nm[d][0])[:3])
        if n3[0] <= xb <= n3[-1]:
            xs = np.array(n3); ys = np.array([nm[d][1] for d in n3])
            la = float(np.polyval(np.polyfit(xs, ys, 2), xb))
        else:
            la = nm[min(nodes, key=lambda d: abs(d - xb))][1]
        for h0 in CC.HUBBLE:
            kap[h0].append(CC.c_hds(h0) / 10 ** la * math.log1p(xb))
    for h0 in CC.HUBBLE:
        k0 = CC.c_hds(h0) / 10 ** two.la * math.log1p(two.delta)
        kb = np.array(kap[h0])
        out(f'   H0 = {h0}: product D ln(1 + delta) = {k0:.3f} at the fit; draws: SD {CC.sd_of(kb):.3f}, 16-84 percent {CC.pct(kb, 16):.3f} to '
            f'{CC.pct(kb, 84):.3f}, at or below 1 in {int(np.sum(kb <= 1))} of {len(kb)}; (product - 1)/SD = {(k0 - 1) / CC.sd_of(kb):.2f}')
    lam = math.log1p(two.delta)
    sig = two.S.sig
    out(f'   the derived divisor 1/ln(1 + delta) = {1 / lam:.3f} at delta = {two.delta:.4f}, +/- {sig / ((1 + two.delta) * lam ** 2):.3f} from the '
        f'statistical width {sig:.4f}; at the rounded 0.30 it is {1 / math.log(1.3):.3f}. On the scale relation the divisor is 1/ln(1.19) = '
        f'{1 / math.log(1.19):.3f} (67.4), and c H_dS/a0 = {CC.c_hds(73.0) / CC.a0_of_delta(0.19, 67.4):.3f} at H0 = 73 for the same a0')
    stamp('scales')


# ====================================================================== part exact
def part_exact(ds):
    out('')
    out('=' * 118)
    out('THE EXACT RE-SOLVE AT THE OPTIMA OF THE FULL FIELD SOLUTION: the objective at each fit\'s ratios with Gamma from a direct AQUAL solution there')
    out('=' * 118)
    have = [p for p in ATOPT if os.path.exists(p)]
    if not have or not os.path.exists(OPT_PATH):
        out('   constrained_solve-atopt-*.csv not present: skipped')
        return
    label = {TAG67: 'the constrained fit (H0 = 67.4)', 'SIM': 'SIM', 'STD': 'STD', 'T11': 'T11'}
    at = {}
    for p in have:
        for r in csv.DictReader(open(p, encoding='utf-8-sig')):
            at[(r['tag'], r['input'], r['galaxy'], round(float(r['R']), 4))] = float(r['Gamma'])
    opt = list(csv.DictReader(open(OPT_PATH, encoding='utf-8-sig')))
    res = {}
    for inp in ('sparc', 'model'):
        for tag in (TAG67, 'SIM', 'STD', 'T11'):
            rows = {r['galaxy']: r for r in opt if r['tag'] == tag and r['input'] == inp}
            if not rows:
                continue
            r0 = next(iter(rows.values()))
            law = r0['law']
            c = CC.ctx(inp, law, full=True, delta=float(r0['delta']) if r0['delta'] else None, a0=float(r0['a0']), gmode='exact')
            gex, nund = [], 0
            for grp in ds.groups:
                if grp.G == 0:
                    gex.append(np.zeros(0)); continue
                keys = [(g['name'], round(float(R_), 4)) for g, s in zip(grp.gl, grp.sel) for R_ in g['R'][s]]
                v = np.array([at.get((tag, inp, k[0], k[1]), np.nan) for k in keys])
                nund += int(np.sum(~np.isfinite(v) & grp.gok))
                gex.append(np.where(np.isfinite(v), v, 0.0))
            c['gexact'] = gex
            Fex = 0.0
            for grp in ds.groups:
                if grp.G == 0:
                    continue
                U = np.array([[math.log10(float(rows[g['name']]['Yd']))] for g in grp.gl])
                V = np.array([[math.log10(float(rows[g['name']]['Yb']))] for g in grp.gl]) if grp.two else None
                F, _ = ds.objective(grp, U, V, c)
                Fex += float(F.sum())
            Fc = sum(float(r['F']) for r in rows.values())
            res[(inp, tag)] = (Fc, Fex)
            out(f'   {inp}, {label[tag]}: corrected objective {Fc:.2f}, exact at the same ratios {Fex:.2f} (difference {Fex - Fc:+.2f}; '
                f'Gamma undefined at {nund} points)')
        if (inp, TAG67) in res:
            Tc, Te = res[(inp, TAG67)]
            out(f'   {inp}: the constrained fit minus each competitor, corrected and with the exact objective (an upper bound for each side): '
                + '; '.join(f'{t}: {Tc - res[(inp, t)][0]:+.1f} and {Te - res[(inp, t)][1]:+.1f}' for t in ('SIM', 'STD', 'T11') if (inp, t) in res))
    stamp('exact')


# ====================================================================== main
CC_DS = None
if __name__ == '__main__':
    parts = (arg('--parts') or 'repro,main,treat,scales,exact').split(',')
    T1, gals, ds, nnew, dmax, ref = load()
    CC_DS = ds
    out('THE CONSTRAINED FIT: THE BOOST LAW WITH a0 = c H_dS ln(1 + delta), kappa = 1, ON THE 153 SPARC ROTATION CURVES')
    out(f'c H_dS = {CC.c_hds(67.4) * 1e10:.4f}e-10 m/s^2 at H0 = 67.4 km/s/Mpc and {CC.c_hds(73.0) * 1e10:.4f}e-10 at 73 (Omega_Lambda = {CC.OMEGA_L}); '
        f'the scale relation gives a0 = {CC.a0_of_delta(0.19, 67.4) * 1e10:.4f}e-10 at delta = 0.19 (67.4) and {CC.a0_of_delta(0.17, 73.0) * 1e10:.4f}e-10 at 0.17 (73)')
    out(f'{len(ds.names)} galaxies; {CC.NBOOT} bootstrap draws by galaxy, seed {CC.SEED}. Added Gamma tables (simple, standard): {nnew} points newly '
        f'undefined; the model\'s Newtonian field agrees with that of aqual-refit-gamma-*.csv to {dmax:.1e}; the standard function\'s correction '
        f'measured at a0 = {ref:.3e} m/s^2.')
    out('On the full field solution Gamma is read between the solver\'s nodes as in aqual-refit-fit.py: a quartic through the five free-volume')
    out('nodes (0.05, 0.2, 0.4, 0.7, 1.2), a Catmull-Rom cubic in log10 a0 on the 0.2-dex nodes at a0/s, and the correction for other')
    out('mass-to-light ratios linear in delta between its nodes 0.03 and 0.3; the constrained curve needs no interpolation of chi^2 between')
    out('scale nodes, since each of its nodes is fitted directly at its own a0. Part exact gives the error that the reading of Gamma adds at the optima.')
    W = RC.boot_weights(len(ds.names), CC.NBOOT, CC.SEED)
    W_ALL = W
    if 'repro' in parts:
        part_repro(ds)
    if any(p in parts for p in ('main', 'scales')):
        for inp, full, title in SETUPS:
            if 'main' not in parts and (full or inp != 'sparc'):
                continue                 # part scales alone needs only the fit of record
            if arg('--setups') and f'{inp}-{"full" if full else "relation"}' not in arg('--setups').split(','):
                continue                 # testing only
            run_setup(ds, inp, full, title, W)
            if not full and inp == 'sparc':
                baseline_extras(ds, gals)
        if 'main' in parts:
            cross_setups()
            write_optimum()
    if 'treat' in parts:
        part_treat(gals)
    if 'scales' in parts:
        part_scales(gals)
    if 'exact' in parts:
        part_exact(ds)
    out(f'\n[run time {time.time() - T00:.0f} s]')
    if arg('--out'):
        with open(arg('--out'), 'w', newline='\n', encoding='utf-8') as fh:
            fh.write('\n'.join(_lines) + '\n')
