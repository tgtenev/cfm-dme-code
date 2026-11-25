"""Direct AQUAL solutions that the constrained fit (constrained_fit.py) needs beyond those of aqual-refit-solve.py:
MOND's simple and standard functions on the full field solution, and the exact re-solve at the optima of the
constrained fit and of its competitors on the full field solution. They support Paper I §4.2 and Paper II §6.1.

The machinery is that of aqual-refit-solve.py and aqual_law.py in this folder (the disk solver of aqual-disk-solver.py
and its galaxy models), imported unchanged. Only the two laws are new, written by the same interface (nu(y), mu(x)
on arrays):
  SIM, MOND's simple function:   nu(y) = 1/2 + sqrt(1/4 + 1/y),          mu(x) = x/(1 + x);
  STD, MOND's standard function: nu(y) = sqrt(1/2 + sqrt(1/4 + 1/y^2)),  mu(x) = x/sqrt(1 + x^2).
The other law codes: T3, the boost law of Paper I, Eq. (8); T11, the empirical function of McGaugh, Lelli and
Schombert, nu(y) = 1/(1 - exp(-sqrt(y))).

Parts (each writes the file named by --out):
  gamma   Gamma = log10 g_full - log10[nu(g_N/a0) g_N] at the prior-centre ratios (disk 0.5, bulge 0.7) for SIM and
          STD at log10 a0 = -10.7 to -9.3 in steps of 0.2 (the grid of the boost law in aqual-refit-solve.py), with
          the model's Newtonian field in total and by component (for a check against aqual-refit-gamma-*.csv).
  upsvar  Gamma of STD at --a0ref (default 1.10e-10 m/s^2) at other mass-to-light ratios, on the grid of the
          correction of refit_core.py: the disk ratio at -1.0 to +1.0 dex from 0.5 in steps of 0.2, and for bulge
          galaxies every pair of disk ratio -0.4 to +0.8 dex and bulge ratio -0.8 to +0.8 dex from 0.7 in steps of
          0.2. It supplies the 'corrected' treatment of STD in constrained_core.py. SIM borrows the boost law's
          correction at delta 0.03.
  atopt   the exact re-solve at an optimum: every 'full' row of --opt (written by constrained_fit.py part main)
          solved directly at its fitted disk and bulge ratios, at its law, free volume and scale. Gamma per point.
Options: --slice k/n splits the galaxies (as in aqual-refit-solve.py), --only G1,G2 restricts them (testing).

Inputs: rotation-curves/aqual-disk-points.csv and rotation-curves/aqual-disk-gas-profiles.csv (the galaxy models of
aqual-disk-solver.py); for part atopt, rotation-curves/constrained_fit-optimum.csv. Run from the repository root:
    python rotation-curves/constrained_solve.py --part gamma --slice 0/2 --out rotation-curves/constrained_solve-gamma-0.csv
    python rotation-curves/constrained_solve.py --part gamma --slice 1/2 --out rotation-curves/constrained_solve-gamma-1.csv
    python rotation-curves/constrained_solve.py --part upsvar --a0ref 1.10e-10 --slice 0/2 --out rotation-curves/constrained_solve-upsvar-0.csv
    python rotation-curves/constrained_solve.py --part upsvar --a0ref 1.10e-10 --slice 1/2 --out rotation-curves/constrained_solve-upsvar-1.csv
    python rotation-curves/constrained_solve.py --part atopt --opt rotation-curves/constrained_fit-optimum.csv --slice 0/3 --out rotation-curves/constrained_solve-atopt-0.csv
and part atopt likewise with --slice 1/3 and 2/3 into constrained_solve-atopt-1.csv and -2.csv. Part atopt reads the
optimum file that constrained_fit.py part main writes. The two slices of part gamma took 150 s and 158 s, those of
part upsvar 185 s and 294 s, and the three of part atopt 95 to 112 s each. Needs numpy only.
"""
import csv
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
import aqual_law as AL

_spec = importlib.util.spec_from_file_location('aqual_refit_solve', os.path.join(HERE, 'aqual-refit-solve.py'))
SV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SV)            # its arg() reads this command line: --slice, --only, --out
T16 = AL.T16
T00 = time.time()
ARGS = sys.argv
MODELS = 'rotation-curves'               # the folder of aqual-disk-points.csv and aqual-disk-gas-profiles.csv
LA_X = [round(-10.7 + 0.2 * k, 1) for k in range(8)]
EPS_D = [round(-1.0 + 0.2 * k, 1) for k in range(11)]
EPS_D2 = [round(-0.4 + 0.2 * k, 1) for k in range(7)]
EPS_B = [round(-0.8 + 0.2 * k, 1) for k in range(9)]


def arg(name, default=None):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default


class LawSimple:
    """MOND's simple function."""
    code = 'SIM'
    shape = None

    def nu(self, y):
        y = np.maximum(np.asarray(y, float), 1e-300)
        return 0.5 * (1.0 + np.sqrt(1.0 + 4.0 / y))

    def mu(self, x):
        x = np.maximum(np.asarray(x, float), 0.0)
        return x / (1.0 + x)

    def label(self):
        return 'SIM'


class LawStandard:
    """MOND's standard function."""
    code = 'STD'
    shape = None

    def nu(self, y):
        y = np.maximum(np.asarray(y, float), 1e-150)
        return np.sqrt(0.5 + 0.5 * np.sqrt(1.0 + 4.0 / (y * y)))

    def mu(self, x):
        x = np.minimum(np.maximum(np.asarray(x, float), 0.0), 1e100)
        return x / np.sqrt(1.0 + x * x)

    def label(self):
        return 'STD'


def make_law(code, delta=None):
    if code == 'SIM':
        return LawSimple()
    if code == 'STD':
        return LawStandard()
    return AL.make_law(code, delta)


def part_gamma():
    path = arg('--out')
    gals, gas = T16.load_inputs(MODELS)
    names = SV.galaxy_names(gals)
    laws = [('STD', la) for la in LA_X] + [('SIM', la) for la in LA_X]
    obj = {'STD': LawStandard(), 'SIM': LawSimple()}
    cols = ['galaxy', 'R', 'gN', 'gNd', 'gNg', 'gNb', 'itmax'] + [f'G_{c}_{la:.1f}' for c, la in laws]
    with open(path, 'w', newline='\n', encoding='utf-8') as fh:
        fh.write(','.join(cols) + '\n')
        for nm in names:
            t = time.time()
            m, rho = SV.build(nm, gals, gas)
            rd, rg, rb = m.rho_parts
            S = AL.LawSolver(m.grid, obj['STD'], AL.A0_T16)
            at = SV.midplane(m)
            _, FR, FZ = S.newton(rho)
            gN = at(FR)
            comp = [at(S.newton(p)[1]) for p in (rd, rg, rb)]
            res, itmax = {}, 0
            for c, la in laws:
                res[(c, la)], it = SV.gamma_of(S, obj[c], AL.a0_units(10.0 ** la), FR, FZ, gN, at)
                itmax = max(itmax, it)
            for i in range(len(m.R)):
                vals = [nm, f'{m.R[i]:.4f}', f'{gN[i]:.6e}', f'{comp[0][i]:.6e}', f'{comp[1][i]:.6e}', f'{comp[2][i]:.6e}', str(itmax)]
                vals += [f'{res[x][i]:.6f}' for x in laws]
                fh.write(','.join(vals) + '\n')
            fh.flush()
            print(f'{nm}: {len(m.R)} points, max AQUAL iterations {itmax}, {time.time() - t:.1f} s', flush=True)
    print(f'[run time {time.time() - T00:.0f} s]')


def ups_col(a0si, sd, sb):
    return f'STD_a{a0si:.4e}_d{sd:+.2f}_b{sb:+.2f}'


def part_upsvar():
    path = arg('--out')
    a0si = float(arg('--a0ref', '1.10e-10'))
    gals, gas = T16.load_inputs(MODELS)
    names = SV.galaxy_names(gals)
    law = LawStandard()
    disk = [(d, 0.0) for d in EPS_D if abs(d) > 1e-9]
    two = [(u, v) for u in EPS_D2 for v in EPS_B if abs(v) > 1e-9]
    cols = [ups_col(a0si, sd, sb) for sd, sb in disk + two]
    with open(path, 'w', newline='\n', encoding='utf-8') as fh:
        fh.write('galaxy,R,' + ','.join(cols) + '\n')
        for nm in names:
            t = time.time()
            m, rho = SV.build(nm, gals, gas)
            rd, rg, rb = m.rho_parts
            hasb = m.bulge is not None
            S = AL.LawSolver(m.grid, law, AL.A0_T16)
            at = SV.midplane(m)
            vals = {}
            for sd, sb in disk + (two if hasb else []):
                r = (10 ** sd) * rd + rg + (10 ** sb) * rb
                _, FR, FZ = S.newton(r)
                vals[ups_col(a0si, sd, sb)] = SV.gamma_of(S, law, AL.a0_units(a0si), FR, FZ, at(FR), at)[0]
            for i in range(len(m.R)):
                fh.write(','.join([nm, f'{m.R[i]:.4f}'] + [(f'{vals[k][i]:.6f}' if k in vals else '') for k in cols]) + '\n')
            fh.flush()
            print(f'{nm}: {len(vals)} cases, {time.time() - t:.1f} s', flush=True)
    print(f'[run time {time.time() - T00:.0f} s]')


def part_atopt():
    path = arg('--out')
    gals, gas = T16.load_inputs(MODELS)
    names = set(SV.galaxy_names(gals))
    rows = [r for r in csv.DictReader(open(arg('--opt'), encoding='utf-8-sig')) if r['model'] == 'full' and r['galaxy'] in names]
    lawobj = {}
    with open(path, 'w', newline='\n', encoding='utf-8') as fh:
        fh.write('tag,input,galaxy,R,Gamma\n')
        built = {}
        for r in sorted(rows, key=lambda r: (r['galaxy'], r['tag'], r['input'])):
            t = time.time()
            nm = r['galaxy']
            if nm not in built:
                built = {nm: SV.build(nm, gals, gas)}
            m, rho = built[nm]
            rd, rg, rb = m.rho_parts
            Yd, Yb = float(r['Yd']), float(r['Yb'])
            sb = Yb / 0.7 if (m.bulge is not None and Yb > 0) else 1.0
            d = float(r['delta']) if r['delta'] else None
            key = (r['law'], d)
            if key not in lawobj:
                lawobj[key] = make_law(r['law'], d)
            S = AL.LawSolver(m.grid, lawobj[key], AL.A0_T16)
            at = SV.midplane(m)
            _, FR, FZ = S.newton((Yd / 0.5) * rd + rg + sb * rb)
            G, it = SV.gamma_of(S, lawobj[key], AL.a0_units(float(r['a0'])), FR, FZ, at(FR), at)
            for i in range(len(m.R)):
                fh.write(f"{r['tag']},{r['input']},{nm},{m.R[i]:.4f},{G[i]:.6f}\n")
            fh.flush()
            print(f"{nm} ({r['tag']}, {r['input']}): {it} iterations, {time.time() - t:.1f} s", flush=True)
    print(f'[run time {time.time() - T00:.0f} s]')


if __name__ == '__main__':
    {'gamma': part_gamma, 'upsvar': part_upsvar, 'atopt': part_atopt}[arg('--part', 'gamma')]()
