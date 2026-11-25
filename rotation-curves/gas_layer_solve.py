"""The full AQUAL solution in the disks averaged over the gas layer: the departure tables of the gas-layer baseline test
(gas_layer_baseline.py).

RESULTS SUPPORTED
  Paper II, Sect. 6.1: the slope test against the vertically gas-averaged accelerations (about 3 SD) and the observed
  Delta chi^2 against its value for noiseless rotation curves made by the full solution (about 5 SD), through
  gas_layer_baseline.py.

WHAT IT COMPUTES
  The rotation curves measure V^2/R of gas that orbits within a layer of finite thickness. Here the radial field is
  averaged over that layer: an exponential gas layer of scale height 0.2 kpc, the layer of the galaxy models of
  aqual-disk-solver.py, with weights equal to the layer's mass fraction in each cell row. For every galaxy of the
  sample of Paper I, Sect. 4.1 (153 galaxies, 3,168 points) the model of aqual-disk-solver.py is built, and at each data
  radius the tables give, against the algebraic relation on the midplane Newtonian field g_N that the fits use,
      gas  = log10 <g_full,R>_gas - log10[nu(g_N/a0) g_N]                    (the full AQUAL solution, averaged)
      Pgas = log10 <g_full,R>_gas - log10 <nu(|g_N,R|/a0) g_N,R>_gas         (AQUAL's own part of the departure)
      rgas = log10 <nu(|g_N,R|/a0) g_N,R>_gas - log10[nu(g_N/a0) g_N]       (the relation applied at each height and
                                                                               averaged: Newton's geometry only)
  so that gas = rgas + Pgas. The law is T11, the empirical function of McGaugh, Lelli and Schombert,
  nu(y) = 1/(1 - exp(-sqrt(y))).
  Part gamma: gas and Pgas at the mass-to-light ratios' prior centres (0.5 and 0.7), at log10 a0 = -10.7 to -9.3 in
      steps of 0.1 (a0 in m/s^2), with gN, the midplane Newtonian field in (km/s)^2/kpc.
  Part upsvar: gas from direct solutions at other mass-to-light ratios, at a0 = 1.05e-10: the disk ratio at -1.0 to
      +1.0 dex from 0.5 in steps of 0.2 and, for bulge galaxies, every pair of disk ratio -0.4 to +0.8 dex and bulge
      ratio -0.8 to +0.8 dex from 0.7 (the cases of aqual-refit-solve.py part upsvar2 for T11). The fits' mode
      'corrected' (refit_core.py) carries the departure to the fitted ratios with them.
  Part relvar: rgas at the same cases, from the Newtonian solutions alone.
  Columns of upsvar and relvar are named <var>_T11_a<a0>_d<dlog10 Yd>_b<dlog10 Yb>; bulge cases are empty for bulgeless
  galaxies.

INPUTS
  rotation-curves/aqual-disk-points.csv and rotation-curves/aqual-disk-gas-profiles.csv

OUTPUTS AND COMMANDS, from the repository root (the slices can run at the same time)
  python rotation-curves/gas_layer_solve.py --part gamma --slice k/6 --out rotation-curves/gas-layer-gamma-k.csv
      for k = 0 to 5 (14 to 15 min each)
  python rotation-curves/gas_layer_solve.py --part upsvar --slice k/6 --out rotation-curves/gas-layer-upsvar-k.csv
      for k = 0 to 5 (14 to 24 min each)
  python rotation-curves/gas_layer_solve.py --part relvar --slice k/2 --out rotation-curves/gas-layer-relvar-k.csv
      for k = 0 and 1 (4 to 6 min each)
  --only G1,G2 restricts the galaxies, as a quick test. Needs numpy.
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import aqual_law as AL

T16 = AL.T16
T00 = time.time()
ARGS = sys.argv
LA = [round(-10.7 + 0.1 * k, 1) for k in range(15)]         # log10 a0 (m/s^2) of part gamma


def arg(name, default=None):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default


def galaxy_names(gals):
    names = sorted(gals)
    if arg("--only"):
        names = [x for x in arg("--only").split(",") if x in gals]
    k, n = (int(x) for x in arg("--slice", "0/1").split("/"))
    return [nm for i, nm in enumerate(names) if i % n == k]


def zweights(grid, h):
    """Mass fraction of an exponential layer of scale height h in each cell row (z >= 0), normalized."""
    w = np.exp(-grid.zf[:-1] / h) - np.exp(-grid.zf[1:] / h)
    return w / w.sum()


class Probe:
    """Midplane and gas-layer-averaged radial fields of one model at its data radii."""

    def __init__(self, m):
        self.m = m
        self.Rf1 = m.grid.Rf[1:]
        self.w = zweights(m.grid, T16.H_GAS)

    def mid(self, F):
        return np.interp(self.m.R, self.Rf1, F[:, 0])

    def avg(self, F):
        return np.interp(self.m.R, self.Rf1, F @ self.w)


def lg(x):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(x > 0, np.log10(np.where(x > 0, x, 1.0)), np.nan)


def departures(P, law, a0, FR, aR, gN):
    """gas and Pgas at the data radii (see the header)."""
    relmid = law.nu(gN / a0) * gN
    relpw = law.nu(np.abs(FR) / a0) * FR
    ok = gN > 0
    gt = P.avg(aR)
    res = {"gas": lg(gt) - lg(relmid), "Pgas": lg(gt) - lg(P.avg(relpw))}
    return {k: np.where(ok, v, np.nan) for k, v in res.items()}


# ---------------------------------------------------------------------------------------------- part gamma
def part_gamma():
    path = arg("--out")
    gals, gas = T16.load_inputs(HERE)
    names = galaxy_names(gals)
    law = AL.make_law("T11")
    cols = ["galaxy", "R", "gN"] + [f"{v}_{la:.1f}" for la in LA for v in ("gas", "Pgas")]
    fh = open(path, "w", newline="\n", encoding="utf-8")
    fh.write(",".join(cols) + "\n")
    for nm in names:
        t = time.time()
        m = T16.GalaxyModel(nm, gals[nm], gas[nm])
        rho = m.density()
        S = AL.LawSolver(m.grid, law, AL.A0_T16)
        P = Probe(m)
        _, FR, FZ = S.newton(rho)
        gN = P.mid(FR)
        res = {}
        for la in LA:
            a0 = AL.a0_units(10.0 ** la)
            S.set_law(law, a0)
            aR, aZ = S.aqual(FR, FZ)
            d = departures(P, law, a0, FR, aR, gN)
            for v in ("gas", "Pgas"):
                res[f"{v}_{la:.1f}"] = d[v]
        for i in range(len(m.R)):
            vals = [nm, f"{m.R[i]:.4f}", f"{gN[i]:.6e}"] + [f"{res[k][i]:.6f}" for k in cols[3:]]
            fh.write(",".join(vals) + "\n")
        fh.flush()
        r = lambda k: np.sqrt(np.nanmean(res[k] ** 2))
        print(f"{nm}: {len(m.R)} pts, grid {m.grid.NR}x{m.grid.NZ}, at log10 a0 = -10.0 rms gas {r('gas_-10.0'):.4f} "
              f"Pgas {r('Pgas_-10.0'):.4f}, {time.time()-t:.1f} s", flush=True)
    fh.close()
    print(f"[run time {time.time()-T00:.0f} s]")


# ---------------------------------------------------------------------------------------------- parts upsvar and relvar
_D7 = [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
_B8 = [-0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8]
UPS_A0 = 1.05e-10
UPS_OFFSETS = [(round(-1.0 + 0.2 * k, 1), 0.0) for k in range(11) if k != 5] + [(u, v) for u in _D7 for v in _B8]


def ups_col(var, sd, sb):
    return f"{var}_T11_a{UPS_A0:.4e}_d{sd:+.2f}_b{sb:+.2f}"


def part_cases(var):
    """var 'gas': the full solution at each case (part upsvar); var 'rgas': the relation alone (part relvar)."""
    path = arg("--out")
    gals, gas = T16.load_inputs(HERE)
    names = galaxy_names(gals)
    law = AL.make_law("T11")
    a0 = AL.a0_units(UPS_A0)
    cols = [ups_col(var, sd, sb) for sd, sb in UPS_OFFSETS]
    fh = open(path, "w", newline="\n", encoding="utf-8")
    fh.write("galaxy,R," + ",".join(cols) + "\n")
    for nm in names:
        t = time.time()
        m = T16.GalaxyModel(nm, gals[nm], gas[nm])
        m.density()
        rd, rg, rb = m.rho_parts
        hasb = m.bulge is not None
        S = AL.LawSolver(m.grid, law, a0)
        P = Probe(m)
        vals = {}
        for sd, sb in UPS_OFFSETS:
            if sb != 0.0 and not hasb:
                continue
            _, FR, FZ = S.newton((10 ** sd) * rd + rg + (10 ** sb) * rb)
            gN = P.mid(FR)
            if var == "gas":
                aR, aZ = S.aqual(FR, FZ)
                v = departures(P, law, a0, FR, aR, gN)["gas"]
            else:
                relmid = law.nu(gN / a0) * gN
                relpw = law.nu(np.abs(FR) / a0) * FR
                v = np.where(gN > 0, lg(P.avg(relpw)) - lg(relmid), np.nan)
            vals[ups_col(var, sd, sb)] = v
        for i in range(len(m.R)):
            fh.write(",".join([nm, f"{m.R[i]:.4f}"] + [(f"{vals[k][i]:.6f}" if k in vals else "") for k in cols]) + "\n")
        fh.flush()
        print(f"{nm}: {len(vals)} cases, {time.time()-t:.1f} s", flush=True)
    fh.close()
    print(f"[run time {time.time()-T00:.0f} s]")


if __name__ == "__main__":
    part = arg("--part", "gamma")
    if part == "gamma":
        part_gamma()
    elif part == "upsvar":
        part_cases("gas")
    elif part == "relvar":
        part_cases("rgas")
    else:
        raise SystemExit(f"unknown part {part}")
