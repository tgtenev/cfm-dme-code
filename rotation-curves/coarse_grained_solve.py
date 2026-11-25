"""The full AQUAL solution in the disks coarse-grained at the catchment length: the departure tables Gamma of the
coarse-grained solution, on the grid of aqual-refit-solve.py, for coarse_grained_fit.py.

RESULTS SUPPORTED
  Paper II, Sect. 6.1: the fits of the full solution coarse-grained at catchment lengths of 100 and 200 pc
  (coarse_grained_fit.py), 1.5 to 1.8 SD worse than the algebraic relation, and the free volume of the two-constant fit
  falling to 0.005 or below.

WHAT IT COMPUTES
  The coarse graining is the scale-selective coupling of Paper I, Sect. 3.7 (form B): the stretch acts on the field
  averaged over a ball one catchment length ell across, and finer structure stays Newtonian,
      g = g_N[rho] + AQUAL[rho_ell] - g_N[rho_ell],    rho_ell = the ball average of rho (catchment.py),
  so the non-Newtonian part of the field is computed from the smoothed source. Codes: B100 and B200 are form B at
  ell = 100 and 200 pc. For each galaxy of aqual-disk-points.csv the model of aqual-disk-solver.py is built (as in
  aqual-refit-solve.py), and Gamma = log10 g_full - log10[nu(g_N/a0) g_N] is written on the grid of aqual-refit-solve.py
  with the same columns: T11 (the empirical function of McGaugh, Lelli and Schombert) at log10 a0 = -10.7 to -9.3 in
  steps of 0.1 and at 1.2e-10, and T3 (the boost law of Paper I, Eq. (8)) at delta_fv = 0.05, 0.2, 0.4, 0.7, 1.2 at
  the a0 nodes of --t3la. The T3 tables at four a0 nodes (files *-T3r-*) are completed to the full a0 grid by
  coarse_grained_merge.py.

INPUTS
  rotation-curves/aqual-disk-points.csv and rotation-curves/aqual-disk-gas-profiles.csv

OUTPUTS AND COMMANDS, from the repository root (the slices of one table can run at the same time)
  python rotation-curves/coarse_grained_solve.py --ell 0.1 --laws T11 --slice k/3 --out rotation-curves/coarse-gamma-B100-T11-k.csv
      for k = 0, 1, 2 (about 1 h each)
  python rotation-curves/coarse_grained_solve.py --ell 0.2 --laws T11 --slice k/4 --out rotation-curves/coarse-gamma-B200-T11-k.csv
      for k = 0, 1, 2, 3 (about 15 min each)
  python rotation-curves/coarse_grained_solve.py --ell 0.1 --laws T3 --t3la -10.7,-10.3,-9.9,-9.5 --slice k/2 --out rotation-curves/coarse-gamma-B100-T3r-k.csv
      for k = 0, 1 (about 2 h each)
  --ell is the catchment length in kpc; --laws T11, T3 or all (default); --only G1,G2 restricts the galaxies, as a
  quick test. Needs numpy.
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import aqual_law as AL
import catchment as C

T16 = AL.T16
T00 = time.time()
ARGS = sys.argv


def arg(name, default=None):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default


LA_T11 = [round(-10.7 + 0.1 * k, 1) for k in range(15)]
LA_T3 = [round(-10.7 + 0.2 * k, 1) for k in range(8)]
T3_DELTAS = [0.05, 0.2, 0.4, 0.7, 1.2]


def galaxy_names(gals):
    names = sorted(gals)
    if arg("--only"):
        names = [x for x in arg("--only").split(",") if x in gals]
    k, n = (int(x) for x in arg("--slice", "0/1").split("/"))
    return [nm for i, nm in enumerate(names) if i % n == k]


def midplane(m):
    Rf1 = m.grid.Rf[1:]
    return lambda F: np.interp(m.R, Rf1, F[:, 0])


def gam(gF, gN, law, a0):
    with np.errstate(invalid="ignore", divide="ignore"):
        G = np.log10(gF) - np.log10(law.nu(gN / a0) * gN)
    return np.where((gN > 0) & (gF > 0), G, np.nan)


class FormB:
    """Form B for one galaxy model: smooth the density components once per ell, then one plain AQUAL solve on the
    smoothed source per (law, a0), warm-started from the previous solution. The field is the Newtonian field of the
    full source plus the AQUAL field of the smoothed source minus its Newtonian field."""

    def __init__(self, m, S, ell):
        self.m, self.S, self.ell = m, S, ell
        op = C.BallAverage(m.grid.Rc, m.grid.zc, ell, r_odd=False, z_odd=False, comp="scalar", zcut="auto")
        self.parts = []
        self.dm = []
        for p in m.rho_parts:
            if np.any(p > 0):
                ps, d = C.smooth_density(m.grid, p, ell, op=op)
            else:
                ps, d = p.copy(), 0.0
            self.parts.append(ps); self.dm.append(d)
        self.start = None
        self.key = None

    def setup(self, sd=1.0, sg=1.0, sb=1.0):
        rd, rg, rb = self.m.rho_parts
        rds, rgs, rbs = self.parts
        _, self.FR, self.FZ = self.S.newton(sd * rd + sg * rg + sb * rb)
        _, self.FRs, self.FZs = self.S.newton(sd * rds + sg * rgs + sb * rbs)
        self.start = None

    def solve(self, law, a0, at):
        self.S.set_law(law, a0)
        sR, sZ = self.S.aqual(self.FRs, self.FZs, start=self.start)
        self.start = (sR, sZ)
        return at(self.FR) + at(sR) - at(self.FRs), self.S.aqual_iters


# ---------------------------------------------------------------------------------------------- the tables
def part_gamma():
    path = arg("--out")
    ell = float(arg("--ell", "0.1"))
    gals, gas = T16.load_inputs(HERE)
    names = galaxy_names(gals)
    which = arg("--laws", "all")            # T11 (with the 1.2e-10 column), T3, or all
    laws = []
    if which in ("T11", "all"):
        laws += [("T11", None, la) for la in LA_T11] + [("T11", None, "T16")]
    if which in ("T3", "all"):
        la3 = [float(x) for x in arg("--t3la").split(",")] if arg("--t3la") else LA_T3
        laws += [("T3", d, la) for d in T3_DELTAS for la in la3]
    lawobj = {"T11": AL.make_law("T11")}
    for d in T3_DELTAS:
        lawobj[("T3", d)] = AL.make_law("T3", d)
    cols = ["galaxy", "R", "gN", "gNd", "gNg", "gNb", "grid", "itmax"]
    cname = lambda c, d, la: (f"G_T11_{la}" if c == "T11" else f"G_T3_{d:g}_{la}")
    cols += [cname(*x) for x in laws]
    fh = open(path, "w", newline="\n", encoding="utf-8")
    fh.write(",".join(cols) + "\n")
    for nm in names:
        t = time.time()
        m = T16.GalaxyModel(nm, gals[nm], gas[nm])
        rho = m.density()
        rd, rg, rb = m.rho_parts
        at = midplane(m)
        S = AL.LawSolver(m.grid, lawobj["T11"], AL.A0_T16)
        _, FR, FZ = S.newton(rho)
        gN = at(FR)
        comp = [at(S.newton(p)[1]) for p in (rd, rg, rb)]
        B = FormB(m, S, ell)
        B.setup()
        res, itmax, prev = {}, 0, None
        for c, d, la in laws:
            law = lawobj["T11"] if c == "T11" else lawobj[("T3", d)]
            a0 = AL.A0_T16 if la == "T16" else AL.a0_units(10.0 ** la)
            if (c, d) != prev:
                B.start = None
            prev = (c, d)
            gF, it = B.solve(law, a0, at)
            res[(c, d, la)] = gam(gF, gN, law, a0)
            itmax = max(itmax, it)
        for i in range(len(m.R)):
            vals = [nm, f"{m.R[i]:.4f}", f"{gN[i]:.6e}", f"{comp[0][i]:.6e}", f"{comp[1][i]:.6e}", f"{comp[2][i]:.6e}",
                    f"{m.grid.NR}x{m.grid.NZ}", str(itmax)]
            vals += [f"{res[x][i]:.6f}" for x in laws]
            fh.write(",".join(vals) + "\n")
        fh.flush()
        print(f"{nm}: {len(m.R)} points, form B ell {ell}, max iterations {itmax}, "
              f"Gamma({cname(*laws[0])}) mean {np.nanmean(res[laws[0]]):+.4f}, {time.time()-t:.1f} s", flush=True)
    fh.close()
    print(f"[run time {time.time()-T00:.0f} s]")


if __name__ == "__main__":
    part_gamma()
