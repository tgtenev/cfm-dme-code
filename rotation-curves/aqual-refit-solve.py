"""The full AQUAL solution of every galaxy of the SPARC sample of Paper I, Sect. 4.1, on a grid of the scale a0 and of
the law's free volume, with the disk solver and galaxy models of aqual-disk-solver.py: the departure tables that the
refits read (aqual-refit-fit.py, constrained_fit.py, coarse_grained_fit.py).

RESULTS SUPPORTED
  Paper I, Sect. 4.2 and Paper II, Sect. 6.1: every fit of the full AQUAL solution in the disks (through
  aqual-refit-fit.py and constrained_fit.py). Part kuzmin validates the solver with the law and the scale as
  parameters.

WHAT IT COMPUTES
  For each galaxy of aqual-disk-points.csv (the 153 galaxies and 3,168 points of the sample, quality flag not 3 and
  inclination at least 30 degrees) the model of aqual-disk-solver.py is built: the 3.6 micron disk at a mass-to-light
  ratio of 0.5 in an exponential layer of h_z = 1.12 R_d/8.06, the deconvolved gas in a layer of 0.2 kpc and the bulge
  as a Hernquist sphere fitted at 0.7. The Newtonian midplane field g_N at the data radii is written in total and by
  component (disk at 0.5, gas, bulge at 0.7), and so is the departure of the full AQUAL solution from the algebraic
  relation on that field,
      Gamma = log10 g_full - log10[nu(g_N/a0) g_N],
  at every node of a grid. Law codes: T11 is the empirical function of McGaugh, Lelli and Schombert,
  nu(y) = 1/(1 - exp(-sqrt(y))); T3 is the boost law of Paper I, Eq. (8). T11 is solved at log10 a0 = -10.7 to -9.3 in
  steps of 0.1 (a0 in m/s^2), and T3 at delta_fv = 0.05, 0.2, 0.4, 0.7, 1.2, each at log10 a0 = -10.7 to -9.3 in steps
  of 0.2. One more column, G_T11_T16, holds T11 at a0 = 1.2e-10, the scale of aqual-disk-solver.py, where it equals that
  script's AQUAL departure; the fits do not read it. The fits interpolate Gamma between the nodes (cubic in log10 a0, a
  quartic through the five nodes in delta). Gamma does not depend on the distance (a distance factor scales lengths by
  f and masses by f^2, leaving g_N and g_full unchanged at the scaled radius), and under a uniform scaling s of all
  masses Gamma(s rho, a0) = Gamma(rho, a0/s) exactly. The fits use the second property to carry Gamma to other
  mass-to-light ratios, and parts upsvar and upsvar2 correct that step against direct solutions.

  Parts upsvar and upsvar2 solve each galaxy again at other mass-to-light ratios (the sets 'fit' and 'fit2' of
  UPS_SETS below): the disk ratio at -1.0 to +1.0 dex from 0.5 in steps of 0.2 and, for bulge galaxies, every pair of
  disk ratio -0.4 to +0.8 dex and bulge ratio -0.8 to +0.8 dex from 0.7, in steps of 0.2; for T11 at a0 = 1.05e-10 and
  for the boost law at delta = 0.3, a0 = 0.92e-10, and at delta = 0.03, a0 = 0.95e-10. These solutions supply the fits'
  mode 'corrected' (refit_core.py). Both write one row per point and one column per case.

  Part kuzmin repeats the Kuzmin-disk validation of aqual-disk-solver.py with the parametrized solver, for T11 at
  a0 = 1.05e-10 and 1.2e-10 and the boost law at delta = 0.3 and 1.2.

INPUTS
  rotation-curves/aqual-disk-points.csv and rotation-curves/aqual-disk-gas-profiles.csv (from aqual-disk-densities.py)

OUTPUTS AND COMMANDS, from the repository root (each pair of slices can run at the same time)
  python rotation-curves/aqual-refit-solve.py --part gamma --slice 0/2 --out rotation-curves/aqual-refit-gamma-0.csv
  python rotation-curves/aqual-refit-solve.py --part gamma --slice 1/2 --out rotation-curves/aqual-refit-gamma-1.csv
      (about 1.5 h each; 5,441 s and 5,521 s)
  python rotation-curves/aqual-refit-solve.py --part upsvar --slice 0/2 --out rotation-curves/aqual-refit-upsvar-0.csv
  python rotation-curves/aqual-refit-solve.py --part upsvar --slice 1/2 --out rotation-curves/aqual-refit-upsvar-1.csv
  python rotation-curves/aqual-refit-solve.py --part upsvar2 --slice 0/2 --out rotation-curves/aqual-refit-upsvar2-0.csv
  python rotation-curves/aqual-refit-solve.py --part upsvar2 --slice 1/2 --out rotation-curves/aqual-refit-upsvar2-1.csv
      (upsvar about 23 min for both slices, upsvar2 25 and 31 min)
  python rotation-curves/aqual-refit-solve.py --part kuzmin --out rotation-curves/aqual-refit-kuzmin-results.txt
      (under a minute)
  --only G1,G2 restricts the galaxies, as a quick test. Needs numpy.
"""
import csv, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import aqual_law as AL

T16 = AL.T16
T00 = time.time()
ARGS = sys.argv


def arg(name, default=None):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default


LA_T11 = [round(-10.7 + 0.1 * k, 1) for k in range(15)]       # log10 a0 (m/s^2), T11: -10.7 to -9.3 by 0.1
LA_T3 = [round(-10.7 + 0.2 * k, 1) for k in range(8)]         # T3: -10.7 to -9.3 by 0.2
T3_DELTAS = [0.05, 0.2, 0.4, 0.7, 1.2]
_lines = []


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def galaxy_names(gals):
    names = sorted(gals)
    if arg("--only"):
        names = [x for x in arg("--only").split(",") if x in gals]
    k, n = (int(x) for x in arg("--slice", "0/1").split("/"))
    return [nm for i, nm in enumerate(names) if i % n == k]


def build(nm, gals, gas):
    m = T16.GalaxyModel(nm, gals[nm], gas[nm])
    rho = m.density()
    return m, rho


def midplane(m):
    Rf1 = m.grid.Rf[1:]
    return lambda F: np.interp(m.R, Rf1, F[:, 0])


def gamma_of(S, law, a0, FR, FZ, gN, at):
    S.set_law(law, a0)
    aR, aZ = S.aqual(FR, FZ)
    gF = at(aR)
    with np.errstate(invalid="ignore", divide="ignore"):
        G = np.log10(gF) - np.log10(law.nu(gN / a0) * gN)
    return np.where((gN > 0) & (gF > 0), G, np.nan), S.aqual_iters


def part_gamma():
    path = arg("--out")
    gals, gas = T16.load_inputs(HERE)
    names = galaxy_names(gals)
    laws = [("T11", None, la) for la in LA_T11] + [("T11", None, "T16")]
    laws += [("T3", d, la) for d in T3_DELTAS for la in LA_T3]
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
        m, rho = build(nm, gals, gas)
        rd, rg, rb = m.rho_parts
        S = AL.LawSolver(m.grid, lawobj["T11"], AL.A0_T16)
        at = midplane(m)
        _, FR, FZ = S.newton(rho)
        gN = at(FR)
        comp = [at(S.newton(p)[1]) for p in (rd, rg, rb)]
        res, itmax = {}, 0
        for c, d, la in laws:
            law = lawobj["T11"] if c == "T11" else lawobj[("T3", d)]
            a0 = AL.A0_T16 if la == "T16" else AL.a0_units(10.0 ** la)
            res[(c, d, la)], it = gamma_of(S, law, a0, FR, FZ, gN, at)
            itmax = max(itmax, it)
        for i in range(len(m.R)):
            vals = [nm, f"{m.R[i]:.4f}", f"{gN[i]:.6e}", f"{comp[0][i]:.6e}", f"{comp[1][i]:.6e}", f"{comp[2][i]:.6e}",
                    f"{m.grid.NR}x{m.grid.NZ}", str(itmax)]
            vals += [f"{res[x][i]:.6f}" for x in laws]
            fh.write(",".join(vals) + "\n")
        fh.flush()
        print(f"{nm}: {len(m.R)} points, grid {m.grid.NR}x{m.grid.NZ}, max AQUAL iterations {itmax}, "
              f"Gamma(T11, -10.0) mean {np.nanmean(res[('T11', None, -10.0)]):+.4f}, {time.time()-t:.1f} s", flush=True)
    fh.close()
    print(f"[run time {time.time()-T00:.0f} s]")


# The direct solutions at other mass-to-light ratios, one row per data point and one column per case, named
# <law>_a<a0>_d<dlog10 Yd>_b<dlog10 Yb> with <law> = T11 or T3d<delta>. Each set is a list of blocks (law cases,
# ratio offsets in dex from 0.5 and 0.7); offsets that change the bulge are solved for bulge galaxies only.
#   'fit' and 'fit2' supply the correction of the fits' mode 'corrected': the disk ratio at -1.0 to +1.0 dex in steps
#     of 0.2 (bulge at 0.7), and for bulge galaxies the grid of disk ratio -0.4 to +0.8 dex by bulge ratio -0.8 to
#     +0.8 dex, both in steps of 0.2; for T11 at a0 = 1.05e-10, the boost law at delta 0.3, a0 = 0.92e-10, and at
#     delta 0.03, a0 = 0.95e-10. ('fit' holds the first cases; 'fit2' completes the grids.)
_D7 = [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
_B8 = [-0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8]
UPS_SETS = {
    "fit": [([("T11", None, 1.05e-10), ("T3", 0.3, 0.92e-10)],
             [(-0.4, 0.0), (-0.2, 0.0), (0.2, 0.0), (0.4, 0.0), (0.6, 0.0), (0.8, 0.0), (0.0, -0.2), (0.0, 0.2)])],
    "fit2": [([("T11", None, 1.05e-10), ("T3", 0.3, 0.92e-10)],
              [(-1.0, 0.0), (-0.8, 0.0), (-0.6, 0.0), (1.0, 0.0)]
              + [(u, v) for u in _D7 for v in _B8 if not (u == 0.0 and abs(v) == 0.2)]),
             ([("T3", 0.03, 0.95e-10)],
              [(round(-1.0 + 0.2 * k, 1), 0.0) for k in range(11) if k != 5] + [(u, v) for u in _D7 for v in _B8])],
}


def law_tag(law, delta):
    return "T11" if law == "T11" else f"T3d{delta:g}"


def ups_col(law, delta, a0si, sd, sb):
    return f"{law_tag(law, delta)}_a{a0si:.4e}_d{sd:+.2f}_b{sb:+.2f}"


def part_upsvar(which):
    """Gamma of the direct solution at the case's law, scale and ratios, relative to the relation on that model's own
    field. Bulge columns are empty for bulgeless galaxies."""
    path = arg("--out")
    gals, gas = T16.load_inputs(HERE)
    names = galaxy_names(gals)
    blocks = UPS_SETS[which]
    lawobj = {}
    for cases, _ in blocks:
        for c, d, a0si in cases:
            lawobj.setdefault((c, d), AL.make_law(c, d))
    cols = []
    for cases, scal in blocks:
        cols += [ups_col(c, d, a0si, sd, sb) for sd, sb in scal for c, d, a0si in cases]
    fh = open(path, "w", newline="\n", encoding="utf-8")
    fh.write("galaxy,R," + ",".join(cols) + "\n")
    for nm in names:
        t = time.time()
        m, rho = build(nm, gals, gas)
        rd, rg, rb = m.rho_parts
        hasb = m.bulge is not None
        S = AL.LawSolver(m.grid, AL.make_law("T11"), AL.A0_T16)
        at = midplane(m)
        vals, newton = {}, {}
        for cases, scal in blocks:
            for sd, sb in scal:
                if sb != 0.0 and not hasb:
                    continue
                if (sd, sb) not in newton:
                    r = (10 ** sd) * rd + rg + (10 ** sb) * rb
                    _, FR, FZ = S.newton(r)
                    newton[(sd, sb)] = (FR, FZ, at(FR))
                FR, FZ, gN = newton[(sd, sb)]
                for c, d, a0si in cases:
                    vals[ups_col(c, d, a0si, sd, sb)] = gamma_of(S, lawobj[(c, d)], AL.a0_units(a0si), FR, FZ, gN, at)[0]
        for i in range(len(m.R)):
            fh.write(",".join([nm, f"{m.R[i]:.4f}"] + [(f"{vals[k][i]:.6f}" if k in vals else "") for k in cols]) + "\n")
        fh.flush()
        print(f"{nm}: {len(vals)} cases, {time.time()-t:.1f} s", flush=True)
    fh.close()
    print(f"[run time {time.time()-T00:.0f} s]")


def part_kuzmin():
    out("AQUAL refit: the Kuzmin disk with the parametrized solver (aqual_law.LawSolver)")
    out("=" * 100)
    out("The Kuzmin disk's MOND field is exactly nu(|g_N|/a0) g_N in AQUAL for any interpolating function (Famaey and")
    out("McGaugh 2012 Sect. 6.1.2; Brada and Milgrom 1995). Grid and sheet as in aqual-disk-solver.py --part validate (sheet")
    out("in the first cell row). Errors: maxima of |g_R(num)/g_R(exact) - 1| over 0.1 b < R < 30 b at the row nearest")
    out("z = b, and in the plane against the razor-thin value (reached linearly in dz0). Also the plane's departure from the")
    out("algebraic relation, log10[g_R/(nu(g_R/a0) g_R)], numerical against exact, at R = b and 2 b.")
    cases = [("T11", None, 1.05e-10), ("T3", 0.3, 0.92e-10), ("T3", 1.2, 0.80e-10), ("T11", None, 1.2e-10)]
    for (M, b) in ((1e10, 2.0), (1e9, 1.0), (1e8, 3.0)):
        for dzf in (0.01, 0.005):
            g = T16.kuzmin_grid(b, dzf)
            Rf = g.Rf
            rho = np.zeros((g.NR, g.NZ))
            Mann = M * b * (1 / np.sqrt(Rf[:-1] ** 2 + b * b) - 1 / np.sqrt(Rf[1:] ** 2 + b * b))
            rho[:, 0] = Mann / (4 * np.pi * g.V[:, 0])
            S = AL.LawSolver(g, AL.make_law("T11"), AL.A0_T16)
            phi, FR, FZ = S.newton(rho)
            R = g.Rf[1:]; sel = (R > 0.1 * b) & (R < 30 * b)
            j = int(np.argmin(np.abs(g.zc - b)))
            for c, d, a0si in cases:
                law = AL.make_law(c, d)
                a0 = AL.a0_units(a0si)
                S.set_law(law, a0)
                aR, aZ = S.aqual(FR, FZ, tol=1e-10)
                res = S.aqual_residual(aR, aZ, FR, FZ)
                gRz, gzz = T16.kuzmin_exact(M, b, R[sel], g.zc[j])
                nz = law.nu(np.hypot(gRz, gzz) / a0)
                eAz = np.max(np.abs(aR[sel, j] / (nz * gRz) - 1))
                gR0, gz0 = T16.kuzmin_exact(M, b, R[sel], 0.0)
                gM0 = law.nu(np.hypot(gR0, gz0) / a0) * gR0
                eA0 = np.max(np.abs(aR[sel, 0] / gM0 - 1))
                row = []
                for f_ in (1.0, 2.0):
                    k = int(np.argmin(np.abs(R - f_ * b)))
                    gRx, gzx = T16.kuzmin_exact(M, b, R[k], 0.0)
                    ex = math.log10(float(law.nu(math.hypot(gRx, gzx) / a0)) / float(law.nu(gRx / a0)))
                    num = math.log10(aR[k, 0] / (float(law.nu(FR[k, 0] / a0)) * FR[k, 0]))
                    row.append(f"R={R[k]/b:.2f}b: {num:+.4f} vs {ex:+.4f}")
                out(f"   M {M:.0e}, b {b:g} kpc (GM/b^2 = {AL.G*M/b**2/a0:.2f} a0), dz0 {dzf:g} b, {c}{'' if d is None else ' delta=%g' % d} "
                    f"a0 {a0si:.3g}: z=b {eAz:.1e}, plane {eA0:.1e}, residual {res:.0e}, {S.aqual_iters} it | Gamma in the plane {'; '.join(row)}")
    out(f"\n[run time {time.time()-T00:.0f} s]")


if __name__ == "__main__":
    part = arg("--part", "gamma")
    if part == "gamma":
        part_gamma()
    elif part == "upsvar":
        part_upsvar("fit")
    elif part == "upsvar2":
        part_upsvar("fit2")
    elif part == "kuzmin":
        part_kuzmin()
        if arg("--out"):
            with open(arg("--out"), "w", newline="\n", encoding="utf-8") as fh:
                fh.write("\n".join(_lines) + "\n")
