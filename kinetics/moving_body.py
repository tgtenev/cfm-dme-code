"""The steady state of a body moving through the fabric, under three rules for the direction of a new layer.

The rules are those of kinetics/steady_dipole.py: A across the field, B across the gradient inside a freshly
smoothed domain, C across the transmitted flux, which Paper II adopts. For each rule and speed: the fixed point of
kinetics.Mover (field solve with the current tensor, steady memory of axes and stretches along the streamlines,
Anderson mixing), started from the first-order dipole solution of l1.py. Speeds are given as v T0 over the body's
radius R (the steady state depends on v T0 only through that ratio), chosen so that the predicted trail 2 v T nu^p at
r = 2 R is 0.01, 0.03, 0.1 and 1 times the radius. Reported against radius: the trail of the axes and of the field
(their centres on the z-axis behind the body), the predicted trail and the measured prefactor, the first-order dipole
solution, and the boost ahead of the body, behind it and across the direction of motion over the static boost on the
same grid. The free volume is the fiducial 0.19, with a0 = 0.9426e-10 m/s^2.

The first speed of a run starts from the dipole solution, each later one from the previous solution scaled by the
ratio of the speeds. The core of the solve (kinetics.Mover.solve) is the region where the first-order trail exceeds a
tenth of the radius, at least --r-core (default 0.15 R) and at most --r-core-max (default 1 R).

Supports Paper II Appendix A.3: the full nonlinear relaxations confirm the first-order trail of
kinetics/steady_dipole.py to within 10 percent.

Inputs: none (imports kinetics/axfem.py, kinetics/kinetics.py and kinetics/l1.py).
Output: kinetics/moving_body-results.txt, which joins the outputs of the runs below, in this order, with the notes
that it states; runs stopped by hand keep the speeds they finished, since the output file is written after every
speed. Run from the repository root, one rule per run:
    python kinetics/moving_body.py --rules A --levels 0,1,2 --whole-rounds --last-cycle --out <file>     (about 11 min)
    python kinetics/moving_body.py --rules B --levels 0,1,2 --whole-rounds --last-cycle --out <file>     (about 11 min)
    python kinetics/moving_body.py --rules C --levels 0,1,2 --r-core 0.3 --out <file>      (level 2 stopped by hand)
    python kinetics/moving_body.py --rules A --levels 2,3 --cycles 2 --max-it 250 --whole-rounds --last-cycle --out <file>
    python kinetics/moving_body.py --rules B --levels 2,3 --cycles 2 --max-it 250 --whole-rounds --last-cycle --out <file>
        (level 3 of these two stopped by hand)
    python kinetics/moving_body.py --fine --rules A --levels 0 --whole-rounds --last-cycle --out <file>  (about 25 min)
    python kinetics/moving_body.py --fine --rules C --levels 0 --r-core 0.3 --last-cycle --out <file>    (about 25 min)
Options: --levels 0,1,2,3 (0.01, 0.03, 0.1, 1 of the radius); --fine (doubled resolution); --static-start (start from the
static state instead of the dipole solution); --whole-rounds (plain rounds of the whole state rather than of the core
alone) and --last-cycle (keep the state after the last cycle rather than the one with the least outer residual);
--cycles, --core-its, --max-it, --m-and, --mix-core, --tol for the solve.
The --out path is taken from the current directory.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import axfem as A  # noqa: E402
import kinetics as KN  # noqa: E402
import l1  # noqa: E402


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


DELTA = 0.19                     # the fiducial free volume
RULES = arg("--rules", "ABC")
LEVELS = [int(c) for c in arg("--levels", "0,1,2").split(",")]
OUT = arg("--out", None)
FINE = "--fine" in sys.argv
STATIC_START = "--static-start" in sys.argv
WHOLE_ROUNDS = "--whole-rounds" in sys.argv   # plain rounds of the whole state, not of the core alone
LAST_CYCLE = "--last-cycle" in sys.argv       # keep the state after the last cycle
R_OUT = 5.4
A0 = 0.9426e-10                  # m/s^2, the law's scale at the fiducial free volume
REF_R = 2.0
TOL = float(arg("--tol", "1e-9"))
R_CORE = float(arg("--r-core", "0.15"))
R_CORE_MAX = float(arg("--r-core-max", "1.0"))
LEVEL_FRAC = (0.01, 0.03, 0.1, 1.0)
RADII = (0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
_lines = []


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def static_state(mov):
    """The static aligned fixed point on the same grid (v = 0)."""
    m, law = mov.m, mov.law
    beta = np.zeros(m.Ne)
    nu = mov.stat.nu_of(m.rc)
    for _ in range(80):
        phi, gr, gt, Fr, Ft = mov.solve_field(beta, nu)
        new = np.sqrt(nu * law.nu_capped(np.hypot(gr, gt)))
        ch = np.abs(np.log(new / nu)).max()
        nu = new
        if ch < 1e-12:
            break
    phi, gr, gt, Fr, Ft = mov.solve_field(beta, nu)
    return nu, np.hypot(gr, gt)


def guess(mov, P, q, vT0):
    """The first-order dipole solution of l1.py mapped onto the grid."""
    m = mov.m
    trail_ax, trail_f, pred, phi1 = P.steady(q, vT0)
    d_n = np.clip(-P.at(trail_ax, m.rc), -0.2 * m.rc, 0.2 * m.rc)      # the first order fails near the centre
    beta = A.wrap_half(np.arctan2(m.Rc, m.zc - d_n) - m.tc)
    dphi1 = np.gradient(phi1, P.r)
    Lp = np.where(P.cap, 0.0, mov.law.dnu(P.X))
    nu1 = np.nan_to_num(Lp * dphi1 + vT0 * P.T * P.dnu)
    nu0 = P.at(P.nu, m.rc)
    nu = nu0 + np.clip(P.at(nu1, m.rc), -0.2 * nu0, 0.2 * nu0) * np.cos(m.tc)
    return np.concatenate([beta, np.log(nu)])


def ring_val(m, arr, rr):
    return float(np.interp(rr, m.rc_ring, arr))


def main():
    t_all = time.time()
    law = A.Law(DELTA)
    body = A.Body(law)
    res = dict(dxi=0.01, nt=256) if FINE else dict(dxi=0.02, nt=128)
    mesh = A.Mesh(A.r_nodes(dxi=res["dxi"], r_a=0.2, r_out=R_OUT), res["nt"])
    P = l1.L1(body, R_OUT)
    a0 = A0
    kpc = 3.0857e19
    M_sun = body.M * a0 * kpc ** 2 / 6.674e-11 / 1.989e30
    out(f"STEADY STATE OF A MOVING BODY, delta = {DELTA}, a0 = {a0:.4e} m/s^2")
    out("=" * 118)
    out(f"body: density rho_c (1 - r^2)^2 inside r = R, least static stretch 4 at r = {body.r_numin:.2f} R; grid {mesh.Nt} x "
        f"{mesh.Nr} cells to r = {R_OUT} R (stretch {P.at(P.nu, R_OUT):.1f} there); with R = 1 kpc the mass is {M_sun:.2e} "
        f"solar masses.")
    out("outer edge: Phi = 0 at r_out, and the fabric beyond r_out holds the static arrangement about the body (pinned).")
    out("speeds: v T0 in units of R. With R = 1 kpc, v = (v T0 / R) x 1 kpc / T0 = (v T0 / R) x 3.3e6 km/s at T0 = 295 yr")
    out("and x 1.5e6 km/s at T0 = 650 yr.")
    out("trail = distance of a centre behind the body (positive behind). Centres: mean over theta of the point where the")
    out("line along the axis (or the field) crosses the z-axis, weights sin^3(theta). pred = 2 v T nu^p (T, nu static at r).")
    out("boost: |g| over the static |g| on the same grid, ahead of the body (theta = 0), behind it (pi), across (pi/2),")
    out("and the mean over the sphere of the radial field over its static value.")
    nu_s, g_s = static_state(KN.Mover(mesh, body, 0, 0.0))
    for rule in RULES:
        q = "ABC".index(rule)
        p = 2 - q
        i2 = np.searchsorted(P.r, REF_R)
        pred_unit = 2.0 * P.T[i2] * P.nu[i2] ** p / REF_R             # predicted trail over r at r = 2, per unit v T0
        out("")
        out(f"RULE {rule} (p = {p}): predicted trail at r = 2 R is {pred_unit:.4g} (v T0 / R) R")
        prev = None
        for lev in LEVELS:
            vT0 = LEVEL_FRAC[lev] / pred_unit
            mov = KN.Mover(mesh, body, q, vT0)
            xs = np.concatenate([np.zeros(mesh.Ne), np.log(nu_s)])
            if STATIC_START:
                x0 = xs
            elif prev is not None:
                x0 = xs + (vT0 / prev[0]) * (prev[2] - xs)               # continuation from the previous speed
                x0[:mesh.Ne] = np.clip(x0[:mesh.Ne], -1.2, 1.2)
            else:
                x0 = guess(mov, P, q, vT0)

            def watch(mv):
                c = mesh.ring_centre(mv.m.tc + mv.last["beta"])
                return "axis trail at r = 1, 2, 3: " + ", ".join(f"{-ring_val(mesh, c, rr):.4e}" for rr in (1.0, 2.0, 3.0))

            t0 = time.time()
            # the core: where the first-order trail exceeds a tenth of the radius, the state is strongly nonlinear
            ta1, tf1, pr1, _ = P.steady(q, vT0)
            big = (P.r > 0.01) & (P.r < 3.0) & (ta1 > 0.1 * P.r)
            r_core = min(max(R_CORE, 1.3 * float(P.r[big].max())) if big.any() else R_CORE, R_CORE_MAX)
            r_conv = r_core + 0.05
            out(f"  v T0 / R = {vT0:.3e} (predicted trail at r = 2 R: {LEVEL_FRAC[lev]} of the radius); core r < {r_core:.2f} R, "
                f"convergence measured at r > {r_conv:.2f} R")
            x, it, resid, snaps = mov.solve(x0, tol=TOL, max_it=int(arg("--max-it", "300")), log=out, watch=watch,
                                            r_core=r_core, mix_core=float(arg("--mix-core", "0.5")),
                                            m_and=int(arg("--m-and", "25")), cycles=int(arg("--cycles", "3")),
                                            core_its=int(arg("--core-its", "40")), r_conv=r_conv,
                                            whole_rounds=WHOLE_ROUNDS, last_cycle=LAST_CYCLE)
            fb, fn = resid[0], resid[1]
            cyc_ch = []
            for a_, b_ in zip(snaps[:-1], snaps[1:]):
                ca = mesh.ring_centre(mesh.tc + a_[:mesh.Ne])
                cb = mesh.ring_centre(mesh.tc + b_[:mesh.Ne])
                sel = mesh.rc_ring > r_conv
                cyc_ch.append(float(np.max(np.abs(cb[sel] - ca[sel]) / np.abs(cb[sel]))))
            beta = x[:mesh.Ne]
            nu = np.exp(x[mesh.Ne:])
            phi, gr, gt, Fr, Ft = mov.solve_field(beta, nu)
            psi_g = mesh.tc + np.arctan2(gt, gr)
            psi_n = mesh.tc + beta
            c_ax = mesh.ring_centre(psi_n)
            c_f = mesh.ring_centre(psi_g)
            X = np.hypot(gr, gt)
            boost = mesh.ring(X / g_s)
            grad_r = mesh.ring(gr / g_s)
            wsph = np.sin(mesh.tc_ring) * (mesh.th[1:] - mesh.th[:-1])
            mono = (grad_r * wsph[None, :]).sum(axis=1) / wsph.sum()
            conv = "converged" if max(fb, fn) < TOL else f"NOT converged to {TOL:.0e}"
            out(f"    {it} iterations in all; residual outside r = {r_conv:.2f} R {fb:.1e} rad (axes), {fn:.1e} (ln nu), {conv}; "
                f"inside, after the last plain round, {resid[2]:.1e} rad, {resid[3]:.1e}; march steps {mov.last['K']}; "
                f"{time.time() - t0:.0f} s")
            out(f"    largest relative change of the trail of the axes at r > {r_conv:.2f} R between cycles: "
                + ", ".join(f"{c:.1e}" for c in cyc_ch))
            out("      r/R    nu | trail axes  trail field | pred 2vTnu^p | axes/pred field/pred | first order: axes/pred "
                "| boost ahead  behind  across  mean g_r | trail/r")
            rows = []
            for rr in RADII:
                ta, tf = -ring_val(mesh, c_ax, rr), -ring_val(mesh, c_f, rr)
                pr = float(P.at(pr1, rr))
                ia = int(np.argmin(np.abs(mesh.rc_ring - rr)))
                bA, bB, bX = boost[ia, 0], boost[ia, -1], float(np.interp(np.pi / 2, mesh.tc_ring, boost[ia]))
                out(f"     {rr:4.1f} {float(P.at(P.nu, rr)):6.2f} | {ta: .4e} {tf: .4e} | {pr: .4e}  | {ta / pr: 8.4f} {tf / pr: 8.4f} |"
                    f"       {float(P.at(ta1, rr)) / pr: 8.4f}       | {bA: 8.4f} {bB: 8.4f} {bX: 8.4f} {mono[ia]: 8.4f} | {ta / rr: .3e}")
                rows.append((rr, ta, tf, pr))
            if prev is not None:
                ratio = [rw[1] / pv[1] for rw, pv in zip(rows, prev[1])]
                out(f"    trail of the axes over that at the previous speed (speed ratio {vT0 / prev[0]:.1f}): "
                    + ", ".join(f"r={rw[0]:.1f}: {x_:.3f}" for rw, x_ in zip(rows, ratio)))
            prev = (vT0, rows, x.copy())
            if OUT:
                with open(OUT, "w", newline="\n", encoding="utf-8") as fh:
                    fh.write("\n".join(_lines) + "\n")
    out(f"\ntotal {time.time() - t_all:.0f} s")
    if OUT:
        with open(OUT, "w", newline="\n", encoding="utf-8") as fh:
            fh.write("\n".join(_lines) + "\n")


if __name__ == "__main__":
    main()
