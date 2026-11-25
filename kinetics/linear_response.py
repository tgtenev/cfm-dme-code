"""The linear response of the stored axes to a rigid displacement of their arrangement.

The static aligned sphere of axfem.Body (stretch 4 at its least, 24.8 at the grid's edge r = 5.4 R) is displaced:
axes and stretches are laid radially about the point d0 z_hat (d0 = 0.01 R) while the mass stays at the origin, and
the field is solved with that tensor frozen. At every ring the script measures the angle, relative to the old axis,
of the field, of the flux and of the vector of rule B, and turns it into a shift of the centre of the axes that one
round of smoothings would lay (axfem.Mesh.ring_shift). The rules for the direction of a new layer are those of
kinetics/steady_dipole.py: A across the field, B across the gradient inside a freshly smoothed domain, C across the
transmitted flux. The share of the offset removed by one round is f = (d0 - new centre)/d0: f_A from the field,
f_B from rule B's vector, f_C from the flux. Locally f_B = nu f_A and f_C = nu^2 f_A exactly, so the test decides
the memory through f_A(r).

Cases: (a) axes and stretches displaced, Phi = 0 at r_out (the fabric beyond holds the static arrangement about the
body); (b) the same with Phi at r_out continuing the displaced arrangement; (c) axes only, the stretches left about
the body. Each is solved on the polar grid (128 x 215 cells) and by the dipole equation of l1.py (20,000 nodes),
at the fiducial free volume of 0.19.

Supports Paper II Appendix A.3: beyond about 2 body radii the field and the flux follow the displaced axes, and one
smoothing removes under 1 percent of the offset (in case (a), f_C falls below 0.01 at r = 2 R).

Inputs: none (imports kinetics/axfem.py and kinetics/l1.py). Output: kinetics/linear_response-results.txt.
Run from the repository root (about a second):
    python kinetics/linear_response.py --out kinetics/linear_response-results.txt
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import axfem as A  # noqa: E402
import l1  # noqa: E402

OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
DELTA = 0.19                     # the fiducial free volume
R_OUT = 5.4
D0 = 0.01
RADII = (0.1, 0.3, 0.6, 0.9, 1.1, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.2, 5.3)
_lines = []


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def solve2d(mesh, body, stat, d0, stretch, outer):
    R, z = mesh.rq * np.sin(mesh.tq), mesh.rq * np.cos(mesh.tq)
    rp = np.hypot(R, z - d0)
    beta = A.wrap_half(np.arctan2(R, z - d0) - mesh.tq)
    nu = stat.nu_of(rp) if stretch else stat.nu_of(mesh.rq)
    K = mesh.stiffness(*mesh.tensor(beta[:, :4], nu[:, :4]))
    b = mesh.load(body.rho(mesh.rq[:, :4]))
    rb, tb = mesh.node_r[mesh.bnd], mesh.node_t[mesh.bnd]
    if outer == "body":
        bv = np.zeros(len(mesh.bnd))
    else:
        bv = stat.phi_of(np.hypot(rb * np.sin(tb), rb * np.cos(tb) - d0))
    phi = mesh.solve(K, b, bv)
    gr, gt, Fr, Ft = A.field_and_flux(mesh, phi, beta[:, 4], nu[:, 4])
    bA, bB, bC = A.rule_vectors(gr, gt, Fr, Ft, beta[:, 4])
    return [-mesh.ring_shift(A.wrap_half(bq - beta[:, 4])) / d0 for bq in (bA, bB, bC)]


def main():
    t0 = time.time()
    out("LINEAR RESPONSE: ONE ROUND OF RE-LAYING AFTER A RIGID DISPLACEMENT d0 = 0.01 R OF THE ARRANGEMENT")
    out("=" * 118)
    out("f = (d0 - centre of the newly laid axes)/d0, the share of the offset that one round removes; f_A from the field,")
    out("f_B from rule B's vector, f_C from the flux. The local estimate: f_A = mu^2, f_B = mu, f_C = 1 (mu = 1/nu).")
    out("Locally f_A/mu^2 = f_B/mu = f_C, so one column gives the share over the estimate for all three rules.")
    out("2D: polar grid 128 x 215 cells; l1: the dipole equation on 20,000 radial nodes. Columns per rule: 2D, l1, and")
    out("the l1 value over the estimate.")
    mesh = A.Mesh(A.r_nodes(dxi=0.02, r_a=0.2, r_out=R_OUT), 128)
    delta = DELTA
    law = A.Law(delta)
    body = A.Body(law)
    stat = A.Static(body, R_OUT)
    P = l1.L1(body, R_OUT)
    out("")
    out(f"delta = {delta}: least stretch 4 at r = {body.r_numin:.2f} R, stretch {P.at(P.nu, 1.0):.2f} at the surface and "
        f"{P.at(P.nu, R_OUT):.2f} at r_out")
    cases = (("(a) axes and stretches displaced; Phi = 0 at r_out (outer edge held about the body)", True, "body"),
             ("(b) axes and stretches displaced; Phi at r_out continues the displaced arrangement", True, "arr"),
             ("(c) axes only displaced, stretches left about the body; Phi = 0 at r_out", False, "body"))
    for lab, stretch, outer in cases:
        f2 = solve2d(mesh, body, stat, D0, stretch, outer)
        fA1 = 1.0 - P.rigid(1.0, stretch=stretch, outer=outer)
        out("  " + lab)
        out("     r/R     nu  |   f_A 2D     f_A l1   |   f_B 2D     f_B l1   |   f_C 2D     f_C l1   | over the estimate |"
            " field centre / d0   flux centre / d0")
        for rr in RADII:
            nu = float(P.at(P.nu, rr))
            a1 = float(P.at(fA1, rr))
            f2r = [float(np.interp(rr, mesh.rc_ring, f)) for f in f2]
            out(f"    {rr:5.2f} {nu:6.2f} | {f2r[0]: .3e} {a1: .3e} | {f2r[1]: .3e} {a1 * nu: .3e} | {f2r[2]: .3e} "
                f"{a1 * nu * nu: .3e} |     {a1 * nu * nu: .3e}     |   {1 - a1: .6f}          {1 - a1 * nu * nu: .4f}")
    # the shares over the arrangement, case (a) and (c), l1
    sel = (P.r > 1.6) & (P.r < 4.0)
    for lab, stretch in (("(a)", True), ("(c)", False)):
        fA1 = 1.0 - P.rigid(1.0, stretch=stretch, outer="body")
        ratio = fA1[sel] * P.nu[sel] ** 2
        out(f"  {lab} over 1.6 < r < 4.0 R: f_A / mu^2 = f_C from {ratio.min():.2e} to {ratio.max():.2e}")
    fA1 = 1.0 - P.rigid(1.0, stretch=True, outer="body")
    inside = (P.r > 0.3) & (P.r < 1.2)
    k = np.argmax(fA1[inside] * P.nu[inside] ** 2)
    out(f"  (a) inside the body: largest f_C = {(fA1[inside] * P.nu[inside] ** 2)[k]:.2f} at r = {P.r[inside][k]:.2f} R "
        f"(f_A/mu^2 there; above 1 the flux points past the body)")
    for thr in (0.1, 0.01):
        below = P.r[(P.r > 1.0) & (fA1 * P.nu ** 2 < thr)]
        out(f"  (a) f_C falls below {thr} at r = {below[0]:.2f} R (stretch {float(P.at(P.nu, below[0])):.2f})")
    out(f"\ntotal {time.time() - t0:.0f} s")
    if OUT:
        with open(OUT, "w", newline="\n", encoding="utf-8") as fh:
            fh.write("\n".join(_lines) + "\n")


if __name__ == "__main__":
    main()
