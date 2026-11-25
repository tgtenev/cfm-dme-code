"""Three checks of the spatial geometry that the bend of the fabric and the reweighted constraints give.

1. The scalar curvature that the bend gives the spatial metric. On a flat background the metric (1 + 2u) dr^2 +
   r^2 dOmega^2 has the scalar curvature 4 (r u)'/r^2 at linear order, so the bend's slope fixes u; on a background
   stretched by l the literal analog weights the flux by 1/l^2 rather than by 1/l.
2. The gradient-stiffness channel: the static equation of a gradient energy of the volumetric strain, written for the
   total strain and for the in-plane displacement alone, is a flux law, lap[div(chi grad e)] = 0, for that strain.
3. The reweighted constraints at linear order, in any geometry and for moving sources: with Paper II's phantom density
   in the Hamiltonian constraint the curvature potential equals the lapse potential for any conductivity, and with
   the polarization current in the momentum constraint the pair holds for any motion and any time dependence of the
   conductivity.

Supports Paper II section 3.3 (part 1: the bend realizes the spatial metric about a spherical body; part 3: the
longitudinal momentum constraint), section 7.3 (part 2: the in-plane dilatation through a gradient stiffness, open
item 1) and Appendix A.4 (part 3: Psi = Phi on each slice, the polarization current -dP/dt).

Inputs: none. Output: field/bending_checks-results.txt. Needs sympy.
Run from the repository root (a few seconds; the script writes its output itself):
    python field/bending_checks.py
"""
import pathlib

import sympy as sp
from sympy.calculus.euler import euler_equations

OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


x, y, z = sp.symbols("x y z", real=True)
X = [x, y, z]
rr = sp.symbols("r", positive=True)


# 1. the scalar curvature of A(r) dr^2 + r^2 dOmega^2, on a flat background and on one stretched by l
th, eps_ = sp.symbols("theta epsilon", positive=True)
lfun = sp.Function("l")(rr)
ufun = sp.Function("u")(rr)


def ricci_scalar(A):
    x = [rr, th, sp.Symbol("ph")]
    g = sp.diag(A, rr ** 2, rr ** 2 * sp.sin(th) ** 2)
    gi = g.inv()
    n = 3
    Gam = [[[sum(gi[a, d] * (sp.diff(g[d, b], x[c]) + sp.diff(g[d, c], x[b]) - sp.diff(g[b, c], x[d]))
                 for d in range(n)) / 2 for c in range(n)] for b in range(n)] for a in range(n)]
    Ric = sp.zeros(n)
    for b in range(n):
        for c in range(n):
            Ric[b, c] = sum(sp.diff(Gam[a][b][c], x[a]) - sp.diff(Gam[a][b][a], x[c])
                            + sum(Gam[a][a][d] * Gam[d][b][c] - Gam[a][c][d] * Gam[d][b][a] for d in range(n))
                            for a in range(n))
    return sp.simplify(sum(gi[b, c] * Ric[b, c] for b in range(n) for c in range(n)))


Rflat = sp.series(ricci_scalar(1 + 2 * eps_ * ufun), eps_, 0, 2).removeO()
say("1. flat background: the bend's curvature minus 4(r u)'/r^2 =",
    sp.simplify(Rflat.coeff(eps_, 1) - 4 * sp.diff(rr * ufun, rr) / rr ** 2))
Reig = sp.series(ricci_scalar(lfun ** 2 * (1 + 2 * eps_ * ufun)), eps_, 0, 2).removeO()
say("   background stretched by l: its own curvature =", sp.simplify(Reig.coeff(eps_, 0)),
    "; at a constant stretch of 2 it is", sp.simplify(Reig.coeff(eps_, 0).subs(lfun, 2).doit()))
say("   background stretched by l: the bend's curvature minus 4(r u/l^2)'/r^2 =",
    sp.simplify(Reig.coeff(eps_, 1) - 4 * sp.diff(rr * ufun / lfun ** 2, rr) / rr ** 2),
    "; so the literal analog weights the flux by 1/l^2, not 1/l")

# 2. the gradient-stiffness channel: which dilatation the static equation of the in-plane motion acts on
psi = sp.Function("psi")(*X)
chi = sp.Function("chi")(*X)
bsh = sp.Function("b")(*X)  # the bend's share of the volumetric strain, |grad w|^2/2
lap3 = lambda f: sum(sp.diff(f, v, 2) for v in X)
grad3 = lambda f: [sp.diff(f, v) for v in X]
div3 = lambda V: sum(sp.diff(V[i], X[i]) for i in range(3))
for label, e in (("total strain (strain-gradient form)", lap3(psi) + bsh), ("in-plane displacement alone", lap3(psi))):
    Lg = sp.Rational(1, 2) * chi * sum(g ** 2 for g in grad3(e))
    EL = euler_equations(Lg, [psi], X)[0].lhs
    ok = sp.simplify(sp.expand(EL + lap3(div3([chi * g for g in grad3(e)])))) == 0
    say(f"2. gradient energy of the {label}: static equation is lap[div(chi grad e)] = 0 for that e: {ok}")
say("    in spherical symmetry, bounded at infinity, that is r^2 chi e' = constant, a flux law for that e")

# 3. the reweighted constraints at linear order about flat space, Newtonian gauge, c = 1.
#     Lapse potential Phi from Paper I's Eq. (6), d_i(C^ij d_j Phi) = 4 pi G rho_hat, with any symmetric C^ij(t, x).
#     Paper II's Eq. (6): 4 pi G rho_ph = d_i[(delta^ij - C^ij) d_j Phi], i.e. rho_ph = div P with P = (delta - C) grad Phi/(4 pi G).
#     Reweighted Hamiltonian constraint: lap Psi = 4 pi G (rho_hat + rho_ph).
#     Momentum constraint, longitudinal part: d_i dPsi/dt = -4 pi G (J + J_ph)_i, with the polarization current J_ph = -dP/dt.
say("3. the reweighted constraints in any geometry and for moving sources, at linear order")
t12 = sp.symbols("t", real=True)
G12 = sp.symbols("G", positive=True)
Phi12 = sp.Function("Phi")(t12, *X)
Cm = [[None] * 3 for _ in range(3)]
for i in range(3):
    for j in range(i, 3):
        f = sp.Function(f"C{i}{j}")(t12, *X)
        Cm[i][j] = f
        Cm[j][i] = f
dPhi = [sp.diff(Phi12, v) for v in X]
flux = [sum(Cm[i][j] * dPhi[j] for j in range(3)) for i in range(3)]          # C grad Phi
rho_hat = sum(sp.diff(flux[i], X[i]) for i in range(3)) / (4 * sp.pi * G12)  # Paper I's Eq. (6) read as the baryons' density
Pvec = [(dPhi[i] - flux[i]) / (4 * sp.pi * G12) for i in range(3)]          # the conductivity deficit's flux
rho_ph = sum(sp.diff(Pvec[i], X[i]) for i in range(3))
lapPhi = sum(sp.diff(Phi12, v, 2) for v in X)
say("    lap Phi - 4 pi G (rho_hat + rho_ph) =", sp.simplify(lapPhi - 4 * sp.pi * G12 * (rho_hat + rho_ph)),
    "for any C^ij(t, x): the reweighted Hamiltonian constraint is solved by Psi = Phi, so lensing equals dynamics in any geometry")
J_b = [-sp.diff(flux[i], t12) / (4 * sp.pi * G12) for i in range(3)]       # the baryons' current that continuity and Paper I's Eq. (6) require
cont_b = sp.simplify(sp.diff(rho_hat, t12) + sum(sp.diff(J_b[i], X[i]) for i in range(3)))
J_ph = [-sp.diff(Pvec[i], t12) for i in range(3)]
cont_ph = sp.simplify(sp.diff(rho_ph, t12) + sum(sp.diff(J_ph[i], X[i]) for i in range(3)))
mom = [sp.simplify(sp.diff(Phi12, t12, X[i]) + 4 * sp.pi * G12 * (J_b[i] + J_ph[i])) for i in range(3)]
say("    continuity of the baryons:", cont_b, "; of the phantom with J_ph = -dP/dt:", cont_ph)
say("    momentum constraint d_i dPsi/dt + 4 pi G (J + J_ph)_i with Psi = Phi:", mom,
    "for any motion and any time dependence of C^ij")
mom_noph = sp.simplify(sp.diff(Phi12, t12, X[0]) + 4 * sp.pi * G12 * J_b[0])
say("    without the phantom current the x-component leaves", mom_noph, "(not zero whenever P changes)")

out = pathlib.Path(__file__).with_name("bending_checks-results.txt")
out.write_text("\n".join(OUT) + "\n", encoding="utf-8")
print("wrote field/bending_checks-results.txt")
