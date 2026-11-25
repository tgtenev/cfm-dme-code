"""The equality of the two potentials: the reweighted Hamiltonian constraint, derived by counting paths of the flux.

1. The identity. With Paper II's phantom density in its source, the Hamiltonian constraint is solved by Psi = Phi for any
   conductivity C^ij(x). Checked symbolically for a general symmetric C^ij in three dimensions.
2. The shortfall without it. If the bend follows the baryons alone, an observer who assumes General Relativity infers a
   lensing mass (1 + 1/nu)/2 of the dynamical one, evaluated on Paper I's law at a free volume of 0.19.
3. The linearized Einstein tensor of the static metric -(1 + 2 Phi)dt^2 + (1 - 2 Psi)dx^2 (c = 1), computed from the
   general linearized formula: G_00 = 2 lap Psi and G_ij = (delta_ij lap - d_i d_j)(Phi - Psi).
4. The spherical derivation. For any enclosed baryonic mass M(r) and any stretch l(r), the lapse law
   (r^2 Phi'/l)' = 4 pi G rho r^2 and the bend's law with its flux counted the same way, (r u/l)' = 4 pi G rho r^2/c^2,
   give u = r Phi'/c^2, the condition for Psi = Phi. Left as it is, the bend's law gives u/(r Phi'/c^2) = 1/l. Counting paths on
   the bend's slope instead, a flat law for the slope per length of material path, gives u/(r Phi'/c^2) = l.
5. What General Relativity reads in each readout, around a point mass of 1e11 solar masses on Paper I's law: with the
   constraint left as it is, the phantom has no energy density and a stress whose trace is rho_ph c^2, so its radial
   pressure approaches rho_ph c^2 in the deep regime. With the reweighted constraint it is a density without stress.

Supports Paper II sections 3.3 and 7.3 and Appendix A.4 (Psi = Phi for any conductivity; without the reweighting
the lensing mass falls toward 0.515 of the dynamical mass, about half) and Paper I section 5.2 (lensing would
underestimate dynamical masses by 0.20 dex at g_bar = 1e-11 m/s^2 and by 0.29 dex at 1e-13 m/s^2).

Inputs: none. Output: field/two_potentials_checks-results.txt. Needs sympy.
Run from the repository root (a few seconds):
    python field/two_potentials_checks.py > field/two_potentials_checks-results.txt
"""
import math

import sympy as sp

# 1. the identity, for a general symmetric conductivity
x = sp.symbols('x0:3', real=True)
G = sp.symbols('G', positive=True)
Phi = sp.Function('Phi')(*x)
C = [[None] * 3 for _ in range(3)]
for i in range(3):
    for j in range(i, 3):
        C[i][j] = C[j][i] = sp.Function(f'C{i}{j}')(*x)
dPhi = [sp.diff(Phi, v) for v in x]
flux = [sum(C[i][j] * dPhi[j] for j in range(3)) for i in range(3)]
rho_hat = sum(sp.diff(flux[i], x[i]) for i in range(3)) / (4 * sp.pi * G)          # Paper I's Eq. (6), solved for the source
rho_ph = sum(sp.diff(dPhi[i] - flux[i], x[i]) for i in range(3)) / (4 * sp.pi * G)  # Paper II's phantom density
lap = sum(sp.diff(Phi, v, 2) for v in x)
resid = sp.simplify(lap - 4 * sp.pi * G * (rho_hat + rho_ph))
print('1. lap(Phi) - 4 pi G (rho_hat + rho_ph) =', resid,
      '-> Psi = Phi solves the reweighted constraint for any symmetric C^ij(x)')

# 2. the lensing mass without the reweighting, on Paper I's law
DELTA, A0 = 0.19, 0.97e-10
LAM = math.log(1 + DELTA)


def nu_of_X(X):
    return 1 + DELTA / math.expm1(LAM * X)


def X_of_Y(Y):
    """Solve Y = X mu(X) = X / nu(X) for the total field X, which rises monotonically with X."""
    lo, hi = 1e-12, 1e6
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if mid / nu_of_X(mid) < Y:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


print(f'2. lensing mass over dynamical mass if the bend follows the baryons alone (free volume {DELTA}, a0 = {A0:.2e} m/s^2)')
for gbar in (1e-9, 1e-10, 3e-11, 1e-11, 3e-12, 1e-12, 1e-13):
    X = X_of_Y(gbar / A0)
    nu = nu_of_X(X)
    ratio = (1 + 1 / nu) / 2
    print(f'   g_bar = {gbar:.0e} m/s^2: nu = {nu:6.2f}, M_lens/M_dyn = {ratio:.3f} ({math.log10(ratio):+.2f} dex)')
print('   at the crossover, X = 1, nu = 2 exactly, so the ratio is 0.75')

# 3. the linearized Einstein tensor of the static metric, from the general formula
t = sp.symbols('t', real=True)
X4 = (t,) + x
eta = sp.diag(-1, 1, 1, 1)
Ph = sp.Function('Phi')(*x)
Ps = sp.Function('Psi')(*x)
h = sp.zeros(4, 4)
h[0, 0] = -2 * Ph
for i in range(1, 4):
    h[i, i] = -2 * Ps
hup = eta * h * eta                      # h^{mu nu}
hmix = eta * h                           # h^mu_nu (row index raised)
htr = sum(hmix[m, m] for m in range(4))


def d(f, m):
    return sp.diff(f, X4[m])


def box(f):
    return -d(d(f, 0), 0) + sum(d(d(f, k), k) for k in range(1, 4))


R = sp.zeros(4, 4)
for m in range(4):
    for n in range(4):
        R[m, n] = sp.Rational(1, 2) * (sum(d(d(hmix[r, n], m), r) + d(d(hmix[r, m], n), r) for r in range(4))
                                       - box(h[m, n]) - d(d(htr, m), n))
Rs = sum(d(d(hup[r, s], r), s) for r in range(4) for s in range(4)) - box(htr)
Gt = (R - sp.Rational(1, 2) * eta * Rs).applyfunc(sp.simplify)
lap3 = lambda f: sum(sp.diff(f, v, 2) for v in x)  # noqa: E731
ok00 = sp.simplify(Gt[0, 0] - 2 * lap3(Ps)) == 0
okij = all(sp.simplify(Gt[i + 1, j + 1] - ((1 if i == j else 0) * lap3(Ph - Ps) - sp.diff(Ph - Ps, x[i], x[j]))) == 0
           for i in range(3) for j in range(3))
ok0i = all(sp.simplify(Gt[0, k]) == 0 for k in range(1, 4))
print(f'3. linearized Einstein tensor: G_00 = 2 lap Psi: {ok00}; G_ij = (delta_ij lap - d_i d_j)(Phi - Psi): {okij}; '
      f'G_0i = 0: {ok0i}')

# 4. the spherical derivation, for any M(r) and any stretch l(r)
r, c = sp.symbols('r c', positive=True)
M = sp.Function('M')(r)
ell = sp.Function('ell')(r)
dPhi_r = ell * G * M / r**2                         # the lapse law integrated: r^2 Phi'/l = G M(r)
u_flux = ell * G * M / (c**2 * r)                   # the bend's law with its flux counted: r u/l = G M(r)/c^2
u_baryons = G * M / (c**2 * r)                      # the bend's law left as it is: r u = G M(r)/c^2
u_slope = ell**2 * G * M / (c**2 * r)               # a flat law for the slope per length of material path
equal_u = r * dPhi_r / c**2
print('4. spherical, any M(r) and l(r):')
print('   lapse law integrated, d/dr(r^2 Phi\'/l) - 4 pi G rho r^2 with 4 pi rho r^2 = M\':',
      sp.simplify(sp.diff(r**2 * dPhi_r / ell, r) - G * sp.diff(M, r)))
print('   bend counted on its flux: u - r Phi\'/c^2 =', sp.simplify(u_flux - equal_u),
      '; its law, d/dr(r u/l) - 4 pi G rho r^2/c^2 =', sp.simplify(sp.diff(r * u_flux / ell, r) - G * sp.diff(M, r) / c**2))
print('   bend left on the baryons: u/(r Phi\'/c^2) =', sp.simplify(u_baryons / equal_u),
      '; bend counted on its slope: u/(r Phi\'/c^2) =', sp.simplify(u_slope / equal_u))
w1 = sp.sqrt(2 * u_flux)
print('   the slope of the counted bend, w\' = sqrt(2u):', sp.simplify(w1),
      '; the per-path slope w\'/l squared over 2 equals u/l^2:', sp.simplify((w1 / ell)**2 / 2 - u_flux / ell**2) == 0)

# 5. what General Relativity reads in each readout, around a point mass
GSI, CSI, MSUN, KPC = 6.674e-11, 2.998e8, 1.989e30, 3.0857e19
Mb = 1e11 * MSUN
r0 = math.sqrt(GSI * Mb / A0)                       # the radius where the Newtonian field equals a0
print(f'5. a point mass of 1e11 solar masses; r0 = sqrt(GM/a0) = {r0 / KPC:.1f} kpc')


def Dprime(rr):
    """The phantom potential's gradient, Phi' - Phi_N' = (nu - 1) g_N, outside the baryons."""
    gN = GSI * Mb / rr**2
    return (nu_of_X(X_of_Y(gN / A0)) - 1) * gN


for k in (0.5, 1, 3, 10, 30):
    rr = k * r0
    hstep = 1e-4 * rr
    D1 = Dprime(rr)
    D2 = (Dprime(rr + hstep) - Dprime(rr - hstep)) / (2 * hstep)
    rho_ph = (D2 + 2 * D1 / rr) / (4 * math.pi * GSI)          # (1/4 pi G r^2)(r^2 D')'
    p_r = CSI**2 * D1 / (4 * math.pi * GSI * rr)               # T_rr with Phi - Psi = D
    p_t = CSI**2 * (D2 + D1 / rr) / (8 * math.pi * GSI)        # T_theta theta
    e = rho_ph * CSI**2
    print(f'   r = {k:4.1f} r0: rho_ph = {rho_ph:.2e} kg/m^3; bend left on the baryons: energy density 0, '
          f'p_r/(rho_ph c^2) = {p_r / e:.3f}, p_t/(rho_ph c^2) = {p_t / e:.3f}, (p_r + 2 p_t)/(rho_ph c^2) = {(p_r + 2 * p_t) / e:.3f}')
print('   bend counted on its flux: Psi = Phi, so T_ij = 0 and the phantom is a density rho_ph without stress')
