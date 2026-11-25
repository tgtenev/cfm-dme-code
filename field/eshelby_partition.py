"""The partition of the Eshelby energy of a roughening event, at the shear continuum's signed bulk modulus.

A roughening event is priced as a spherical unit of volume v with a volumetric misfit delta, whose exterior responds
by pure shear. The shear continuum's bulk modulus is K0 = -(4/3) mu0, the vanishing P-wave modulus. The self-noise
closure uses only the energy that each event delivers to the exterior, E_ext, through the bare noise sum
Z1 = 2 E_ext / (|K0| delta^2 v).

Checks:
 0. The exterior shear energy of an expressed dilation e is E_ext = (2/3) mu0 e^2 v, so in general Z1 = (e/delta)^2.
 1. An interior of positive stiffness +|K0| that carries the misfit expresses only e = delta/2, and Z1 = 1/4.
 2. With the continuum's own signed modulus as the interior stiffness, the energy is linear in e: there is no
    equilibrium, and the continuum offers no static restoring force against the misfit.
 3. With the free volume held by the unit's own stiffness (k > 0, natural dilation delta) and the continuum's interior
    at K0 = -(4/3) mu0 with no misfit of its own, E_int + E_ext = 0 for a compatible dilation, so the continuum is
    passive. The unit expresses e = delta for every k, and Z1 = 1, its largest value.

Supports Paper II section 5.1 and Eq. (11): E_int + E_ext = (1/6)(3 K0 + 4 mu0) eps_u^2 v_u = 0 and
Z1 = (eps_u/delta_fv)^2, with Z1 = 1 at full pinning.

Inputs: none. Output: field/eshelby_partition-results.txt. Needs sympy.
Run from the repository root (a few seconds):
    python field/eshelby_partition.py > field/eshelby_partition-results.txt
"""
import sympy as sp

mu, delta, e, v, a, r, k = sp.symbols('mu delta e v a r k', positive=True)
K = sp.Symbol('K', real=True)

# exterior field u = A r^-2 with A = e a^3 / 3: energy density mu*eps:eps = 6 mu A^2 / r^6
A = e * a**3 / 3
E_ext = sp.integrate(6 * mu * A**2 / r**6 * 4 * sp.pi * r**2, (r, a, sp.oo))
E_ext = sp.simplify(E_ext.subs(a, (3 * v / (4 * sp.pi))**sp.Rational(1, 3)))
print('0. exterior shear energy E_ext =', E_ext, ' (Paper II Eq. (11): (2/3) mu0 eps_u^2 v_u)')
K0abs = sp.Rational(4, 3) * mu                       # |K0|, which prices the demand of the self-noise closure
Z1_of = lambda Eext: sp.simplify(2 * Eext / (K0abs * delta**2 * v))
print('   general: Z1 =', Z1_of(E_ext), ' i.e. (e/delta)^2')

# 1. an interior of stiffness +|K0| with eigenstrain delta
u_pos = sp.Rational(1, 2) * K0abs * (e - delta)**2 * v + E_ext
e_pos = sp.solve(sp.diff(u_pos, e), e)[0]
print('1. interior of stiffness +|K0| carrying the misfit: expressed dilation e =', e_pos, '; Z1 =',
      Z1_of(E_ext.subs(e, e_pos)))

# 2. the continuum's own signed modulus as the interior, carrying the misfit
K0 = -sp.Rational(4, 3) * mu
u_signed = sp.expand(sp.Rational(1, 2) * K0 * (e - delta)**2 * v + E_ext)
print('2. signed interior: u(e) =', u_signed, '; coefficient of e^2 =', u_signed.coeff(e, 2),
      '; du/de =', sp.diff(u_signed, e), '(no stationary point)')
print('   the general equilibrium e = 3 K delta/(3 K + 4 mu) has denominator', sp.simplify((3 * K + 4 * mu).subs(K, K0)),
      'at K = K0')

# 3. free volume held by the unit's own stiffness; the continuum's interior with no misfit of its own
u_pin = sp.expand(sp.Rational(1, 2) * K0 * e**2 * v + sp.Rational(1, 2) * k * (e - delta)**2 * v + E_ext)
print('3. continuum energy of a compatible dilation e, E_int + E_ext:',
      sp.simplify(sp.Rational(1, 2) * K0 * e**2 * v + E_ext), '(passive at a zero P-wave modulus)')
e_pin = sp.solve(sp.diff(u_pin, e), e)[0]
print('   with the unit holding its free volume: e =', e_pin, 'for every k > 0; total stored energy',
      sp.simplify(u_pin.subs(e, e_pin)), '; delivered E_ext =', sp.simplify(E_ext.subs(e, e_pin)),
      '; Z1 =', Z1_of(E_ext.subs(e, e_pin)))
