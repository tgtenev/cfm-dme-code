"""A point mass in a medium with a uniform stored axis: the exact field.

Near a body much smaller than the catchment length the axis and the coefficient are those of the ambient field, so
the body sits in a uniform medium with the coefficient mu along the axis and 1/mu across it, mu = 1/stretch. The
field equation, div(A grad Phi) = 4 pi G rho with A = mu n n + (1 - n n)/mu (Paper II Eq. (3)), then has the closed
solution
    Phi = - G M / sqrt(R^2 + z^2 / mu^2),      z along the axis, R across it,
which this script checks and reads:
  1. the flux of A grad Phi through spheres about the mass is 4 pi G M (Gauss), by quadrature;
  2. the field across the axis is Newton's, and along the axis it is mu times Newton's;
  3. the field lies atan(mu^2 tan(chi)) from the axis at a place chi from it, so it follows the axis.
Supports Paper II Appendix A.3: where the flux crosses the stored axis at chi = 45 degrees at a stretch of 10
(mu = 0.1), the potential gradient lies 0.6 degrees from the axis (0.573 in part 3).

Inputs: none. Output: kinetics/uniform_axis_point_mass-results.txt.
Run from the repository root (under a second):
    python kinetics/uniform_axis_point_mass.py > kinetics/uniform_axis_point_mass-results.txt
"""
import math

G_M = 1.0


def phi(R, z, mu):
    return -G_M / math.sqrt(R * R + z * z / (mu * mu))


def grad(R, z, mu):
    d3 = (R * R + z * z / (mu * mu)) ** 1.5
    return G_M * R / d3, G_M * z / (mu * mu) / d3


def flux_through_sphere(r, mu, n=20000):
    """Integral of (A grad Phi) . r_hat over the sphere of radius r, by the midpoint rule in cos(chi)."""
    tot = 0.0
    for i in range(n):
        u = -1.0 + (i + 0.5) * 2.0 / n
        s = math.sqrt(1.0 - u * u)
        gR, gz = grad(r * s, r * u, mu)
        tot += (gR / mu * s + mu * gz * u) * 2.0 / n
    return 2.0 * math.pi * r * r * tot


print("A POINT MASS IN A MEDIUM WITH A UNIFORM AXIS: Phi = - G M / sqrt(R^2 + z^2/mu^2)")
print("=" * 100)
print("1. Gauss: the flux of A grad Phi through spheres, over 4 pi G M")
for mu in (0.7, 0.3, 0.1, 0.05):
    print(f"   mu = {mu:4.2f}: " + ", ".join(f"r = {r:g}: {flux_through_sphere(r, mu) / (4.0 * math.pi * G_M):.6f}" for r in (0.5, 1.0, 7.0)))
print()
print("2. The field over Newton's at unit distance: across the axis and along it")
for mu in (0.7, 0.3, 0.1, 0.05):
    gR, _ = grad(1.0, 0.0, mu)
    _, gz = grad(0.0, 1.0, mu)
    print(f"   mu = {mu:4.2f}: across {gR / G_M:.4f}, along {gz / G_M:.4f}; the potential along the axis is {-phi(0.0, 1.0, mu) / G_M:.4f} of Newton's")
print()
print("3. The angle of the field from the axis, at a place chi from the axis (degrees)")
print("   mu    | chi = 10      30      45      60      80      89")
for mu in (0.7, 0.3, 0.1, 0.05):
    cells = []
    for chi in (10.0, 30.0, 45.0, 60.0, 80.0, 89.0):
        c = math.radians(chi)
        gR, gz = grad(math.sin(c), math.cos(c), mu)
        cells.append(f"{math.degrees(math.atan2(gR, gz)):7.3f}")
    print(f"   {mu:4.2f}  |   " + " ".join(cells))
print("   (the closed form is atan(mu^2 tan chi): a share mu^2 of a small offset)")
