r"""The MOND-type algebraic law that mond_classes.py and mond_gc.py set against the data.

The law is MOND's simple interpolation function at a0 = 1.2e-10 m/s^2. In its nu form,
    g = nu(g_N/a0) g_N,   nu(y) = 1/2 + sqrt(1/4 + 1/y),
which is mu(x) = x/(1 + x) with x = g/a0. Its inverse gives the Newtonian field whose boosted value is g,
g_N = g^2/(a0 + g).

An external field enters through the angle-averaged radial field of the quasi-linear formulation (QUMOND)
for a spherical system: a radial internal Newtonian field g_N in a uniform external Newtonian field e_N gives
    g = (1/2) int_{-1}^{1} nu(|g_N r_hat + e_N z_hat|/a0) (g_N - e_N xi) d xi,   xi = cos(angle),
evaluated with 96-point Gauss-Legendre quadrature. A field of total value e enters as its Newtonian equivalent
e_N = e^2/(a0 + e).
"""
import numpy as np

# constants, SI
G = 6.67430e-11
MSUN = 1.98847e30
PC = 3.0856775814913673e16
MPC = 3.0856775814913673e22
A0 = 1.2e-10                        # MOND's a0, m/s^2

_XI, _WXI = np.polynomial.legendre.leggauss(96)


def nu_simple(y):
    """The simple interpolation function in its nu form, y = g_N/a0."""
    y = np.maximum(np.asarray(y, float), 1e-300)
    return 0.5 + np.sqrt(0.25 + 1.0 / y)


def mond_g(gN, a0=A0):
    """The algebraic relation g = nu(g_N/a0) g_N, exact in spherical symmetry with no external field."""
    gN = np.asarray(gN, float)
    return nu_simple(np.maximum(gN, 1e-40) / a0) * gN


def mond_newtonian_of_total(g, a0=A0):
    """The Newtonian field whose boosted value is g: g_N = mu(g/a0) g = g^2/(a0 + g)."""
    g = np.asarray(g, float)
    return g * g / (a0 + g)


def mond_monopole(gN, eN, a0=A0):
    """The angle-averaged radial field of a radial Newtonian field gN in a uniform Newtonian external field eN."""
    gN = np.atleast_1d(np.asarray(gN, float))
    if eN <= 0:
        return nu_simple(np.maximum(gN, 1e-40) / a0) * gN
    out = np.zeros_like(gN)
    for x, w in zip(_XI, _WXI):
        mag = np.sqrt(gN * gN + eN * eN - 2.0 * gN * eN * x)
        out += w * nu_simple(np.maximum(mag, 1e-40) / a0) * (gN - eN * x)
    return 0.5 * out
