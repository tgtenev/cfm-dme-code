"""The dipole (l = 1) part of the problem at first order, on a fine radial grid.

Around the static aligned sphere (axes radial, stretch nu0(r), field dPhi0/dr = nu0 gN, flux gN) let the axes lie
radially about the point d_n(r) z_hat, let the stretch be nu0 + nu1(r) cos(theta), and let the potential be
Phi0 + phi1(r) cos(theta). With the mass at rest, the l = 1 part of div(C grad Phi) = 4 pi G rho is

    (r^2 mu0 phi1')' - 2 nu0 phi1 = (r^2 mu0 nu1 gN)' - 2 (1 - nu0^2) gN d_n .                          (1)

The field's direction centre is s_g = -phi1 / Phi0'. Relative to the old axis the field lies at the angle
(s_g - d_n) sin(theta)/r, the vector of rule B at nu0 times that, and the flux at nu0^2 times that, so the centres of
the axes that one round lays are s_q = d_n + nu0^q (s_g - d_n), q = 0, 1, 2 for rules A, B, C.

Linear response (rigid displacement): d_n = d0 and nu1 = -nu0' d0 (axes and stretches displaced), or nu1 = 0 (axes
only). The share of the offset removed by one round is f_q = (d0 - s_q)/d0 = nu0^q f_A.

Steady motion at speed v (fabric flowing past at -v z_hat), at first order in v: the stored axis at x is the rule's
axis at x + 2 v T z_hat (the stretch-weighted mean age is 2T) and the stretch is the law at the drive at x + v T z_hat
(its excess relaxes in one interval T). So d_n = s_q - 2 v T, s_g - d_n = w = 2 v T / nu0^q, and
nu1 = L' phi1' + v T nu0', L' = d nu / dX at the static drive. Eliminating d_n and nu1 turns (1) into

    (r^2 mu0 (1 - gN L') phi1')' - 2 mu0 phi1 = (r^2 mu0 v T nu0' gN)' - 2 (nu0^2 - 1) gN w ,            (2)

whose left side is the dipole operator of AQUAL linearized about the static solution. The trail of the axes is
-d_n = -s_g + w, and that of the field is -s_g.

Boundary conditions: phi1(0) = 0 (regular centre) and a Dirichlet value at r_out: phi1 = 0 pins the field at r_out
to the body (the medium beyond holds the static arrangement about the body); phi1 = -d0 Phi0' continues the rigidly
displaced arrangement beyond r_out. Conservative finite differences on a geometric grid; tridiagonal solve.

A module, with no output of its own. Imported by kinetics/steady_dipole.py, kinetics/linear_response.py and
kinetics/moving_body.py. Needs numpy and scipy.
"""
import numpy as np
from scipy.linalg import solve_banded


class L1:
    def __init__(self, body, r_out, n=20000, r_min=1e-5, cap_r=None):
        """cap_r: if given, the cap on the stretch applies only inside r < cap_r (to extend the arrangement
        beyond the radius where the stretch reaches the cap)."""
        law = body.law
        self.body, self.law, self.r_out = body, law, r_out
        r = np.concatenate([[0.0], np.geomspace(r_min, r_out, n)])
        self.r = r
        rh = 0.5 * (r[1:] + r[:-1])
        self.rh = rh
        self.h = np.diff(r)

        def stat(x):
            y = np.maximum(body.gN(x), 1e-300)
            X, nu, cap = law.static(y)
            if cap_r is not None:
                Xu = law.X_of_y(y)
                out = x >= cap_r
                X, nu, cap = np.where(out, Xu, X), np.where(out, law.nu(Xu), nu), cap & ~out
            return y, X, nu, cap

        self.y, self.X, self.nu, self.cap = stat(np.maximum(r, 1e-12))
        self.yh, self.Xh, self.nuh, self.caph = stat(rh)
        self.dphi = self.nu * self.y                              # Phi0' at the nodes
        self.dnuh = np.diff(self.nu) / self.h                     # nu0' at the half points
        self.dnu = np.interp(r, rh, self.dnuh)
        self.Lh = np.where(self.caph, 0.0, law.dnu(self.Xh))      # d nu / dX at the half points
        self.T = law.kin(self.X)[0]                               # interval between smoothings over T0, nodes
        self.Th = law.kin(self.Xh)[0]                             # (held at its value at the cap of the stretch)
        if cap_r is not None:                                     # uncapped outside cap_r, like the stretch
            self.T = np.where(r >= cap_r, 1.0 / law.phi(self.X), self.T)
            self.Th = np.where(rh >= cap_r, 1.0 / law.phi(self.Xh), self.Th)

    def _solve(self, P, Q, J, S, phi_out):
        """Solve (P phi')' - Q phi = J' + S on the nodes; P, J at half points; Q, S at nodes; phi(0) = 0."""
        r, h = self.r, self.h
        n = len(r)
        m = n - 2                                                 # unknowns 1..n-2
        hm, hp = h[:-1], h[1:]
        hav = 0.5 * (hm + hp)
        lo = P[:-1] / hm / hav
        up = P[1:] / hp / hav
        di = -(lo + up) - Q[1:-1]
        rhs = (J[1:] - J[:-1]) / hav + S[1:-1]
        rhs[-1] -= up[-1] * phi_out
        ab = np.zeros((3, m))
        ab[0, 1:] = up[:-1]
        ab[1, :] = di
        ab[2, :-1] = lo[1:]
        phi = np.zeros(n)
        phi[1:-1] = solve_banded((1, 1), ab, rhs)
        phi[-1] = phi_out
        return phi

    def rigid(self, d0=1.0, stretch=True, outer="body"):
        """Linear response to a rigid displacement d0 of the axes (and stretches). Returns s_g(r) at the nodes."""
        P = self.rh ** 2 / self.nuh
        Q = 2.0 * self.nu
        nu1h = -self.dnuh * d0 if stretch else np.zeros_like(self.rh)
        J = self.rh ** 2 * nu1h * self.yh / self.nuh
        S = -2.0 * (1.0 - self.nu ** 2) * self.y * d0
        phi_out = 0.0 if outer == "body" else -d0 * self.dphi[-1]
        phi = self._solve(P, Q, J, S, phi_out)
        return -phi / np.maximum(self.dphi, 1e-300)

    def steady(self, q, vT0, outer_phi=0.0):
        """First-order steady state at v T0 (scaled), rule q = 0, 1, 2. Returns (trail of the axes, trail of the
        field, the predicted trail 2 v T nu0^(2-q)) at the nodes; trail = distance behind the body."""
        P = self.rh ** 2 / self.nuh * (1.0 - self.yh * self.Lh)
        Q = 2.0 / self.nu
        J = self.rh ** 2 / self.nuh * vT0 * self.Th * self.dnuh * self.yh
        w = 2.0 * vT0 * self.T / self.nu ** q
        S = -2.0 * (self.nu ** 2 - 1.0) * self.y * w
        phi = self._solve(P, Q, J, S, outer_phi)
        s_g = -phi / np.maximum(self.dphi, 1e-300)
        d_n = s_g - w
        pred = 2.0 * vT0 * self.T * self.nu ** (2 - q)
        return -d_n, -s_g, pred, phi

    def at(self, arr, r):
        return np.interp(r, self.r, arr)
