"""The boost that a trailing arrangement of stored axes leaves at a galaxy's data radii (Paper II, section 3.1).

The model is a sphere. Its baryons are laid in shells so that the Newtonian field at each data radius equals the
galaxy's tabulated baryonic field there, M(<r) = g_bar r^2 / G, rising as r^3 inside the innermost radius and constant
beyond the last. The static aligned arrangement of the law is laid about it: axes radial, stretch nu(X) with
X mu(X) = g_N / a0, capped at CAP. The arrangement is then displaced behind the body, along the direction of motion,
by the trail that a rule for the direction of a new layer gives. The field is solved with that tensor frozen, by the
solver of kinetics/axfem.py (Galerkin elements on a polar grid of the meridional half-plane, symmetric positive
definite for any tilt).

Two displacements are offered.
  sheared: each shell of the arrangement at radius s is laid about a centre displaced by its own trail,
           D(s) = F 2 v T(s) nu(s)^p, T = (1/Gamma) (nu - 1 + delta)/delta, with p = 1 under the gradient rule
           (a new layer laid across the local interior gradient) and p = 0 under the flux rule (across the flux);
           inside the innermost data radius the trail is held at its value there, since the core's own state is
           not known (the moving-body solution of kinetics/moving_body.py did not settle in its core either).
  rigid:   the whole arrangement is displaced by the trail at the innermost data radius.
The displacement scales with one number, K = v F/Gamma in km/s yr, with 1/Gamma the mean interval between renewal
events at a domain, so each galaxy is solved on a grid of K and the result is interpolated in log K.

The boost left at a data radius r is the field toward the body, averaged as a rotation curve averages it, over the
field of the undisplaced arrangement on the same grid:
  in the plane: the mean of g_r over the circle of radius r in a plane that contains the direction of motion;
  normal:       g_r on the circle of radius r in the plane normal to the direction of motion (theta = 90 deg).
A disk that moves in its own plane is read by the first, one that moves along its axis by the second. The sphere has
no disk, so the faster turn of the field across a disk's gas and stellar layers is not in the model.

Units: G = a0 = 1 and lengths in kpc. Imported by layer_direction_test.py; reads no data.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "kinetics")))
import axfem as A  # noqa: E402

PC_PER_YR_KMS = 977792.2          # 1 pc/yr in km/s, as in kinetics/stored_axis_kinetics.py
CAP = 200.0                       # cap on the stretch near the centre, where the field vanishes


class GalaxySphere:
    """The spherical model of one galaxy and its solves."""

    def __init__(self, R, y, delta, cap=CAP, out_factor=4.0, dxi=0.02, nt=128):
        R = np.asarray(R, float)
        y = np.asarray(y, float)
        M = np.maximum.accumulate(np.maximum(y * R * R, 1e-12))
        self.R, self.M, self.delta = R, M, delta
        self.law = A.Law(delta, cap=cap)
        self.r_out = out_factor * R[-1]
        self.mesh = A.Mesh(A.r_nodes(dxi=dxi, r_a=0.5 * R[0], r_out=self.r_out), nt)
        # fine radial table of the static arrangement
        rt = np.concatenate([[0.0], np.geomspace(R[0] * 1e-4, self.r_out * 3.0, 40001)])
        Mt = self.mass(rt)
        yt = np.where(rt > 0, Mt / np.maximum(rt, 1e-300) ** 2, 0.0)
        X, nu, _ = self.law.static(np.maximum(yt, 1e-300))
        nu[0] = nu[1]
        self.rt, self.nut = rt, nu
        dphi = nu * yt
        phi = np.concatenate([[0.0], np.cumsum(0.5 * (dphi[1:] + dphi[:-1]) * np.diff(rt))])
        self.phit = phi - np.interp(self.r_out, rt, phi)
        m = self.mesh
        self.b = m.load(self.rho(m.rq[:, :4]))
        self.ref = None

    # --- the baryons
    def mass(self, r):
        r = np.asarray(r, float)
        R, M = self.R, self.M
        core = M[0] * (np.minimum(r, R[0]) / R[0]) ** 3
        mid = np.interp(r, R, M)
        return np.where(r < R[0], core, np.where(r > R[-1], M[-1], mid))

    def rho(self, r):
        r = np.asarray(r, float)
        R, M = self.R, self.M
        slope = np.diff(M) / np.diff(R)
        k = np.clip(np.searchsorted(R, r) - 1, 0, len(slope) - 1)
        dM = np.where(r < R[0], 3.0 * M[0] * r * r / R[0] ** 3, np.where(r > R[-1], 0.0, slope[k]))
        return dM / (4.0 * np.pi * np.maximum(r, 1e-300) ** 2)

    def nu_static(self, s):
        return np.interp(s, self.rt, self.nut)

    def phi_static(self, s):
        return np.interp(s, self.rt, self.phit)

    # --- the trail
    def trail(self, s, kd, p):
        """D(s) in kpc for the displacement scale kd = K = v F/Gamma (km/s yr), held inside the innermost radius at its
        value there."""
        nu = self.nu_static(np.maximum(s, self.R[0]))
        T = (nu - 1.0 + self.delta) / self.delta
        return kd * 2.0 * T * nu ** p / PC_PER_YR_KMS / 1000.0

    def trail_ratio_inner(self, kd, p):
        return float(self.trail(self.R[0], kd, p)) / self.R[0]

    # --- the solve
    def solve(self, D_fun):
        m = self.mesh
        Rq, zq = m.rq * np.sin(m.tq), m.rq * np.cos(m.tq)
        D = D_fun(m.rq)
        zc = zq + D                              # the centre of each shell's arrangement lies at -D z_hat
        rp = np.hypot(Rq, zc)
        beta = A.wrap_half(np.arctan2(Rq, zc) - m.tq)
        nu = self.nu_static(rp)
        K = m.stiffness(*m.tensor(beta[:, :4], nu[:, :4]))
        rb, tb = m.node_r[m.bnd], m.node_t[m.bnd]
        Db = D_fun(rb)
        bv = self.phi_static(np.hypot(rb * np.sin(tb), rb * np.cos(tb) + Db))
        phi = m.solve(K, self.b, bv)
        gr, gt = m.grad(phi, 4)
        G = m.ring(gr)
        nt = m.Nt
        merid = G.mean(axis=1)
        equat = 0.5 * (G[:, nt // 2 - 1] + G[:, nt // 2])
        return merid, equat

    def reference(self):
        if self.ref is None:
            self.ref = self.solve(lambda s: np.zeros_like(np.asarray(s, float)))
        return self.ref

    def boost_left(self, kd, p, rigid=False):
        """(in the plane, normal) boost left at the data radii, over the undisplaced solve on the same grid."""
        m0, e0 = self.reference()
        if rigid:
            d_in = float(self.trail(self.R[0], kd, p))
            fun = lambda s: np.full_like(np.asarray(s, float), d_in)
        else:
            fun = lambda s: self.trail(np.asarray(s, float), kd, p)
        m1, e1 = self.solve(fun)
        rc = self.mesh.rc_ring
        bi = np.interp(self.R, rc, m1) / np.interp(self.R, rc, m0)
        bn = np.interp(self.R, rc, e1) / np.interp(self.R, rc, e0)
        return bi, bn

    def static_check(self):
        """The undisplaced solve against the spherical law at the data radii: g_r / (nu y) - 1."""
        m0, e0 = self.reference()
        rc = self.mesh.rc_ring
        y = self.mass(self.R) / self.R ** 2
        g = self.nu_static(self.R) * y
        return np.interp(self.R, rc, m0) / g - 1.0, np.interp(self.R, rc, e0) / g - 1.0
