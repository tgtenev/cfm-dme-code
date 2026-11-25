"""The ball average over one catchment length on the axisymmetric grid of aqual-disk-solver.py, for the full AQUAL
solution coarse-grained at the catchment length (coarse_grained_solve.py). A library: it prints nothing.

Paper I, Sect. 3.7: under scale-selective coupling the stretch acts on the field averaged over a volume one catchment
length across, and finer structure couples to nothing. coarse_grained_solve.py computes this as form B,
    g = g_N[rho] + { AQUAL[rho_ell] - g_N[rho_ell] },     rho_ell = the ball average of rho over radius ell,
so that the non-Newtonian part of the field comes from the smoothed source and the remainder g_N[rho] - g_N[rho_ell]
stays Newtonian. This module supplies the ball average (BallAverage) and the mass-conserving smoothing of a density
(smooth_density).

The ball average of an axisymmetric field at (R, z) is done slice by slice:
    <f>(R, z) = sum_q W_q  D_{a_q}[ f(., z + u_q) ](R),   u_q Gauss-Legendre on [-ell, ell], a_q = sqrt(ell^2 - u_q^2),
    W_q = w_q * 3 (ell^2 - u_q^2) / (4 ell^3),
with D_a the in-plane average over the disk of radius a centred on (R, 0), integrated in polar coordinates about the
centre (Gauss-Legendre in rho, uniform in alpha) against linear interpolation of the field in R. For the radial
component of a vector the kernel carries cos(phi'), the projection of R-hat(phi') on R-hat(0). Reflection symmetry
about z = 0 is used (radial components and scalars even, the vertical component odd). Form B uses the scalar average.
"""
import math
import numpy as np


class BallAverage:
    """Linear operator: ball average of radius ell of an (NR, NZ) array given at nodes (Rn, zn).
    r_odd: the field vanishes at R = 0 (radial component of a vector; a node R = 0 with value 0 is added),
           otherwise it is even in R (held constant below the first node).
    z_odd: the field is odd in z (vertical component; a node z = 0 with value 0 is added), otherwise even.
    comp:  'vecR' projects on R-hat (kernel factor cos phi'), 'scalar' takes the plain average.
    zcut:  rows with zn > zcut are returned unchanged (default: no cut)."""

    def __init__(self, Rn, zn, ell, r_odd, z_odd, comp, Q=None, nrho=24, nalpha=48, zcut=None):
        self.Rn, self.zn = np.asarray(Rn, float), np.asarray(zn, float)
        self.ell, self.r_odd, self.z_odd, self.comp = float(ell), r_odd, z_odd, comp
        NR, NZ = len(self.Rn), len(self.zn)
        self.NR, self.NZ = NR, NZ
        if Q is None:
            dz0 = float(np.min(np.diff(self.zn)))
            Q = int(min(64, max(12, 2 * math.ceil(1.5 * ell / dz0 - 1e-9))))
        Q += Q % 2
        self.Q = Q
        x, w = np.polynomial.legendre.leggauss(Q)
        u = ell * x
        W = w * ell * 3.0 * (ell * ell - u * u) / (4.0 * ell ** 3)
        # use the symmetric pairs: nodes with u > 0 (and their mirrors)
        pos = u > 0
        self.u = u[pos]
        self.W = W[pos]                       # weight of each of the pair members (equal by symmetry)
        if zcut == "auto":
            # rows above the height where the node spacing first exceeds ell are left unchanged: there the ball
            # average differs from the field by about ell^2/(5 z^2) relative, below 1e-3 for these grids
            sp = np.diff(self.zn)
            big = np.where(sp > ell)[0]
            zcut = self.zn[big[0]] if len(big) else None
        self.zcut = zcut
        self.J = NZ if zcut is None else int(np.searchsorted(self.zn, zcut, side="right"))
        self.J = max(self.J, 1)
        # ---- in-plane disk operators, one per pair
        rho, wr = np.polynomial.legendre.leggauss(nrho)
        al = (np.arange(nalpha) + 0.5) * (2 * math.pi / nalpha)
        if r_odd:
            Rext = np.concatenate([[0.0], self.Rn])
        else:
            Rext = self.Rn
        self.D = []
        for uq in self.u:
            a = math.sqrt(max(ell * ell - uq * uq, 0.0))
            rr = 0.5 * a * (rho + 1.0)
            wrr = 0.5 * a * wr * rr                      # rho d rho
            X = self.Rn[:, None, None] + rr[None, :, None] * np.cos(al)[None, None, :]
            Y = rr[None, :, None] * np.sin(al)[None, None, :] + 0.0 * X
            Rp = np.hypot(X, Y)
            f = np.where(Rp > 0, X / np.where(Rp > 0, Rp, 1.0), 1.0) if comp == "vecR" else np.ones_like(Rp)
            wt = (wrr[None, :, None] * (2 * math.pi / nalpha) / (math.pi * a * a)) * f
            wt = np.broadcast_to(wt, Rp.shape)
            k = np.clip(np.searchsorted(Rext, Rp, side="right") - 1, 0, len(Rext) - 2)
            t = np.clip((Rp - Rext[k]) / (Rext[k + 1] - Rext[k]), 0.0, 1.0)
            rows = np.broadcast_to(np.arange(NR)[:, None, None], Rp.shape)
            base = (rows * len(Rext) + k).ravel()
            M = np.bincount(base, (wt * (1 - t)).ravel(), minlength=NR * len(Rext))
            M = M + np.bincount(base + 1, (wt * t).ravel(), minlength=NR * len(Rext))
            M = M[:NR * len(Rext)].reshape(NR, len(Rext))
            if r_odd:
                M = M[:, 1:]                       # the R = 0 node carries the value 0
            self.D.append(np.ascontiguousarray(M))
        # ---- vertical shifts with reflection, per pair (+u and -u)
        zext = np.concatenate([[0.0], self.zn]) if z_odd else self.zn
        self.zext = zext
        zo = self.zn[:self.J]
        self.sh = []
        for uq in self.u:
            pair = []
            for sgn in (1.0, -1.0):
                zq = zo + sgn * uq
                s = np.where(zq < 0, -1.0, 1.0) if z_odd else np.ones_like(zq)
                za = np.abs(zq)
                k = np.clip(np.searchsorted(zext, za, side="right") - 1, 0, len(zext) - 2)
                t = np.clip((za - zext[k]) / (zext[k + 1] - zext[k]), 0.0, 1.0)
                pair.append((k, t, s))
            self.sh.append(pair)

    def __call__(self, F):
        F = np.asarray(F, float)
        Fe = np.concatenate([np.zeros((F.shape[0], 1)), F], axis=1) if self.z_odd else F
        out = F.copy()
        acc = np.zeros((self.NR, self.J))
        for (pp, pm), Dq, Wq in zip(self.sh, self.D, self.W):
            (k1, t1, s1), (k2, t2, s2) = pp, pm
            Yq = s1 * (Fe[:, k1] * (1 - t1) + Fe[:, k1 + 1] * t1) + s2 * (Fe[:, k2] * (1 - t2) + Fe[:, k2 + 1] * t2)
            acc += Wq * (Dq @ Yq)
        out[:, :self.J] = acc
        return out


def smooth_density(grid, rho, ell, zcut=None, Q=None, op=None):
    """Ball-smoothed density at the cell centres, renormalized to the total mass of rho (the gather operator with
    linear interpolation conserves mass only to quadrature accuracy; the correction is reported)."""
    if ell <= 0:
        return rho.copy(), 0.0
    if op is None:
        op = BallAverage(grid.Rc, grid.zc, ell, r_odd=False, z_odd=False, comp="scalar", zcut=zcut, Q=Q)
    rs = op(rho)
    m0 = (rho * grid.V).sum(); m1 = (rs * grid.V).sum()
    if m1 > 0:
        rs = rs * (m0 / m1)
    return rs, (m1 / m0 - 1.0) if m0 > 0 else 0.0
