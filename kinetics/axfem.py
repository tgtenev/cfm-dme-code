"""Axisymmetric solver for a uniaxial, strongly anisotropic elliptic equation with stored axes.

    div( C grad Phi ) = 4 pi G rho,     C = mu n n + (1/mu)(1 - n n),     mu = 1/nu,

with the axis n in the meridional plane and no mirror symmetry about z = 0. The grid is polar in the meridional
half-plane, r from 0 (one node at the origin) to r_out and theta from 0 to pi, so it covers both signs of z.

Discretization. Galerkin finite elements, bilinear in (r, theta), 2 x 2 Gauss quadrature, volume element
2 pi r^2 sin(theta) dr dtheta, gradient (d/dr, (1/r) d/dtheta). The element matrix is the quadrature of
grad(N_a) . C . grad(N_b), so the global matrix is symmetric and positive definite for every positive tensor,
whatever the tilt of the axis and the anisotropy ratio nu^2. The tensor is given at the Gauss points in the local
polar basis by (beta, nu), beta being the angle of n from the radial direction toward the polar unit vector.
A static aligned arrangement (n radial) is then exactly aligned with the grid; a tilted axis brings the cross term
C_r,theta. Dirichlet values are imposed at r_out (and at r[0] when r[0] > 0, for an annulus).

Units: G = 1, a0 = 1, the body's radius R = 1. Angles psi in the (R, z) plane are measured from +z toward +R;
the radial direction at polar angle theta has psi = theta.

A module, with no output of its own. Imported by kinetics/kinetics.py, kinetics/steady_dipole.py,
kinetics/linear_response.py, kinetics/moving_body.py and rotation-curves/boost_left.py. Needs numpy and scipy.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_g = 1.0 / np.sqrt(3.0)
QS = np.array([-_g, _g, -_g, _g, 0.0])          # reference s (along r) of the 4 Gauss points and the centre
QT = np.array([-_g, -_g, _g, _g, 0.0])          # reference t (along theta)
AS = np.array([-1.0, 1.0, -1.0, 1.0])           # node a: (i, j), (i+1, j), (i, j+1), (i+1, j+1)
AT = np.array([-1.0, -1.0, 1.0, 1.0])
NQ = (1.0 + AS[None, :] * QS[:, None]) * (1.0 + AT[None, :] * QT[:, None]) / 4.0     # (5, 4)
DNS = AS[None, :] * (1.0 + AT[None, :] * QT[:, None]) / 4.0                           # dN/ds (5, 4)
DNT = AT[None, :] * (1.0 + AS[None, :] * QS[:, None]) / 4.0                           # dN/dt (5, 4)


def wrap_half(a):
    """Wrap an angle difference between two axes (lines) into (-pi/2, pi/2]."""
    return a - np.pi * np.round(a / np.pi)


def r_nodes(dxi=0.02, r_a=0.2, r_out=5.0):
    """Radial nodes: 0, uniform steps r_a*dxi up to r_a, then a geometric progression of ratio exp(dxi) to r_out."""
    h = r_a * dxi
    n_a = max(int(round(r_a / h)), 1)
    inner = np.linspace(0.0, r_a, n_a + 1)
    n_b = max(int(np.ceil(np.log(r_out / r_a) / dxi)), 1)
    outer = r_a * np.exp(np.linspace(0.0, np.log(r_out / r_a), n_b + 1))
    return np.concatenate([inner, outer[1:]])


class Mesh:
    def __init__(self, r, n_theta):
        r = np.asarray(r, float)
        self.r, self.Nr, self.Nt = r, len(r) - 1, int(n_theta)
        self.th = np.linspace(0.0, np.pi, self.Nt + 1)
        self.origin = r[0] == 0.0
        Nr, Nt = self.Nr, self.Nt
        I, J = np.meshgrid(np.arange(Nr + 1), np.arange(Nt + 1), indexing="ij")
        if self.origin:
            ids = np.where(I == 0, 0, 1 + (I - 1) * (Nt + 1) + J)
            self.nn = 1 + Nr * (Nt + 1)
        else:
            ids = I * (Nt + 1) + J
            self.nn = (Nr + 1) * (Nt + 1)
        self.node_id = ids
        # node coordinates
        self.node_r = np.zeros(self.nn)
        self.node_t = np.zeros(self.nn)
        self.node_r[ids.ravel()] = r[I.ravel()]
        self.node_t[ids.ravel()] = self.th[J.ravel()]
        # elements
        ie, je = np.meshgrid(np.arange(Nr), np.arange(Nt), indexing="ij")
        ie, je = ie.ravel(), je.ravel()
        self.ie, self.je, self.Ne = ie, je, Nr * Nt
        self.conn = np.stack([ids[ie, je], ids[ie + 1, je], ids[ie, je + 1], ids[ie + 1, je + 1]], axis=1)
        dr = (r[1:] - r[:-1])[ie]
        dt = (self.th[1:] - self.th[:-1])[je]
        self.dr, self.dt = dr, dt
        self.rq = r[ie][:, None] + (1.0 + QS[None, :]) / 2.0 * dr[:, None]          # (Ne, 5)
        self.tq = self.th[je][:, None] + (1.0 + QT[None, :]) / 2.0 * dt[:, None]
        self.Gr = DNS[None, :, :] * (2.0 / dr)[:, None, None]                       # (Ne, 5, 4)
        self.Gt = DNT[None, :, :] * (2.0 / dt)[:, None, None] / self.rq[:, :, None]
        self.w = 2.0 * np.pi * self.rq[:, :4] ** 2 * np.sin(self.tq[:, :4]) * (dr * dt / 4.0)[:, None]
        # element-centre coordinates
        self.rc, self.tc = self.rq[:, 4], self.tq[:, 4]
        self.Rc, self.zc = self.rc * np.sin(self.tc), self.rc * np.cos(self.tc)
        self.vol = self.w.sum(axis=1)
        # boundary nodes
        bnd = list(ids[Nr, :])
        if not self.origin:
            bnd += list(ids[0, :])
        self.bnd = np.unique(np.array(bnd))
        free = np.ones(self.nn, bool)
        free[self.bnd] = False
        self.free = np.nonzero(free)[0]
        rows = np.repeat(self.conn, 4, axis=1).ravel()
        cols = np.tile(self.conn, (1, 4)).ravel()
        self._rows, self._cols = rows, cols
        self.rc_ring = 0.5 * (r[1:] + r[:-1])
        self.tc_ring = 0.5 * (self.th[1:] + self.th[:-1])

    def ring(self, x):
        """Reshape an element array to (Nr, Nt)."""
        return np.asarray(x).reshape(self.Nr, self.Nt)

    # ----------------------------------------------------------------------------------------------------------
    def tensor(self, beta, nu):
        """Polar components of C from the axis angle beta (from r_hat toward theta_hat) and the stretch nu."""
        mu = 1.0 / nu
        c, s = np.cos(beta), np.sin(beta)
        return mu * c * c + nu * s * s, (mu - nu) * s * c, mu * s * s + nu * c * c

    def stiffness(self, Crr, Crt, Ctt):
        """Crr, Crt, Ctt: arrays (Ne, 4) at the Gauss points, or (Ne,) for one value per element."""
        if np.ndim(Crr) == 1:
            Crr, Crt, Ctt = (np.repeat(np.asarray(c)[:, None], 4, axis=1) for c in (Crr, Crt, Ctt))
        Gr, Gt, w = self.Gr[:, :4, :], self.Gt[:, :4, :], self.w
        Fr = Crr[:, :, None] * Gr + Crt[:, :, None] * Gt              # C . grad N_b, r component
        Ft = Crt[:, :, None] * Gr + Ctt[:, :, None] * Gt
        Ke = np.einsum("eq,eqa,eqb->eab", w, Gr, Fr) + np.einsum("eq,eqa,eqb->eab", w, Gt, Ft)
        K = sp.csr_matrix((Ke.ravel(), (self._rows, self._cols)), shape=(self.nn, self.nn))
        return K

    def load(self, rho_q):
        """b_a = int N_a 4 pi G rho dV, rho_q at the Gauss points (Ne, 4)."""
        be = np.einsum("eq,qa,eq->ea", self.w, NQ[:4], 4.0 * np.pi * rho_q)
        b = np.zeros(self.nn)
        np.add.at(b, self.conn.ravel(), be.ravel())
        return b

    def solve(self, K, b, phi_bnd):
        """Solve K Phi = -b with Phi = phi_bnd on the boundary nodes (array over self.bnd)."""
        phi = np.zeros(self.nn)
        phi[self.bnd] = phi_bnd
        f = self.free
        Kff = K[f][:, f].tocsc()
        rhs = -b[f] - K[f][:, self.bnd] @ phi[self.bnd]
        lu = spla.splu(Kff, permc_spec="MMD_AT_PLUS_A")
        phi[f] = lu.solve(rhs)
        return phi

    def grad(self, phi, q=4):
        """Polar components (d/dr, (1/r) d/dtheta) of grad Phi at point q of every element (q = 4: the centre)."""
        pe = phi[self.conn]
        return np.einsum("ea,ea->e", self.Gr[:, q, :], pe), np.einsum("ea,ea->e", self.Gt[:, q, :], pe)

    # ----------------------------------------------------------------------------------------------------------
    def ring_shift(self, eps):
        """For an angle field eps (Ne,) of a vector relative to a reference axis field (toward theta_hat), the shift
        of the centre that it implies at each ring: r <eps sin> / <sin^2>, weights sin(theta) dtheta. Linear order."""
        E = self.ring(eps)
        s = np.sin(self.tc_ring)
        w = s * (self.th[1:] - self.th[:-1])
        return self.rc_ring * (E * (s * w)[None, :]).sum(axis=1) / (s * s * w).sum()

    def ring_centre(self, psi):
        """Centre on the z-axis of a direction field psi (Ne,) at each ring: the weighted mean over theta of the
        point where the line through the element centre along psi crosses the axis, weights sin^3(theta) dtheta.
        Exact for a field radial about a point of the axis; equal to ring_shift at linear order."""
        R, z = self.ring(self.Rc), self.ring(self.zc)
        P = self.ring(psi)
        vR, vz = np.sin(P), np.cos(P)
        ok = np.abs(vR) > 1e-3
        zi = np.where(ok, z - R * vz / np.where(ok, vR, 1.0), 0.0)
        s = np.sin(self.tc_ring)
        w = (s ** 3 * (self.th[1:] - self.th[:-1]))[None, :] * ok
        return (zi * w).sum(axis=1) / w.sum(axis=1)


# --------------------------------------------------------------------------------------------------------------
class Law:
    """nu(X) = 1 + delta/(exp(lambda X) - 1), lambda = ln(1 + delta); coverage phi = 1 - exp(-lambda X); cap on nu."""

    def __init__(self, delta, cap=25.0):
        self.delta, self.lam, self.cap = delta, np.log1p(delta), cap
        self.X_cap = np.log1p(delta / (cap - 1.0)) / self.lam     # the drive at which nu reaches the cap

    def kin(self, X):
        """Kinetic data at the drive X with the cap: (T/T0, E = target stretch - 1). Below X_cap the stretch is held
        at the cap and the interval at its value there, T0 (cap - 1 + delta)/delta."""
        Xk = np.maximum(X, self.X_cap)
        return 1.0 / self.phi(Xk), self.nu(Xk) - 1.0

    def nu(self, X):
        return 1.0 + self.delta / np.expm1(self.lam * np.maximum(X, 1e-300))

    def dnu(self, X):
        e = np.expm1(self.lam * X)
        return -self.delta * self.lam * (e + 1.0) / e ** 2

    def phi(self, X):
        return -np.expm1(-self.lam * X)

    def nu_capped(self, X):
        return np.minimum(self.nu(X), self.cap)

    def X_of_y(self, y):
        """Solve X / nu(X) = y for X (uncapped), by bisection in ln X."""
        y = np.asarray(y, float)
        lo, hi = np.full(y.shape, -40.0), np.full(y.shape, 15.0)
        for _ in range(90):
            mid = 0.5 * (lo + hi)
            X = np.exp(mid)
            big = X / self.nu(X) > y
            hi = np.where(big, mid, hi)
            lo = np.where(big, lo, mid)
        return np.exp(0.5 * (lo + hi))

    def static(self, y):
        """Static aligned state at Newtonian field y = g_N/a0: (X, nu, capped flag), with the cap applied."""
        X = self.X_of_y(y)
        nu = self.nu(X)
        cap = nu > self.cap
        nu = np.where(cap, self.cap, nu)
        X = np.where(cap, self.cap * y, X)
        return X, nu, cap


class Body:
    """A compact body of density rho_c (1 - r^2)^2 inside r = 1, scaled so that the least static stretch is nu_min."""

    def __init__(self, law, nu_min=4.0):
        self.law = law
        u = np.linspace(1e-4, 1.0, 200001)
        m = u / 3.0 - 2.0 * u ** 3 / 5.0 + u ** 5 / 7.0                       # g_N / (4 pi rho_c), inside
        Xs = np.log1p(law.delta / (nu_min - 1.0)) / law.lam
        ystar = Xs / nu_min
        self.rho_c = ystar / (4.0 * np.pi * m.max())
        self.M = 4.0 * np.pi * self.rho_c * (8.0 / 105.0)
        self.r_numin = float(u[np.argmax(m)])

    def rho(self, r):
        r = np.asarray(r, float)
        return np.where(r < 1.0, self.rho_c * (1.0 - r * r) ** 2, 0.0)

    def drho(self, r):
        r = np.asarray(r, float)
        return np.where(r < 1.0, -4.0 * self.rho_c * r * (1.0 - r * r), 0.0)

    def gN(self, r):
        r = np.asarray(r, float)
        rr = np.minimum(r, 1.0)
        inside = 4.0 * np.pi * self.rho_c * (rr / 3.0 - 2.0 * rr ** 3 / 5.0 + rr ** 5 / 7.0)
        return np.where(r < 1.0, inside, self.M / np.maximum(r, 1e-300) ** 2)


class Static:
    """The static aligned sphere on a fine radial grid: X0, nu0, dPhi0/dr, Phi0 (zero at r_out), and derivatives."""

    def __init__(self, body, r_out, n=40001):
        law = body.law
        self.body, self.law, self.r_out = body, law, r_out
        r = np.concatenate([[0.0], np.geomspace(1e-6, r_out * 1.5, n)])
        y = body.gN(r)
        y[0] = 0.0
        X, nu, cap = law.static(np.maximum(y, 1e-300))
        self.r, self.y, self.X, self.nu, self.cap = r, y, X, nu, cap
        self.dphi = nu * y
        phi = np.concatenate([[0.0], np.cumsum(0.5 * (self.dphi[1:] + self.dphi[:-1]) * np.diff(r))])
        self.phi = phi - np.interp(r_out, r, phi)

    def at(self, name, r):
        return np.interp(r, self.r, getattr(self, name))

    def nu_of(self, r):
        return np.interp(r, self.r, self.nu)

    def phi_of(self, r):
        return np.interp(r, self.r, self.phi)


def cart(vr, vt, th):
    """Polar components (along r_hat, theta_hat) to (R, z) components."""
    return vr * np.sin(th) + vt * np.cos(th), vr * np.cos(th) - vt * np.sin(th)


def polar(vR, vz, th):
    return vR * np.sin(th) + vz * np.cos(th), vR * np.cos(th) - vz * np.sin(th)


def field_and_flux(mesh, phi, beta_c, nu_c):
    """Field g = grad Phi and flux F = C g at the element centres, polar components, with C from (beta_c, nu_c)."""
    gr, gt = mesh.grad(phi, 4)
    Crr, Crt, Ctt = mesh.tensor(beta_c, nu_c)
    return gr, gt, Crr * gr + Crt * gt, Crt * gr + Ctt * gt


def rule_vectors(gr, gt, Fr, Ft, beta_n):
    """The new axis under rules A, B, C as angles from r_hat (toward theta_hat), given the field, the flux and the
    old axis at angle beta_n. Rule B: component along the old axis from the flux, across it from the field."""
    nr, nt = np.cos(beta_n), np.sin(beta_n)
    Fn = Fr * nr + Ft * nt
    gn = gr * nr + gt * nt
    br, bt = Fn * nr + (gr - gn * nr), Fn * nt + (gt - gn * nt)
    return np.arctan2(gt, gr), np.arctan2(bt, br), np.arctan2(Ft, Fr)
