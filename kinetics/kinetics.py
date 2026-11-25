"""Steady state of a body moving through the fabric: stored axes and stretches advected past it.

Body frame: the body (axfem.Body) rests at the origin and the fabric flows past at -v z_hat (the body moves at +v).
The coarse-grained state at each element centre is the stretch nu = 1 + tr S and the axis n of S, where S is the
stretch-weighted axis tensor of the renewal domains and Q their unweighted axis tensor. Following the fabric,

    dQ/dt = (m m - Q) / T,        dS/dt = (E / T) Q - S / T,        T = T0 / phi(X),   E = nu_law(X) - 1,

with X = |g|/a0 the local drive and m the axis that a smoothing lays (rule A, B or C). Where the drive is so weak
that the stretch reaches its cap of 25 (only near the body's centre), T and E are held at their values at the cap. This is the exact mean of a
population of domains smoothed at the rate 1/T, each gaining stretch at the rate E/T = delta (1 - phi)/T0 along its own
axis; the stretch-weighted mean age is 2T. With axes in the meridional plane the axis angles obey, to first order in
the spread of axes over the domains (angles psi from +z toward +R),

    d psi_Q/dt = (psi_m - psi_Q)/T,     d psi_S/dt = rho (psi_Q - psi_S)/T,   rho = E/(nu - 1),
    psi_m = psi_S + kappa (psi_g - psi_S),   kappa = atan(nu^q tan(psi_g - psi_S)) / (psi_g - psi_S),

q = 0, 1, 2 for rules A, B, C: tan of the offset of rule B's vector (flux along the old axis, field across it) from the
old axis is nu times the field's, and that of the flux nu^2 times. kappa is taken from the previous iterate, so the
rule is exact at the fixed point while the march along a streamline stays linear and stable for any kappa.

Steady state: along every streamline (fixed R, z increasing upstream) the pair (psi_Q, psi_S) and the stretch are
marched downstream from 12 intervals upstream with the exact solution of the linear system for coefficients frozen
over each step and the field direction linear over it. The field is that of the current tensor (Dirichlet Phi = 0
at r_out: the fabric beyond r_out holds the static arrangement about the body, and paths that leave the grid
upstream meet that state). The fixed point of (state -> field -> march -> state) is found with Anderson mixing; the
stretch update is damped by a geometric mean, which leaves the fixed point unchanged.

A module, with no output of its own. Imported by kinetics/moving_body.py. Needs numpy and scipy.
"""
import time

import numpy as np

import axfem as A


class Interp:
    """Bilinear interpolation of element-centre fields on the (r, theta) grid of element centres, with one ghost
    column beyond each end of the z-axis taken from the mirror image (scalars and cos 2psi even, sin 2psi odd);
    static values beyond r_out are supplied by the caller."""

    def __init__(self, mesh):
        self.m = mesh
        self.rc = mesh.rc_ring
        t_ring = mesh.tc_ring
        self.tc = np.concatenate([[-t_ring[0]], t_ring, [2.0 * np.pi - t_ring[-1]]])
        self.ir = np.arange(mesh.Nr, dtype=float)
        self.jt = np.arange(mesh.Nt + 2, dtype=float)

    def extend(self, F, odd=False):
        s = -1.0 if odd else 1.0
        return np.concatenate([s * F[:, :1], F, s * F[:, -1:]], axis=1)

    def weights(self, r, th):
        fi = np.interp(r, self.rc, self.ir)
        fj = np.interp(th, self.tc, self.jt)
        i0 = np.minimum(np.floor(fi).astype(int), self.m.Nr - 2)
        j0 = np.minimum(np.floor(fj).astype(int), self.m.Nt)
        return i0, j0, fi - i0, fj - j0

    @staticmethod
    def apply(F, w):
        i0, j0, a, b = w
        return ((1 - a) * (1 - b) * F[i0, j0] + a * (1 - b) * F[i0 + 1, j0]
                + (1 - a) * b * F[i0, j0 + 1] + a * b * F[i0 + 1, j0 + 1])


class Mover:
    def __init__(self, mesh, body, rule_q, vT0, c_age=12.0, k_min=48, cells_per_step=1.0, k_max=600, h_floor=0.02):
        self.m, self.body, self.law, self.q, self.vT0 = mesh, body, body.law, rule_q, vT0
        self.stat = A.Static(body, mesh.r[-1])
        self.r_out = mesh.r[-1]
        self.rho_q = body.rho(mesh.rq[:, :4])
        self.b = mesh.load(self.rho_q)
        self.ip = Interp(mesh)
        self.c_age, self.k_min, self.cps, self.k_max = c_age, k_min, cells_per_step, k_max
        self.zero_bnd = np.zeros(len(mesh.bnd))
        # local cell size at each element centre
        self.h = np.maximum(np.minimum(mesh.dr, mesh.rc * mesh.dt), h_floor)

    # ------------------------------------------------------------------------------------------------------
    def solve_field(self, beta, nu):
        m = self.m
        K = m.stiffness(*m.tensor(beta, nu))
        phi = m.solve(K, self.b, self.zero_bnd)
        gr, gt, Fr, Ft = A.field_and_flux(m, phi, beta, nu)
        return phi, gr, gt, Fr, Ft

    def local(self, beta, nu, gr, gt):
        """Field angle psi_g, the rule gain kappa, ln T (T in units of T0), and E at the element centres."""
        m, law = self.m, self.law
        X = np.hypot(gr, gt)
        psi_g = m.tc + np.arctan2(gt, gr)
        psi_S = m.tc + beta
        dg = A.wrap_half(psi_g - psi_S)
        if self.q == 0:
            kap = np.ones_like(dg)
        else:
            nq = nu ** self.q
            small = np.abs(dg) < 1e-9
            kap = np.where(small, nq, np.arctan(nq * np.tan(dg)) / np.where(small, 1.0, dg))
        Tk, E = law.kin(X)
        lnT = np.log(Tk)
        return psi_g, kap, lnT, E, X

    def march(self, psi_g, kap, lnT, E):
        """Steady memory along the streamlines through every element centre. Returns (psi_S, nu) there."""
        m, ip, st = self.m, self.ip, self.stat
        Nr, Nt = m.Nr, m.Nt
        F = {"c2": ip.extend(m.ring(np.cos(2 * psi_g))), "s2": ip.extend(m.ring(np.sin(2 * psi_g)), odd=True),
             "lnT": ip.extend(m.ring(lnT)), "E": ip.extend(m.ring(E)), "kap": ip.extend(m.ring(kap))}
        T_p = np.exp(lnT)
        a_max = self.c_age * T_p                                  # ages in units of T0 (time = age * T0)
        n_need = a_max * self.vT0 / (self.cps * self.h)
        K = int(min(max(self.k_min, np.ceil(n_need.max())), self.k_max))
        dA = a_max / K
        R, z0 = m.Rc, m.zc
        qpow = self.q

        def sample(k):
            z = z0 + self.vT0 * k * dA
            r = np.hypot(R, z)
            th = np.arctan2(R, z)
            w = ip.weights(r, th)
            c2, s2 = ip.apply(F["c2"], w), ip.apply(F["s2"], w)
            psi = 0.5 * np.arctan2(s2, c2)
            lt, e, kp = ip.apply(F["lnT"], w), ip.apply(F["E"], w), ip.apply(F["kap"], w)
            far = r > self.r_out
            if far.any():
                nu_s = st.nu_of(r[far])
                Tk, Ek = self.law.kin(st.at("X", r[far]))
                psi = np.where(far, th, psi)
                lt[far] = np.log(Tk)
                e[far] = Ek
                kp[far] = nu_s ** qpow if qpow else 1.0
            return psi, lt, e, kp

        psi_prev, lt_a, e_a, kp_a = sample(K)
        pQ = psi_prev.copy()
        pS = psi_prev.copy()
        x = e_a.copy()                                             # nu - 1
        for k in range(K - 1, -1, -1):
            praw, lt_b, e_b, kp_b = sample(k)
            psi_new = psi_prev + A.wrap_half(praw - psi_prev)
            T = np.exp(0.5 * (lt_a + lt_b))
            tau = dA / T
            # stretch, exact for E linear over the step
            Ed = (e_b - e_a) / dA
            ex = np.exp(-tau)
            x_new = e_b - Ed * T + (x - e_a + Ed * T) * ex
            rho = 0.5 * (e_a + e_b) / np.maximum(0.5 * (x + x_new), 1e-12)
            kap = 0.5 * (kp_a + kp_b)
            # angles: e' = (M/T) e - gdot (1, 1), e = (psiQ, psiS) - psi_g(t)
            gdot = (psi_new - psi_prev) / dA
            e1, e2 = pQ - psi_prev, pS - psi_prev
            m11, m12, m21, m22 = -np.ones_like(rho), 1.0 - kap, rho, -rho
            s = 0.5 * (m11 + m22) * tau
            det = (m11 * m22 - m12 * m21) * tau * tau
            qq = np.sqrt((s * s - det).astype(complex))
            small = np.abs(qq) < 1e-8
            sh = np.where(small, 1.0 + qq * qq / 6.0, np.sinh(qq) / np.where(small, 1.0, qq))
            ch = np.cosh(qq)
            es = np.exp(s)
            E11 = np.real(es * (ch + sh * (m11 * tau - s)))
            E12 = np.real(es * sh * m12 * tau)
            E21 = np.real(es * sh * m21 * tau)
            E22 = np.real(es * (ch + sh * (m22 * tau - s)))
            # A^{-1} (E - I) c, c = -gdot (1,1), A^{-1} = T M^{-1}
            c1 = -gdot
            u1 = (E11 - 1.0) * c1 + E12 * c1
            u2 = E21 * c1 + (E22 - 1.0) * c1
            dt_ = m11 * m22 - m12 * m21
            w1 = T * (m22 * u1 - m12 * u2) / dt_
            w2 = T * (-m21 * u1 + m11 * u2) / dt_
            n1 = E11 * e1 + E12 * e2 + w1
            n2 = E21 * e1 + E22 * e2 + w2
            pQ, pS = psi_new + n1, psi_new + n2
            x = x_new
            psi_prev, lt_a, e_a, kp_a = psi_new, lt_b, e_b, kp_b
        return pS, 1.0 + x, K

    # ------------------------------------------------------------------------------------------------------
    def G(self, xvec):
        Ne = self.m.Ne
        beta, lnnu = xvec[:Ne], xvec[Ne:]
        nu = np.exp(lnnu)
        phi, gr, gt, Fr, Ft = self.solve_field(beta, nu)
        psi_g, kap, lnT, E, X = self.local(beta, nu, gr, gt)
        pS, nu_new, K = self.march(psi_g, kap, lnT, E)
        beta_new = A.wrap_half(pS - self.m.tc)
        out = np.concatenate([beta_new, 0.5 * (lnnu + np.log(nu_new))])
        self.last = dict(phi=phi, gr=gr, gt=gt, Fr=Fr, Ft=Ft, psi_g=psi_g, X=X, K=K, beta=beta, nu=nu)
        return out

    def _resid(self, x):
        Ne = self.m.Ne
        gx = self.G(x)
        f = gx - x
        f[:Ne] = A.wrap_half(f[:Ne])                              # axes are lines: beta and beta + pi agree
        return f

    def solve(self, x0, tol=1e-9, max_it=400, m_and=25, mix=1.0, log=None, watch=None, r_core=0.15, mix_core=0.5,
              cycles=3, core_its=30, r_conv=0.2, whole_rounds=False, last_cycle=False):
        """Fixed point of x = G(x), x = (axis angles from the radial direction, ln nu) at the element centres.

        Within r_core of the centre the drive is weak, the stretch sits at its cap, and at the lowest speed the field's
        zero already lies a few hundredths of R behind the centre; the direction is singular there and the state does
        not settle under any mixing tried. The solve therefore alternates, for a number of cycles: core_its rounds of
        plain mixing of the state inside r_core with the outer state held; then Anderson mixing (type II, depth m_and)
        of the state outside r_core with the core frozen, to the tolerance tol on the residual outside r_conv. The
        change of the outer state between cycles measures what the unsettled core does to it, and the state after the
        cycle with the least outer residual is returned. With whole_rounds the plain rounds mix the whole state, core
        and outside alike; with last_cycle the state after the last cycle is returned. The runs of rules A and B in
        kinetics/moving_body-results.txt take both, and every cycle of them converged. Rule C does not settle that way
        with a core of 0.16 R, and its runs take rounds of the core alone and a core of 0.3 R. Returns x, total
        iterations, residuals (outer axes in radians, outer ln nu, core axes, core ln nu at the last plain round), and
        a list of the outer states after each cycle."""
        x = x0.copy()
        Ne = self.m.Ne
        core1 = self.m.rc < r_core
        core = np.concatenate([core1, core1])
        oi = np.nonzero(~core)[0]
        conv1 = self.m.rc > r_conv
        t0 = time.time()
        n_it = 0
        snaps = []
        cres = (0.0, 0.0)
        best = None
        for cyc in range(cycles):
            for k in range(core_its):
                f = self._resid(x)
                n_it += 1
                cres = (np.abs(f[:Ne][core1]).max(), np.abs(f[Ne:][core1]).max())
                if whole_rounds:
                    x = x + mix_core * f                         # the whole state is mixed
                else:
                    x[core] = x[core] + mix_core * f[core]       # the outer state is held during these rounds
            if log is not None:
                w = watch(self) if watch else ""
                log(f"      cycle {cyc + 1}: after {core_its} plain rounds, core residual {cres[0]:.1e} rad, {cres[1]:.1e}; {w} "
                    f"({time.time() - t0:.0f} s)")
            Xh, Fh = [], []
            for it in range(1, max_it + 1):
                f = self._resid(x)
                n_it += 1
                fo = f[oi]
                res = (np.abs(f[:Ne][conv1]).max(), np.abs(f[Ne:][conv1]).max())
                if log is not None and (it <= 2 or it % 20 == 0):
                    w = watch(self) if watch else ""
                    log(f"        it {it:4d}: outer residual axes {res[0]:.2e} rad, ln nu {res[1]:.2e}; K = {self.last['K']}; {w} "
                        f"({time.time() - t0:.0f} s)")
                if max(res) < tol:
                    break
                xo = x[oi]
                Xh.append(xo.copy())
                Fh.append(fo.copy())
                if len(Xh) > m_and + 1:
                    Xh.pop(0)
                    Fh.pop(0)
                if len(Xh) >= 2:
                    dX = np.array([Xh[i + 1] - Xh[i] for i in range(len(Xh) - 1)]).T
                    dF = np.array([Fh[i + 1] - Fh[i] for i in range(len(Fh) - 1)]).T
                    gam, *_ = np.linalg.lstsq(dF, fo, rcond=1e-12)
                    x[oi] = xo + mix * fo - (dX + mix * dF) @ gam
                else:
                    x[oi] = xo + mix * fo
            if log is not None:
                log(f"      cycle {cyc + 1}: outer solve {it} iterations, residual {res[0]:.1e} rad, {res[1]:.1e}")
            snaps.append(x.copy())
            rec = (max(res), cyc, x.copy(), (res[0], res[1], cres[0], cres[1]))
            if best is None or rec[0] < best[0] or last_cycle:
                best = rec
        self.best_cycle = best[1] + 1
        if log is not None and not last_cycle:
            log(f"      kept the state after cycle {best[1] + 1}, the one with the least outer residual")
        return best[2], n_it, best[3], snaps
