r"""The relaxed Chandra clusters of Vikhlinin et al. (2006, ApJ 640, 691): gas, stars and the hydrostatic mass.

The input is mprof.tex, the TeX source of the paper (arXiv:astro-ph/0507092), at
<data>/vikhlinin2006/x/mprof.tex. Its Table 2 gives an analytic model of n_p n_e for each cluster, and its
Eq. (9) the average 3D temperature profile, T/T_mg as a function of r/r500, which the clusters above 2.5 keV
follow to within 15 percent at r > 0.05 r500. The hydrostatic mass is the paper's Eq. (7),
    M(r) = -3.71e13 Msun T[keV] r[Mpc] (dln rho_g/dln r + dln T/dln r),
with the gas density rho_g = 1.252 m_p (n_p n_e)^(1/2). Radii and masses are in the paper's units
(H0 = 72 km/s/Mpc, Omega_m = 0.3).

The stars are an assumption: the stellar fraction within r500 of Giodini et al. (2009),
f_* = 0.05 (M500/5e13 Msun)^-0.37, of which 1e12 Msun is a central galaxy (a Hernquist profile with
a = 20 kpc) and the rest follows the paper's NFW fit of the total mass.
"""
import math
import re

import numpy as np

SAMPLE = ['A133', 'A383', 'A478', 'A907', 'A1413', 'A1795', 'A2029']   # T > 2.5 keV, A2390 excluded
M_P_G = 1.67262192e-24          # proton mass, g
MPC_CM = 3.0856775814913673e24
GCM3_TO_MSUN_MPC3 = MPC_CM ** 3 / 1.98847e33


def clean(s):
    s = s.replace('\\pz', '').replace('\\dotfill', '').replace('$', '').strip()
    return s


def parse_tables(tex):
    """The paper's Tables 1 to 3 per cluster: redshift, the gas-density model and the masses."""
    t = open(tex, encoding='latin-1').read()

    def table(label):
        i = t.index(label)
        a = t.index('\\startdata', i)
        b = t.index('\\enddata', a)
        return t[a + len('\\startdata'):b]
    out = {}
    for row in table('tab:sample').split('\\\\'):
        f = [clean(x) for x in row.split('&')]
        if len(f) >= 4 and f[0]:
            out.setdefault(f[0].replace('\\,', '').replace('~', ''), {})['z'] = float(f[1])
    for row in table('tab:rhogas:fit').split('\\\\'):
        f = [clean(x) for x in row.split('&')]
        if len(f) < 11 or not f[0]:
            continue
        name = f[0].replace('\\,', '').replace('~', '')
        nd = lambda s: None if 'nodata' in s else float(s)
        out.setdefault(name, {}).update(dict(rdet=float(f[1]), n0=float(f[2]), rc=float(f[3]), rs=float(f[4]),
                                             alpha=float(f[5]), beta=float(f[6]), eps=float(f[7]),
                                             n02=nd(f[8]), rc2=nd(f[9]), beta2=nd(f[10])))
    for row in table('tab:masses').split('\\\\'):
        f = [clean(x) for x in row.split('&')]
        if len(f) < 7 or not f[0]:
            continue
        name = f[0].replace('\\,', '').replace('~', '')
        num = lambda s: None if 'nodata' in s else float(re.split(r'\\pm|\^', s)[0])
        out.setdefault(name, {}).update(dict(r500=num(f[1]), Tspec=num(f[2]), Tmg=num(f[3]), c500=num(f[4]),
                                             M2500=num(f[5]), M500=num(f[6])))
    return out


class Cluster:
    """One cluster on a radial grid of 1 kpc to 4 Mpc: the gas mass, the temperature and the hydrostatic mass."""

    def __init__(self, name, p):
        self.name, self.p = name, p
        self.z = p['z']
        self.r500 = p['r500'] / 1e3          # Mpc
        self.r = np.geomspace(1e-3, 4.0, 3000)  # Mpc
        self.npne = self._npne(self.r * 1e3)
        self.rho_g = 1.252 * M_P_G * np.sqrt(self.npne) * GCM3_TO_MSUN_MPC3      # Msun/Mpc^3
        lr = np.log(self.r)
        dV = 4.0 * math.pi * self.r ** 3 * np.gradient(lr)
        self.Mgas = np.cumsum(self.rho_g * dV)
        self.T = self._T(self.r)
        dlnrho = np.gradient(np.log(self.rho_g), lr)
        dlnT = np.gradient(np.log(self.T), lr)
        self.Mhse = -3.71e13 * self.T * self.r * (dlnrho + dlnT)

    def _npne(self, rk):
        """The paper's Eq. (3) for n_p n_e, radius in kpc."""
        p = self.p
        g = 3.0
        a = p['n0'] * 1e-3
        x = rk / p['rc']
        v = a * a * x ** (-p['alpha']) / (1 + x * x) ** (3 * p['beta'] - p['alpha'] / 2) \
            / (1 + (rk / p['rs']) ** g) ** (p['eps'] / g)
        if p['n02'] is not None:
            b = p['n02'] * 1e-1
            v = v + b * b / (1 + (rk / p['rc2']) ** 2) ** (3 * p['beta2'])
        return v

    def _T(self, r):
        """The average temperature profile, the paper's Eq. (9)."""
        x = r / self.r500
        y = (x / 0.045) ** 1.9
        return self.p['Tmg'] * 1.35 * (y + 0.45) / (y + 1.0) / (1.0 + (x / 0.6) ** 2) ** 0.45

    def M_at(self, arr, r):
        return np.interp(np.log(r), np.log(self.r), arr)


def stars(cl):
    """The enclosed stellar mass on the cluster's grid (Msun) and the stellar fraction within r500."""
    M500 = cl.p['M500'] * 1e14
    fstar = 0.05 * (M500 / 5e13) ** (-0.37)
    Mtot = fstar * M500
    Mbcg = 1e12
    a = 0.02
    bcg = Mbcg * cl.r ** 2 / (cl.r + a) ** 2
    rs = cl.r500 / cl.p['c500']
    m = lambda x: np.log1p(x) - x / (1 + x)
    sat = (Mtot - Mbcg) * m(cl.r / rs) / m(cl.p['c500'])
    return bcg + sat, fstar
