r"""A MOND-type algebraic law against the data of the cluster cores, the groups and the web.

The law is MOND's simple interpolation function at a0 = 1.2e-10 m/s^2 (mond_law.py). The script computes the
entries of Paper I's Table 1, last column, for the groups and clusters and for the large-scale structure, on
which the features listed below the table in Paper I's Section 2.4 rest:
  1. the cores of seven relaxed Chandra clusters (Vikhlinin et al. 2006): the law's field of the gas and the
     stars over the hydrostatic field, a median of 0.25 and 0.31 at 0.1 and 0.3 r500, "short of ... the cores'
     field";
  2. the GAMA groups stacked in the KiDS-1000 shear: the law's lensing over the measured one at 0.1 to 0.3 Mpc,
     a median of 0.61 over the six bins of 1e13 to 1e14 Msun, "short of the groups' lensing inside 0.3 Mpc";
  3. the lensing of the web: the baryons' share of the standard matter field, boosted by the law, against the
     CMB lensing of Planck 2018 and ACT DR6 and the cosmic shear of KiDS-1000. The law's lensing power is 4.1 to
     12.5 times the measured CMB lensing power and 14 to 247 times the measured shear, "above the measured
     lensing".
Table 1's entry for the galaxies comes from Paper I's own fits, and its entry for the small systems from
mond_gc.py.

Inputs, under the data folder passed as the first argument (default data):
    vikhlinin2006/x/mprof.tex                    the TeX source of Vikhlinin et al. (2006), arXiv:astro-ph/0507092
    planck2018-lensing/                          the Planck 2018 lensing bandpowers, covariance, fiducial
                                                 correction and the window files win/ and lensdelta/
    act-dr6/v1.2/                                binning_matrix_act.txt, clkk_bandpowers_act.txt and
                                                 covmat_act_cmbmarg.txt of the ACT DR6 lensing likelihood
    kids1000/xipm_k1000.fits                     the KiDS-1000 cosmic-shear data vector, covariance and n(z)
and, deposited beside this script, groups-stack-profiles.json, the stacked profiles of the groups and their halo
fits, which groups_stack.py writes from the KiDS-1000 shear catalogue and the GAMA groups (group_models.py
describes the fields). None of the public releases is redistributed.

Output: mond-classes/mond_classes-results.txt, also printed.
Command, from the repository root (about 10 s, most of it the CAMB run):
    python -I -B mond-classes/mond_classes.py data
Needs numpy, scipy, astropy and camb.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8')
from mond_law import A0, G, MSUN, MPC, mond_g, mond_newtonian_of_total, nu_simple  # noqa: E402
import cluster_models as cm  # noqa: E402
import group_models as gm  # noqa: E402
import web_lensing as wl  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else 'data'
RESULTS = os.path.join(HERE, 'mond_classes-results.txt')
GROUPS = os.path.join(HERE, 'groups-stack-profiles.json')

from background import OMEGA_M as OM, OMEGA_B as OB  # noqa: E402  (the Planck 2018 background of the template)
FB = OB / OM             # the baryons' share of the matter

LINES = []


def P(s=''):
    LINES.append(s)
    print(s, flush=True)


# ----------------------------------------------------------------------------------------------
# 0. the checks of the law
# ----------------------------------------------------------------------------------------------
def section0():
    P('0. CHECKS OF THE LAW')
    P('-' * 110)
    gN = np.geomspace(1e-15, 1e-8, 400)
    g = mond_g(gN)
    P(f'  the inverse of the simple relation: largest relative error {np.max(np.abs(mond_newtonian_of_total(g) / gN - 1.0)):.1e}')
    P(f'  the deep and Newtonian limits: g/sqrt(g_N a0) at g_N = 1e-16 m/s^2 is {float(mond_g(1e-16) / math.sqrt(1e-16 * A0)):.5f};'
      f' g/g_N at g_N = 1e-7 m/s^2 is {float(mond_g(1e-7) / 1e-7):.5f}')
    P('')


# ----------------------------------------------------------------------------------------------
# 1. the cluster cores
# ----------------------------------------------------------------------------------------------
def section1():
    P('1. THE CORES OF SEVEN RELAXED CHANDRA CLUSTERS (Vikhlinin et al. 2006)')
    P('-' * 110)
    P('The paper\'s gas models and average temperature profile, the stars of Giodini et al. (2009) (cluster_models.py);')
    P('the law\'s field of the gas and the stars over the hydrostatic field, g/g_HSE, at x = r/r500, median of the seven clusters.')
    tabs = cm.parse_tables(os.path.join(DATA, 'vikhlinin2006', 'x', 'mprof.tex'))
    xs = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    xfine = np.geomspace(0.05, 1.0, 60)
    per, clusters = [], []
    for name in cm.SAMPLE:
        cl = cm.Cluster(name, tabs[name])
        Mst, _ = cm.stars(cl)
        rr = xfine * cl.r500
        rm = rr * MPC
        Mh = cl.M_at(cl.Mhse, rr)
        Mb = cl.M_at(cl.Mgas, rr) + cl.M_at(Mst, rr)
        g_hse = G * Mh * MSUN / rm ** 2
        g_bar = G * Mb * MSUN / rm ** 2
        ratio = mond_g(g_bar) / g_hse
        per.append(np.interp(np.log(xs), np.log(xfine), ratio))
        clusters.append((name, ratio))
    med = np.median(np.array(per), axis=0)
    P('  ' + 'x = r/r500'.ljust(52) + ''.join(f'{x:8.2f}' for x in xs))
    P('  ' + 'MOND, algebraic'.ljust(52) + ''.join(f'{v:8.3f}' for v in med))
    P(f'  MOND gives {med[1]:.2f}, {med[3]:.2f} and {med[6]:.2f} of the hydrostatic field at 0.1, 0.3 and 1 r500.')
    P('  Per cluster, MOND at 0.1, 0.3 and 1 r500: ' + '; '.join(
        f'{name} ' + ', '.join(f'{v:.3f}' for v in np.interp(np.log([0.1, 0.3, 1.0]), np.log(xfine), ratio))
        for name, ratio in clusters))
    P('')


# ----------------------------------------------------------------------------------------------
# 2. the groups
# ----------------------------------------------------------------------------------------------
GAS = (0.5, 0.707107, 1.0, 1.41421, 2.0)     # factors on the gas fraction of Lovisari et al. (2015)


def section2():
    P('2. THE GAMA GROUPS STACKED IN THE KiDS-1000 SHEAR')
    P('-' * 110)
    bins = gm.load_profiles(GROUPS)
    keys = list(bins)
    models = {'MOND, algebraic': lambda gN: mond_g(gN)}
    P('Each bin\'s baryons (group_models.py): the bin\'s stars and BCG at 2 Msun/Lsun in the halo fitted to its profile,')
    P('and the hot gas of Lovisari et al. (2015) at the fitted M500c, truncated at r200c. They are boosted by the law and')
    P('projected to the excess surface density. Model over data from the amplitude fit with the jackknife covariance')
    P('(Hartlap-corrected) in each band, +- its error, and the chi2 of the model as computed for the band\'s radial bins.')
    P('The gas fraction is scanned from half to twice the relation.')
    out = {}
    for key in keys:
        prof = bins[key]
        R = np.asarray(prof['R'])
        d = np.asarray(prof['esd']) * 1e12
        C = np.asarray(prof['cov']) * 1e24
        npatch = prof['npatch']
        b0 = gm.bin_halo(prof['M200m'], prof['z'], prof['Mc'], prof['Mstar_tot'])
        rec = dict(z=prof['z'], M500c=b0['M500c'], in_range=1e13 <= b0['M500c'] <= 1e14, models={}, gas_scan={})
        for fg in GAS:
            gN = gm.GSI * gm.baryons(b0, fgas=fg) / gm.R_GRID ** 2
            for lab, fn in models.items():
                Meff = fn(gN) * (gm.R_GRID * MPC) ** 2 / G / MSUN
                e = gm.esd_from_mass(R, gm.R_GRID, Meff)
                bs = gm.band_stats(d, C, npatch, e, R)
                if fg == 1.0:
                    rec['models'][lab] = bs
                rec['gas_scan'].setdefault(lab, {})[f'{fg:g}'] = {k: v['ratio'] for k, v in bs.items()}
        out[key] = rec
    P('')
    bands = list(gm.BANDS)
    P('  bin  M500c (1e13 Msun)  z      in range | ' + ' | '.join(f'{b} Mpc'.ljust(27) for b in bands))
    for lab in models:
        P(f'  {lab}')
        for key in keys:
            r = out[key]
            bb = r['models'][lab]
            P(f"   {key}   {r['M500c'] / 1e13:6.2f}           {r['z']:.3f}  {'yes' if r['in_range'] else 'no '}     | "
              + ' | '.join(f"{bb[b]['ratio']:.2f} +- {bb[b]['ratio_err']:.2f} ({bb[b]['chi2']:5.1f}/{bb[b]['n']})".ljust(27) for b in bands))
    inr = [k for k in keys if out[k]['in_range']]
    P('')
    P(f'  Over the {len(inr)} bins with M500c of 1e13 to 1e14 Msun (' + ', '.join(inr) + '): the median model over data, the')
    P('  range over the bins, the range over the gas scan, the summed chi2.')
    for lab in models:
        cells = []
        for b in bands:
            v = [out[k]['models'][lab][b]['ratio'] for k in inr]
            c2 = sum(out[k]['models'][lab][b]['chi2'] for k in inr)
            n = sum(out[k]['models'][lab][b]['n'] for k in inr)
            gs = [out[k]['gas_scan'][lab][f'{fg:g}'][b] for k in inr for fg in GAS]
            cells.append(f'{np.median(v):.2f} ({min(v):.2f} to {max(v):.2f}; gas {min(gs):.2f} to {max(gs):.2f}; chi2 {c2:.1f}/{n})')
        P(f'  {lab:30s} ' + ' | '.join(cells))
    P('')


# ----------------------------------------------------------------------------------------------
# 3. the web
# ----------------------------------------------------------------------------------------------
def boosts(tpl, kgrid_h, zgrid, omega, R_smooth=2.0):
    """The law's boost of the baryons' Newtonian field of structure on a (k, z) grid, k in h/Mpc: per scale, and
    with the rms field smoothed at R_smooth (Mpc) acting as an external field (the QUMOND linear response)."""
    h = tpl.h
    H0 = 100.0 * h
    k_mpc = kgrid_h * h
    nu_s = np.empty((kgrid_h.size, zgrid.size))
    nu_e = np.empty_like(nu_s)
    ks_h = np.geomspace(1e-4, 30.0, 600)
    for j, z in enumerate(zgrid):
        Pl = tpl.P(np.full_like(kgrid_h, z), kgrid_h, nonlinear=False) / h ** 3
        gN = wl.accel_per_lnk(omega, z, k_mpc, Pl, H0)
        nu_s[:, j] = nu_simple(gN / A0)
        Prow = tpl.P(np.full_like(ks_h, z), ks_h, nonlinear=False) / h ** 3
        ye = wl.accel_rms(omega, z, ks_h * h, Prow, H0, R=R_smooth) / A0
        nu0 = float(nu_simple(np.array([ye]))[0])
        s = float((np.log(nu_simple(np.array([ye * 1.001]))[0]) - math.log(nu0)) / math.log(1.001))
        nu_e[:, j] = nu0 * math.sqrt(1.0 + 2.0 * s / 3.0 + s * s / 5.0)
    return nu_s, nu_e


def section3():
    P('3. THE LENSING OF THE WEB: THE BARYONS\' FIELD OF STRUCTURE UNDER THE LAW')
    P('-' * 110)
    P(f'The lensing contrast is f_b nu_b(k, z) delta_m: the baryons, a share f_b = {OB:.4f}/{OM:.4f} of the matter, trace the')
    P('standard matter field delta_m of the Planck 2018 template, and nu_b is the law\'s boost of the baryons\' own')
    P('Newtonian field, either per scale or with the rms field smoothed at 2 Mpc acting as an external field with the')
    P(f'QUMOND response. The distances and the lensing kernels use the template\'s flat background, Omega_m = {OM:.4f} (background.py).')
    tpl = wl.Template(zs=wl.default_z_cmb())
    geom = wl.Geometry(om=OM)
    Lc = np.unique(np.round(np.geomspace(2, 4000, 260)).astype(int)).astype(float)
    zc = wl.cmb_lensing_z(n=900)
    grid = wl.LimberGrid(geom, Lc, zc)
    Pz = grid.power(tpl)
    kern = wl.cmb_kernel_sq(geom, zc)
    kt = np.geomspace(1e-4, 50.0, 160)
    zt = np.unique(np.concatenate([np.linspace(0.0, 5.0, 26), np.geomspace(5.5, 1100.0, 16)]))
    tab = boosts(tpl, kt, zt, OB)

    def on_grid(t, kk, zz):
        lk = np.log(np.clip(kk, kt[0], kt[-1]))
        out = np.empty_like(kk)
        for j in range(kk.shape[1]):
            col = np.array([np.interp(zz[j], zt, t[i, :]) for i in range(kt.size)])
            out[:, j] = np.interp(lk[:, j], np.log(kt), col)
        return out

    def at(t, z, k_inv_mpc):
        return float(np.interp(math.log(k_inv_mpc / tpl.h), np.log(kt), [np.interp(z, zt, t[i]) for i in range(kt.size)]))
    P('  the per-scale boost of the baryons at k = 0.03 /Mpc: ' + ', '.join(f'z = {z}: {at(tab[0], z, 0.03):.1f}' for z in (0.5, 2.0)))
    P('  the boost in the external field (independent of k): ' + ', '.join(f'z = {z}: {at(tab[1], z, 0.03):.2f}' for z in (0.0, 0.5, 2.0)))
    names = {'standard': None, 'MOND, external field': 1, 'MOND, per scale': 0}
    curves = {}
    for n, spec in names.items():
        comp = np.ones_like(grid.k) if spec is None else FB * on_grid(tab[spec], grid.k, zc)
        curves[n] = grid.cl(comp ** 2 * Pz, kern)

    def full(c):
        L = np.arange(0, 4001)
        out = np.interp(L, Lc, c)
        out[:2] = 0.0
        return out
    pl = wl.PlanckLensing(os.path.join(DATA, 'planck2018-lensing'))
    act = wl.ACTLensing(os.path.join(DATA, 'act-dr6', 'v1.2'))
    P('')
    P('  CMB lensing: the amplitude of the data relative to the model (data = alpha x model), its error, chi2 at alpha = 1;')
    P('  model over data in power = 1/alpha, and in the lensing amplitude (the boost of the contrast) = 1/sqrt(alpha).')
    P('  ' + 'model'.ljust(22) + 'Planck alpha        chi2 (9)    1/alpha  1/sqrt   ACT alpha           chi2 (10)   1/alpha  1/sqrt')
    for n in names:
        c = full(curves[n])
        ap, ape = pl.amplitude(c)
        aa, aae = act.amplitude(c)
        P(f'  {n:21s} {ap:7.4f} +/- {ape:.4f}  {pl.chi2(c):10.1f}   {1 / ap:6.2f}   {1 / math.sqrt(ap):5.2f}   '
          f'{aa:7.4f} +/- {aae:.4f}  {act.chi2(c):10.1f}   {1 / aa:6.2f}   {1 / math.sqrt(aa):5.2f}')
    sd = wl.load_kids1000(os.path.join(DATA, 'kids1000', 'xipm_k1000.fits'))
    sm = wl.ShearModel(sd, geom)
    Ps = sm.grid.power(tpl)
    zs = sm.z
    Cinv = np.linalg.inv(sd['cov'])
    P('')
    P(f"  KiDS-1000 cosmic shear, {len(sd['data'])} points of xi+ and xi-, survey cuts, intrinsic alignment off: alpha = data/model")
    P('  (the amplitude of xi), chi2 at alpha = 1 and at the best alpha, model over data = 1/alpha.')
    P('  ' + 'model'.ljust(22) + 'alpha       chi2 at 1        chi2 at best   1/alpha')
    for n, spec in names.items():
        comp = np.ones_like(sm.grid.k) if spec is None else FB * on_grid(tab[spec], sm.grid.k, zs)
        v = sm.vector(comp ** 2 * Ps)
        d = sd['data']
        a = float((v @ Cinv @ d) / (v @ Cinv @ v))
        r1, rb = d - v, d - a * v
        P(f'  {n:21s} {a:8.4f}  {float(r1 @ Cinv @ r1):14.1f}  {float(rb @ Cinv @ rb):10.1f}   {1 / a:8.2f}')
    P('')


# ----------------------------------------------------------------------------------------------
def main():
    P('A MOND-TYPE LAW AGAINST THE CLUSTER CORES, THE GROUPS AND THE WEB')
    P('=' * 110)
    P(f'data folder: {DATA}')
    P('The law: MOND\'s simple interpolation function, nu(y) = 1/2 + sqrt(1/4 + 1/y) with y = g_N/a0, at')
    P('a0 = 1.2e-10 m/s^2 (mond_law.py). In the web, one prescription lets the field of structure act as an external field.')
    P('')
    section0()
    section1()
    section2()
    section3()
    with open(RESULTS, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LINES) + '\n')


if __name__ == '__main__':
    main()
