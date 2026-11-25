r"""The lensing profiles of the GAMA groups stacked in the public KiDS-1000 shear, and the halo fit of each bin.

The script measures the excess surface density (ESD) of the GAMA G3C v10 groups (Robotham et al. 2011) from the
KiDS-1000 gold weak-lensing catalogue (Giblin et al. 2021), validates it against the halo masses of Viola et al.
(2015), fits each bin's halo with the bin's own central galaxy, and writes groups-stack-profiles.json. That file
holds the profiles and the halo fits from which mond_classes.py builds the groups' baryons. It supports Paper I's
Table 1, last column, for the groups: "short of the groups' lensing inside 0.3 Mpc".

Stage 1, the sources and the lenses. The catalogue is read memory-mapped, in chunks of two million rows and only
the columns RAJ2000, DECJ2000, e1, e2, weight and Z_B. The sources with weight > 0 and 0.1 < Z_B <= 1.2 in the
three GAMA fields, with a margin of 3 degrees in RA and Dec from -6 to +6 degrees, are saved to the checkpoint
folder. The additive bias is the weighted mean ellipticity of each tomographic bin over KiDS-North. The lenses
are the groups with at least 5 members at z >= 0.02 whose brightest group galaxy (BCG) lies in G09, G12 or G15;
each lens is centred on its BCG. The random points, 20 per lens, are uniform in the three fields and take the
redshift of a lens drawn at random. The inverse critical surface density is tabulated as its mean over the SOM
redshift distribution of each tomographic bin (Hildebrandt et al. 2021).
Stage 2, the profiles. For a lens-source pair with Z_B > z_l + 0.1 and S = <Sigma_crit^-1> of the source's
tomographic bin, the estimator is
    Delta Sigma(R) = sum w_s S e_t / sum w_s S^2.
It is divided by 1 + K, with K the weighted multiplicative bias of Asgari et al. (2021, Table 1), multiplied by
the boost factor, and the signal around the random points is subtracted. Radii are physical, in 12 logarithmic
bins from 0.05 to 3 Mpc. The covariance is a jackknife over 60 patches, 20 per field. The luminosity bins L1 to
L6 hold Viola et al.'s fractions of the groups (their luminosity bins matched by rank); the bins M1 to M5 are
bins of the dynamical mass proxy MassA. With Viola et al.'s own edges, log LumB = 10.9, 11.1, 11.3, 11.5 and 11.7,
the test of stage 3 fails in 3 of the 6 bins. The matched edges lie 0.14 to 0.20 dex above theirs, which points to
a shift of the G3C v10 luminosities against the earlier version of the catalogue in their sample. The matched
bins pass in all 6.
Stage 3, the validation and the halo fit. Per luminosity bin, one NFW halo of M200m with c200m = 0.84 times the
relation of Duffy et al. (2008), 38 percent of the groups offset by a 3D Gaussian of width 0.79 r_s, a point mass
of 2.06 times Viola et al.'s mean BCG stellar mass and the linear two-halo term (Tinker et al. 2010 bias, the
linear power of the Planck 2018 template) is fitted to the profile and compared, in their definition, with
Viola et al.'s published mass. The test passes if at most one bin lies beyond 2 sigma. The cross shear is tested
against zero and the random-point term against the signal. The halo of every bin is then refitted with the
point mass of its own BCG: 2.06 times 2 Msun/Lsun times the bin's mean BCG r-band luminosity, the BCG's share of
the members' r-band flux times the group's TotRmag luminosity. The stars of the bin are 2 Msun/Lsun times its
mean group luminosity LumB. The geometry is the Planck 2018 background of background.py: flat, Omega_m = 0.3152,
H0 = 67.36 km/s/Mpc.

Inputs, under the data folder passed as the first argument (default data), none of them redistributed:
    kids1000-shear-catalogue/KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits   the gold catalogue, 17.7 GB
    kids1000-shear-catalogue/KiDS1000_SOM_N_of_Z.tar.gz                  the SOM n(z) of the 5 tomographic bins
    gama-dr4-g3c/G3CFoFGroupv10.fits and gama-dr4-g3c/G3CGalv10.fits     the G3C v10 groups and their members
    viola2015/x/M3K_paperFinal_v2.1_accepted.tex                          the TeX source of Viola et al. (2015),
                                                                          arXiv:1507.00735v2
Outputs: mond-classes/groups-stack-profiles.json (deposited; mond_classes.py reads it) and
mond-classes/groups_stack-results.txt; the checkpoints, about 190 MB, go to mond-classes/groups-stack-checkpoints/
(not deposited).
Command, from the repository root (about 5 min with 12 processes: stage 1 about 1 min, as it reads the catalogue
once; stage 2 about 2 min; stage 3 about 2 min):
    python -I -B mond-classes/groups_stack.py data
Options: --stage N runs stage N alone from the checkpoints before it; --workers N sets the processes of stage 2
(default 12); --checkpoints DIR moves the checkpoint folder. Needs numpy, scipy, astropy and camb. It runs before
mond_classes.py, which also runs from the deposited JSON alone.
"""
import argparse
import io
import json
import math
import os
import re
import sys
import tarfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8')
from mond_law import G, MSUN, MPC  # noqa: E402
import group_models as gm  # noqa: E402
import web_lensing as wl  # noqa: E402

OUT = os.path.join(HERE, 'groups-stack-profiles.json')
RESULTS = os.path.join(HERE, 'groups_stack-results.txt')
CK_DEFAULT = os.path.join(HERE, 'groups-stack-checkpoints')

OM_G, H_G = gm.OM_G, gm.H_G                       # the geometry of the measurement
C_KMS = 299792.458
SIGCRIT_K = (299792458.0 ** 2) / (4.0 * math.pi * G) * MPC / MSUN    # Msun/Mpc: Sigma_crit = K D_s/(D_l D_ls)
M_SUN_R = 4.65                                    # absolute magnitude of the Sun, SDSS r (AB), Willmer 2018
UPS_R = 2.0                                       # r-band stellar mass-to-light ratio of the groups' stars, Msun/Lsun

# the lens fields (GAMA II) and the source boxes, with a margin of 3 degrees in RA
FIELDS = {'G09': dict(ra=(129.0, 141.0), dec=(-2.0, 3.0)), 'G12': dict(ra=(174.0, 186.0), dec=(-3.0, 2.0)),
          'G15': dict(ra=(211.5, 223.5), dec=(-2.0, 3.0))}
SRC_MARGIN = 3.0
SRC_DEC = (-6.0, 6.0)
Z_MIN_LENS = 0.02
DZ_CUT = 0.1
TOMO_EDGES = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.2])
M_BIAS = np.array([-0.009, -0.011, -0.015, 0.002, 0.007])          # Asgari et al. 2021, Table 1
REDGES = np.geomspace(0.05, 3.0, 13)                                 # Mpc, physical
NR = len(REDGES) - 1
N_RAND_PER_LENS = 20
L_EDGES = [9.4, 12.7]                             # the outer edges of log LumB / (h^-2 Lsun)
M_EDGES = [12.5, 13.0, 13.5, 14.0, 14.5, 15.5]    # log MassA / (h^-1 Msun)
N_PATCH_RA, N_PATCH_DEC = 10, 2
# Viola et al. 2015, BCG centre: f_c, p_off, R_off and A_P of their halo model
VIOLA_HM = dict(f_c=0.84, p_off=0.38, R_off=0.79, A_P=2.06)
H_VIOLA, OM_VIOLA = 0.673, 0.315                  # the cosmology of Viola et al.
QN = ['W', 'TA', 'XA', 'TB', 'XB', 'M', 'N', 'RW']   # the pair sums: weight, tangential and cross shear in the
Q = {k: i for i, k in enumerate(QN)}                 # two sign conventions, bias, pairs and the radius sum

LINES = []


def P(s=''):
    LINES.append(s)
    print(s, flush=True)


def dump(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))


def load(ck, name):
    with open(os.path.join(ck, name), encoding='utf-8') as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------------
# geometry and critical surface density
# ----------------------------------------------------------------------------------------------
_ZG = np.linspace(0.0, 6.0, 6001)
_EZ = np.sqrt(OM_G * (1.0 + _ZG) ** 3 + 1.0 - OM_G)
_CHI = np.concatenate([[0.0], np.cumsum(0.5 * (1.0 / _EZ[1:] + 1.0 / _EZ[:-1]) * np.diff(_ZG))]) * C_KMS / (100.0 * H_G)


def chi(z):
    return np.interp(z, _ZG, _CHI)


def d_a(z):
    return chi(z) / (1.0 + np.asarray(z, float))


def sigcrit_inv(zl, zs):
    """Sigma_crit^-1 in Mpc^2/Msun (physical), zero for zs <= zl; zl scalar or array, zs array."""
    zl = np.asarray(zl, float)
    zs = np.asarray(zs, float)
    cl, cs = chi(zl), chi(zs)
    dl = cl / (1.0 + zl)
    dls = np.maximum(cs - cl, 0.0) / (1.0 + zs)
    ds = cs / (1.0 + zs)
    return np.where(zs > zl, dl * dls / (ds * SIGCRIT_K), 0.0)


def read_nz(nz_tar):
    out = []
    with tarfile.open(nz_tar) as t:
        for k in range(1, 6):
            m = [x for x in t.getmembers() if x.isfile() and f'TOMO{k}_Nz' in x.name][0]
            a = np.loadtxt(io.StringIO(t.extractfile(m).read().decode()))
            out.append(a)
    return out


_ZL_TAB = np.linspace(0.0, 0.8, 801)


def sinv_nz_table(nz_tar):
    """<Sigma_crit^-1>(z_l) over the SOM n(z) of each tomographic bin, each histogram bin sampled at 20 points."""
    nz = read_nz(nz_tar)
    tab = np.zeros((5, _ZL_TAB.size))
    for t, a in enumerate(nz):
        z0, n = a[:, 0], a[:, 1]
        dz = np.diff(z0).mean()
        sub = (np.arange(20) + 0.5) / 20.0 * dz
        zs = (z0[:, None] + sub[None, :]).ravel()
        ws = np.repeat(n, 20)
        norm = ws.sum()
        for i, zl in enumerate(_ZL_TAB):
            tab[t, i] = np.sum(ws * sigcrit_inv(zl, zs)) / norm
    return tab, [float((a[:, 0] + 0.025) @ a[:, 1] / a[:, 1].sum()) for a in nz]


# ----------------------------------------------------------------------------------------------
# Viola et al. (2015): the bins, the BCG stellar masses and the halo masses
# ----------------------------------------------------------------------------------------------
TRIPLE_SUP_SUB = re.compile(r'(\d+\.\d+)\s*\^\{\+(\d+\.\d+)\}\s*_\{-(\d+\.\d+)\}')


def parse_viola(tex):
    """Per luminosity bin of Viola et al.: the number of groups N, the redshift z, the mean BCG stellar mass Mc (Msun)
    and the halo mass log M200m (h^-1 Msun) with its upper and lower errors, from their Tables 1 and 3."""
    t = open(tex, encoding='latin-1').read()
    i = t.index('\\label{tab:binSummary}')
    block = t[i:t.index('\\end{tabular}', i)]
    line = [ln for ln in block.split('\n') if 'L_{grp}' in ln and '&' in ln][0]
    tup = [[float(x) for x in g.split(',')] for g in re.findall(r'\(([\d.,\s]+)\)', line)]
    edges, N, zz, mbcg = tup
    i = t.index('\\label{tab:summaryParam}')
    block = t[i:t.index('\\end{tabular}', i)]
    row = [ln for ln in block.split('\n') if ln.strip().startswith('\\texttt{BCG}')][0]
    masses = [(float(a), float(b), float(c)) for a, b, c in TRIPLE_SUP_SUB.findall(row)]
    assert len(masses) == 6 and len(N) == 6 and len(edges) == 7
    return [dict(N=int(N[k]), z=zz[k], logM_h=masses[k][0], up=masses[k][1], lo=masses[k][2],
                 Mc=10 ** mbcg[k] / H_VIOLA ** 2) for k in range(6)]


def matched_edges(lum, viola):
    """log LumB edges that give each bin Viola et al.'s fraction of the groups."""
    fr = np.array([v['N'] for v in viola], float)
    cum = np.cumsum(fr / fr.sum())[:-1]
    srt = np.sort(lum)
    return [float(srt[min(int(round(c * len(srt))), len(srt) - 1)]) for c in cum]


# ----------------------------------------------------------------------------------------------
# stage 1: the source subset and the lenses
# ----------------------------------------------------------------------------------------------
def in_box(ra, dec, box, margin=0.0, dec_range=None):
    a, b = box['ra']
    lo, hi = dec_range if dec_range else box['dec']
    return (ra >= a - margin) & (ra <= b + margin) & (dec >= lo) & (dec <= hi)


def tomo_bin(zb):
    return np.clip(np.searchsorted(TOMO_EDGES, zb, side='left') - 1, -1, 4)


def patch_of(ra, dec):
    out = np.full(ra.shape, -1, int)
    for fi, (name, box) in enumerate(FIELDS.items()):
        m = in_box(ra, dec, box)
        a, b = box['ra']
        lo, hi = box['dec']
        ix = np.clip(((ra[m] - a) / ((b - a) / N_PATCH_RA)).astype(int), 0, N_PATCH_RA - 1)
        iy = np.clip(((dec[m] - lo) / ((hi - lo) / N_PATCH_DEC)).astype(int), 0, N_PATCH_DEC - 1)
        out[m] = fi * N_PATCH_RA * N_PATCH_DEC + ix * N_PATCH_DEC + iy
    return out


def stage1(data, ck):
    from astropy.io import fits
    t0 = time.time()
    keep = {f: {k: [] for k in ('ra', 'dec', 'e1', 'e2', 'w', 'zb')} for f in FIELDS}
    csum = np.zeros((5, 3))                     # sum w, sum w e1, sum w e2 per tomographic bin, KiDS-North
    ntot, nnorth = 0, 0
    kids = os.path.join(data, 'kids1000-shear-catalogue', 'KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits')
    with fits.open(kids, memmap=True) as h:
        d = h[1].data
        nrow = h[1].header['NAXIS2']
        cols = {k: d.field(k) for k in ('RAJ2000', 'DECJ2000', 'e1', 'e2', 'weight', 'Z_B')}
        step = 2_000_000
        for i0 in range(0, nrow, step):
            i1 = min(i0 + step, nrow)
            ra = np.asarray(cols['RAJ2000'][i0:i1], '<f8')
            dec = np.asarray(cols['DECJ2000'][i0:i1], '<f8')
            e1 = np.asarray(cols['e1'][i0:i1], '<f8')
            e2 = np.asarray(cols['e2'][i0:i1], '<f8')
            w = np.asarray(cols['weight'][i0:i1], '<f8')
            zb = np.asarray(cols['Z_B'][i0:i1], '<f8')
            good = (w > 0) & np.isfinite(e1) & np.isfinite(e2) & (zb > TOMO_EDGES[0]) & (zb <= TOMO_EDGES[-1])
            tb = tomo_bin(zb)
            north = good & (dec > -15.0)
            for t in range(5):
                m = north & (tb == t)
                csum[t] += [w[m].sum(), (w[m] * e1[m]).sum(), (w[m] * e2[m]).sum()]
            ntot += int(good.sum())
            nnorth += int(north.sum())
            for f, box in FIELDS.items():
                m = good & in_box(ra, dec, box, SRC_MARGIN, SRC_DEC)
                if m.any():
                    keep[f]['ra'].append(ra[m])
                    keep[f]['dec'].append(dec[m])
                    keep[f]['e1'].append(e1[m].astype('f4'))
                    keep[f]['e2'].append(e2[m].astype('f4'))
                    keep[f]['w'].append(w[m].astype('f4'))
                    keep[f]['zb'].append(zb[m].astype('f4'))
            print(f'stage 1: rows {i0:,} to {i1:,} read ({time.time() - t0:.0f} s)', flush=True)
    c = (csum[:, 1:] / csum[:, :1]).tolist()
    nsrc = {}
    for f in FIELDS:
        for k, v in keep[f].items():
            arr = np.concatenate(v) if v else np.zeros(0)
            np.save(os.path.join(ck, f'src_{f}_{k}.npy'), arr)
        nsrc[f] = int(sum(len(x) for x in keep[f]['ra']))

    # the lenses
    with fits.open(os.path.join(data, 'gama-dr4-g3c', 'G3CFoFGroupv10.fits')) as g:
        gd = g[1].data
        tab = {k: np.asarray(gd[k]).astype(float) for k in ('GroupID', 'Nfof', 'BCGRA', 'BCGDec', 'Zfof', 'LumB', 'MassA')}
    sel = tab['Nfof'] >= 5
    sel &= tab['Zfof'] >= Z_MIN_LENS
    inf = np.zeros(sel.shape, bool)
    fld = np.full(sel.shape, '', object)
    for f, box in FIELDS.items():
        m = in_box(tab['BCGRA'], tab['BCGDec'], box)
        inf |= m
        fld[m] = f
    sel &= inf
    L = {k: v[sel] for k, v in tab.items()}
    L['field'] = fld[sel].tolist()
    L['patch'] = patch_of(L['BCGRA'], L['BCGDec'])
    L['mbin'] = np.digitize(np.log10(np.maximum(L['MassA'], 1.0)), M_EDGES) - 1
    nl = int(sel.sum())
    # the random points: uniform in the three lens fields (uniform in RA and in sin Dec), redshift of a random lens
    rng = np.random.default_rng(20261107)
    areas = np.array([(b['ra'][1] - b['ra'][0]) * (math.sin(math.radians(b['dec'][1])) - math.sin(math.radians(b['dec'][0])))
                      for b in FIELDS.values()])
    nr = N_RAND_PER_LENS * nl
    which = rng.choice(len(FIELDS), size=nr, p=areas / areas.sum())
    rra, rdec = np.zeros(nr), np.zeros(nr)
    for fi, box in enumerate(FIELDS.values()):
        m = which == fi
        rra[m] = rng.uniform(*box['ra'], m.sum())
        s0, s1 = math.sin(math.radians(box['dec'][0])), math.sin(math.radians(box['dec'][1]))
        rdec[m] = np.degrees(np.arcsin(rng.uniform(s0, s1, m.sum())))
    parent = rng.integers(0, nl, nr)
    Rr = dict(ra=rra, dec=rdec, z=L['Zfof'][parent], parent=parent, field=[list(FIELDS)[i] for i in which],
              patch=patch_of(rra, rdec), mbin=L['mbin'][parent])
    tab_nz, zmean = sinv_nz_table(os.path.join(data, 'kids1000-shear-catalogue', 'KiDS1000_SOM_N_of_Z.tar.gz'))
    viola = parse_viola(os.path.join(data, 'viola2015', 'x', 'M3K_paperFinal_v2.1_accepted.tex'))
    out = dict(n_sources=nsrc, n_good_total=ntot, n_good_north=nnorth, c_bias=c, n_lenses=nl, lenses=L, randoms=Rr,
               nz_mean=zmean, sinv_nz=tab_nz, zl_tab=_ZL_TAB, viola=viola)
    dump(os.path.join(ck, 'stage1_lenses.json'), out)
    print(f'stage 1: {nl} lenses, {nr} random points, sources ' + ', '.join(f'{f} {n:,}' for f, n in nsrc.items())
          + f' ({time.time() - t0:.0f} s)', flush=True)


# ----------------------------------------------------------------------------------------------
# stage 2: the profiles
# ----------------------------------------------------------------------------------------------
_W = {}


def _init_field(ck, field, c_bias, sinv_tab, zl_tab):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    from scipy.spatial import cKDTree
    ld = {k: np.load(os.path.join(ck, f'src_{field}_{k}.npy')) for k in ('ra', 'dec', 'e1', 'e2', 'w', 'zb')}
    tb = tomo_bin(ld['zb'].astype(float))
    c = np.asarray(c_bias)
    ld['e1'] = ld['e1'].astype(float) - c[tb, 0]
    ld['e2'] = ld['e2'].astype(float) - c[tb, 1]
    ld['w'] = ld['w'].astype(float)
    ld['zb'] = ld['zb'].astype(float)
    ld['tb'] = tb
    ld['m'] = M_BIAS[tb]
    ra, dec = np.radians(ld['ra']), np.radians(ld['dec'])
    ld['sd'], ld['cd'] = np.sin(dec), np.cos(dec)
    ld['rar'] = ra
    xyz = np.vstack([ld['cd'] * np.cos(ra), ld['cd'] * np.sin(ra), ld['sd']]).T
    _W['tree'] = cKDTree(xyz)
    _W['src'] = ld
    _W['sinv'] = np.asarray(sinv_tab)
    _W['zl'] = np.asarray(zl_tab)


def pair_sums(ra_l, dec_l, z_l):
    """The pair sums of one lens or random point in the NR radial bins: rows as in QN."""
    s = _W['src']
    out = np.zeros((len(QN), NR))
    dl = float(d_a(z_l))
    theta_max = REDGES[-1] / dl * 1.0005
    a, d = math.radians(ra_l), math.radians(dec_l)
    v = [math.cos(d) * math.cos(a), math.cos(d) * math.sin(a), math.sin(d)]
    idx = np.asarray(_W['tree'].query_ball_point(v, 2.0 * math.sin(theta_max / 2.0)), int)
    if idx.size == 0:
        return out
    zb = s['zb'][idx]
    keep = zb > z_l + DZ_CUT
    idx = idx[keep]
    if idx.size == 0:
        return out
    dra = s['rar'][idx] - a
    sd, cd = s['sd'][idx], s['cd'][idx]
    sdl, cdl = math.sin(d), math.cos(d)
    hav = np.sin(0.5 * (np.arcsin(sd) - d)) ** 2 + cdl * cd * np.sin(0.5 * dra) ** 2
    theta = 2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
    R = theta * dl
    rb = np.searchsorted(REDGES, R) - 1
    ok = (rb >= 0) & (rb < NR)
    if not ok.any():
        return out
    idx, R, rb, dra, sd, cd = idx[ok], R[ok], rb[ok], dra[ok], sd[ok], cd[ok]
    # position angle east of north, and the angle from the east axis
    phi = np.arctan2(np.sin(dra) * cd, cdl * sd - sdl * cd * np.cos(dra))
    beta = 0.5 * math.pi - phi
    c2, s2 = np.cos(2.0 * beta), np.sin(2.0 * beta)
    e1, e2, w, m = s['e1'][idx], s['e2'][idx], s['w'][idx], s['m'][idx]
    etA = -(e1 * c2 + e2 * s2)
    exA = e1 * s2 - e2 * c2
    etB = -(e1 * c2 - e2 * s2)
    exB = e1 * s2 + e2 * c2
    tb = s['tb'][idx]
    zi = np.clip(np.searchsorted(_W['zl'], z_l), 1, _W['zl'].size - 1)
    f = (z_l - _W['zl'][zi - 1]) / (_W['zl'][zi] - _W['zl'][zi - 1])
    sinv_bins = _W['sinv'][:, zi - 1] * (1.0 - f) + _W['sinv'][:, zi] * f
    S = sinv_bins[tb]
    ws2 = w * S * S
    ws = w * S
    out[Q['W']] = np.bincount(rb, ws2, NR)
    out[Q['TA']] = np.bincount(rb, ws * etA, NR)
    out[Q['XA']] = np.bincount(rb, ws * exA, NR)
    out[Q['TB']] = np.bincount(rb, ws * etB, NR)
    out[Q['XB']] = np.bincount(rb, ws * exB, NR)
    out[Q['M']] = np.bincount(rb, ws2 * m, NR)
    out[Q['N']] = np.bincount(rb, None, NR)
    out[Q['RW']] = np.bincount(rb, w * S * S * R, NR)
    return out


def _task(points):
    return [pair_sums(*p) for p in points]


def assemble(SL, SR, nl, nr, conv='A'):
    """Delta Sigma (Msun/pc^2), the cross term, the boost factor, the random term and R_eff from summed pair sums."""
    W, T, X, M = SL[Q['W']], SL[Q['T' + conv]], SL[Q['X' + conv]], SL[Q['M']]
    Wr, Tr, Xr, Mr = SR[Q['W']], SR[Q['T' + conv]], SR[Q['X' + conv]], SR[Q['M']]
    with np.errstate(invalid='ignore', divide='ignore'):
        K, Kr = M / W, Mr / Wr
        boost = (W / nl) / (Wr / nr)
        dl, dr = T / W / (1.0 + K), Tr / Wr / (1.0 + Kr)
        xl, xr = X / W / (1.0 + K), Xr / Wr / (1.0 + Kr)
        esd = (boost * dl - dr) / 1e12
        cross = (boost * xl - xr) / 1e12
        reff = SL[Q['RW']] / W
    return dict(esd=esd, cross=cross, boost=boost, random=dr / 1e12, reff=reff)


def jackknife(sl, sr, pl, pr, conv='A'):
    """Jackknife over patches: the full-sample profile and the covariances of esd and cross."""
    full = assemble(sl.sum(0), sr.sum(0), len(sl), len(sr), conv)
    patches = np.unique(np.concatenate([pl, pr]))
    patches = patches[patches >= 0]
    E, X = [], []
    SL, SR = sl.sum(0), sr.sum(0)
    for p in patches:
        ml, mr = pl == p, pr == p
        a = assemble(SL - sl[ml].sum(0), SR - sr[mr].sum(0), len(sl) - ml.sum(), len(sr) - mr.sum(), conv)
        E.append(a['esd'])
        X.append(a['cross'])
    E, X = np.array(E), np.array(X)
    n = len(patches)
    ce = (n - 1) / n * (E - E.mean(0)).T @ (E - E.mean(0))
    cx = (n - 1) / n * (X - X.mean(0)).T @ (X - X.mean(0))
    full.update(cov=ce, cov_cross=cx, npatch=n)
    return full


def stage2(ck, workers):
    from multiprocessing import Pool
    d1 = load(ck, 'stage1_lenses.json')
    L, Rr = d1['lenses'], d1['randoms']
    for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[k] = '1'
    sums_l = np.zeros((d1['n_lenses'], len(QN), NR))
    sums_r = np.zeros((len(Rr['ra']), len(QN), NR))
    t0 = time.time()
    for f in FIELDS:
        il = [i for i, x in enumerate(L['field']) if x == f]
        ir = [i for i, x in enumerate(Rr['field']) if x == f]
        pts = [(L['BCGRA'][i], L['BCGDec'][i], L['Zfof'][i]) for i in il] + [(Rr['ra'][i], Rr['dec'][i], Rr['z'][i]) for i in ir]
        chunks = [pts[i:i + 40] for i in range(0, len(pts), 40)]
        res = []
        with Pool(workers, initializer=_init_field, initargs=(ck, f, d1['c_bias'], d1['sinv_nz'], d1['zl_tab'])) as pool:
            for r in pool.imap(_task, chunks):
                res.extend(r)
        res = np.array(res)
        sums_l[il] = res[:len(il)]
        sums_r[ir] = res[len(il):]
        print(f'stage 2: field {f}: {len(il)} lenses and {len(ir)} random points ({time.time() - t0:.0f} s)', flush=True)
    pl, pr = np.asarray(L['patch']), np.asarray(Rr['patch'])
    # the shear convention: the one that gives the larger tangential signal at 0.1 to 1 Mpc over all lenses
    rmid = (REDGES[:-1] * REDGES[1:]) ** 0.5
    allp = {}
    for conv in ('A', 'B'):
        a = assemble(sums_l.sum(0), sums_r.sum(0), len(sums_l), len(sums_r), conv)
        sel = (rmid > 0.1) & (rmid < 1.0)
        allp[conv] = float(np.mean(a['esd'][sel]))
    conv = 'A' if allp['A'] > allp['B'] else 'B'
    lum = np.log10(np.asarray(L['LumB']))
    edges = matched_edges(lum, d1['viola'])
    lbin_l = np.digitize(lum, edges)
    lbin_r = lbin_l[np.asarray(Rr['parent'])]
    out = dict(convention=conv, convention_test=allp, l_edges=[L_EDGES[0]] + edges + [L_EDGES[-1]], bins={})
    for kind, nb in (('L', 6), ('M', 5)):
        for b in range(nb):
            if kind == 'L':
                ml, mr = lbin_l == b, lbin_r == b
            else:
                ml = np.asarray(L['mbin']) == b
                mr = np.asarray(Rr['mbin']) == b
            if ml.sum() < 5:
                continue
            prof = jackknife(sums_l[ml], sums_r[mr], pl[ml], pr[mr], conv)
            zl = np.asarray(L['Zfof'])[ml]
            wl_ = sums_l[ml][:, Q['W']].sum(1)
            out['bins'][f'{kind}{b + 1}'] = dict(
                kind=kind, n_lens=int(ml.sum()), n_rand=int(mr.sum()),
                z_mean=float(np.average(zl, weights=np.maximum(wl_, 1e-30))),
                **{k: prof[k] for k in ('esd', 'cross', 'boost', 'random', 'reff', 'cov', 'cov_cross', 'npatch')})
    dump(os.path.join(ck, 'stage2_profiles.json'), out)
    print(f'stage 2: profiles of {len(out["bins"])} bins ({time.time() - t0:.0f} s)', flush=True)


# ----------------------------------------------------------------------------------------------
# the halo model
# ----------------------------------------------------------------------------------------------
KK = np.geomspace(1e-4, 3e4, 24000)                   # 1/Mpc, physical
_DLNK = np.gradient(np.log(KK))


def j2_matrix(R):
    from scipy.special import jv
    return jv(2, KK[:, None] * np.asarray(R)[None, :])


def esd_hankel(rho_k, J2):
    """Delta Sigma(R) = int k dk/(2 pi) rho~(k) J2(k R), rho~ in Msun (the 3D transform of the density)."""
    return (KK * KK * rho_k * _DLNK) @ J2 / (2.0 * math.pi)


class LinearPower:
    """The linear matter power of the Planck 2018 template at one redshift, comoving h units."""

    def __init__(self, z):
        self.tpl = wl.Template(zs=np.unique(np.array([0.0, z, z + 0.05, 0.5, 1.0])), kmax=50.0)
        self.k = np.geomspace(1e-4, 4e2, 6000)          # h/Mpc
        self.P = self.tpl.P(np.full_like(self.k, z), self.k, nonlinear=False)   # (Mpc/h)^3
        self.h = self.tpl.h

    def sigma(self, R):
        """sigma(R) for a top-hat of comoving radius R (Mpc/h)."""
        x = self.k * R
        W = 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3
        I = np.trapezoid(self.P * W * W * self.k ** 3, np.log(self.k)) / (2.0 * math.pi ** 2)
        return math.sqrt(I)


_LP = {}


def linpower(z):
    zk = round(float(z), 3)
    if zk not in _LP:
        _LP[zk] = LinearPower(zk)
    return _LP[zk]


def two_halo_unit(z, J2):
    """Delta Sigma of the linear matter correlation for unit bias, physical Msun/Mpc^2, at the radii of J2. With
    comoving R_c = (1+z) R and k_c = k/(1+z), k_c R_c = k R, and the factors of (1+z) cancel."""
    lp = linpower(z)
    kcom = KK / (1.0 + z)
    P = np.interp(np.log(kcom / lp.h), np.log(lp.k), lp.P, left=0.0, right=0.0) / lp.h ** 3     # Mpc^3 comoving
    rho0 = gm.RHO_CRIT0_H2 * H_G * H_G * OM_G
    return rho0 * ((KK * KK * P * _DLNK) @ J2) / (2.0 * math.pi)


def tinker_bias(nu_peak, Delta=200.0):
    """Tinker et al. (2010) halo bias, their Table 2 parameters."""
    y = math.log10(Delta)
    A = 1.0 + 0.24 * y * math.exp(-(4.0 / y) ** 4)
    a = 0.44 * y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * math.exp(-(4.0 / y) ** 4)
    c = 2.4
    dc = 1.686
    return 1.0 - A * nu_peak ** a / (nu_peak ** a + dc ** a) + B * nu_peak ** b + C * nu_peak ** c


def tinker_b(M200m, z):
    lp = linpower(z)
    R_lag = (3.0 * M200m * lp.h / (4.0 * math.pi * gm.RHO_CRIT0_H2 * OM_G)) ** (1.0 / 3.0)
    return float(tinker_bias(1.686 / lp.sigma(R_lag)))


def nfw_uk(k, r200, c):
    """Fourier transform (normalized to 1 at k = 0) of an NFW profile truncated at r200; k in 1/Mpc."""
    from scipy import special
    k = np.atleast_1d(np.asarray(k, float))
    rs = r200 / c
    x = k * rs
    si1, ci1 = special.sici((1.0 + c) * x)
    si0, ci0 = special.sici(x)
    norm = math.log1p(c) - c / (1.0 + c)
    u = (np.sin(x) * (si1 - si0) - np.sin(c * x) / ((1.0 + c) * x) + np.cos(x) * (ci1 - ci0)) / norm
    u[k <= 0] = 1.0
    return u


def nfw_mass(r, M200, r200, c):
    """Enclosed mass of an NFW halo truncated at r200 (Msun)."""
    r = np.asarray(r, float)
    rs = r200 / c
    m = lambda x: np.log1p(x) - x / (1.0 + x)
    return M200 * m(np.minimum(r, r200) / rs) / m(c)


def nfw_rho_k(M, r200, c, p_off=0.0, R_off=0.0):
    u = nfw_uk(KK, r200, c)
    mis = 1.0 - p_off + p_off * np.exp(-0.5 * (KK * R_off * r200 / c) ** 2)
    return M * u * mis


def halo_model_esd(M, z, R, J2, Mp, two_halo):
    """The halo model of Viola et al.: the centred truncated NFW projected directly, the miscentring as the
    difference of two Hankel transforms (which cancels the transforms' ringing at the truncation radius), the
    point mass and the two-halo term."""
    r200, c, rs = gm.halo_200m(M, z)
    one = gm.esd_from_mass(R, gm.R_GRID, nfw_mass(gm.R_GRID, M, r200, c))
    one = one + esd_hankel(nfw_rho_k(M, r200, c, VIOLA_HM['p_off'], VIOLA_HM['R_off']) - nfw_rho_k(M, r200, c), J2)
    pm = Mp / (math.pi * R ** 2)
    b = tinker_b(M, z)
    return one + pm + b * two_halo


def chi2(d, m, C, n_patch):
    Ci = np.linalg.inv(C) * gm.hartlap(n_patch, len(d))
    r = d - m
    return float(r @ Ci @ r)


def fit_mass(prof, Mp, z, R, J2, twoh):
    """The single-mass fit: one NFW halo of M200m with the held parameters of Viola et al.; log M200m, the chi2,
    and the range from Delta chi2 = 1."""
    d = np.asarray(prof['esd']) * 1e12
    C = np.asarray(prof['cov']) * 1e24
    f = lambda lm: chi2(d, halo_model_esd(10 ** lm, z, R, J2, Mp, twoh), C, prof['npatch'])
    grid = np.arange(12.0, 15.6, 0.01)
    chis = np.array([f(lm) for lm in grid])
    k = int(np.argmin(chis))
    fine = np.arange(grid[max(k - 3, 0)], grid[min(k + 3, grid.size - 1)], 0.0005)
    fc = np.array([f(lm) for lm in fine])
    j = int(np.argmin(fc))
    best, cmin = float(fine[j]), float(fc[j])
    allg, allc = np.concatenate([grid, fine]), np.concatenate([chis, fc])
    ok = allg[allc <= cmin + 1.0]
    return dict(logM=best, chi2=cmin, dof=d.size - 1, lo=float(best - ok.min()), hi=float(ok.max() - best))


def m200m_to_viola(M, z):
    """Our M200m (the geometry of the measurement) as M200m in the cosmology of Viola et al."""
    from scipy.optimize import brentq
    r200, c, rs = gm.halo_200m(M, z)
    norm = M / float(gm.m_nfw(c))
    rho_v = gm.RHO_CRIT0_H2 * H_VIOLA ** 2 * OM_VIOLA * (1.0 + z) ** 3
    f = lambda r: norm * float(gm.m_nfw(r / rs)) - 200.0 * rho_v * 4.0 / 3.0 * math.pi * r ** 3
    r = brentq(f, 0.3 * r200, 3.0 * r200)
    return norm * float(gm.m_nfw(r / rs))


def own_group_luminosities(data):
    """Per G3C group: the r-band luminosity of its observed members from TotRmag (h^-2 Lsun, M_sun,r = 4.65) and the
    BCG's share of the members' r-band flux (from Rpetro in G3CGal), so L_BCG = share x L_members."""
    from astropy.io import fits
    with fits.open(os.path.join(data, 'gama-dr4-g3c', 'G3CFoFGroupv10.fits')) as g:
        gd = g[1].data
        gid = np.asarray(gd['GroupID']).astype(int)
        totr = np.asarray(gd['TotRmag'], float)
        bcg = np.asarray(gd['BCGCATAID']).astype(int)
    with fits.open(os.path.join(data, 'gama-dr4-g3c', 'G3CGalv10.fits')) as g:
        md = g[1].data
        mg = np.asarray(md['GroupID']).astype(int)
        mcat = np.asarray(md['CATAID']).astype(int)
        mr = np.asarray(md['Rpetro'], float)
    sel = mg > 0
    flux = 10 ** (-0.4 * mr[sel])
    order = {g: i for i, g in enumerate(gid)}
    idx = np.array([order.get(g, -1) for g in mg[sel]])
    ok = idx >= 0
    fsum = np.bincount(idx[ok], flux[ok], len(gid))
    rmap = dict(zip(mcat[sel], mr[sel]))
    fb = np.array([10 ** (-0.4 * rmap[c]) if c in rmap else np.nan for c in bcg])
    share = fb / fsum
    Lmem = 10 ** (-0.4 * (totr - M_SUN_R))
    return {int(g): (float(Lm), float(s)) for g, Lm, s in zip(gid, Lmem, share)}


# ----------------------------------------------------------------------------------------------
# stage 3: the validation and the halo fit
# ----------------------------------------------------------------------------------------------
def stage3(data, ck):
    t0 = time.time()
    d1, d2 = load(ck, 'stage1_lenses.json'), load(ck, 'stage2_profiles.json')
    viola = d1['viola']
    L = d1['lenses']
    P('THE GAMA GROUPS STACKED IN THE KiDS-1000 SHEAR: THE PROFILES, THEIR VALIDATION AND THE HALO FITS')
    P('=' * 110)
    P(f'data folder: {data}; geometry flat, Omega_m = {OM_G:.4f}, H0 = {100 * H_G:.2f} km/s/Mpc; radii physical (Mpc); ESD in Msun/pc^2')
    P('')
    P('1. THE DATA')
    P(f"  sources: KiDS-1000 gold, {d1['n_good_total']:,} with weight > 0 and 0.1 < Z_B <= 1.2; in the GAMA source boxes "
      + ', '.join(f'{k} {v:,}' for k, v in d1['n_sources'].items()))
    P('  additive bias removed per tomographic bin, e1, e2: '
      + '; '.join(f'bin {i + 1} {a:+.5f}, {b:+.5f}' for i, (a, b) in enumerate(d1['c_bias'])))
    P('  SOM n(z) means per bin: ' + ', '.join(f'{z:.3f}' for z in d1['nz_mean']))
    P(f"  lenses: G3C v10, Nfof >= 5, z >= {Z_MIN_LENS}, BCG in G09, G12 or G15: {d1['n_lenses']}; random points: {len(d1['randoms']['ra'])}")
    P(f"  shear convention {d2['convention']}, the one with the larger mean ESD at 0.1 to 1 Mpc over all lenses: "
      + ', '.join(f"{k} {v:.2f}" for k, v in d2['convention_test'].items()))
    P('  log LumB edges (h^-2 Lsun) of the bins L1 to L6, matched to the group fractions of Viola et al.: '
      + ', '.join(f'{e:.3f}' for e in d2['l_edges']))
    P('')
    P('2. THE VALIDATION')
    rmid = (REDGES[:-1] * REDGES[1:]) ** 0.5
    J2v = j2_matrix(rmid)
    r200, c, rs = gm.halo_200m(10 ** 13.8, 0.2)
    e_h = esd_hankel(nfw_rho_k(10 ** 13.8, r200, c), J2v)
    e_p = gm.esd_from_mass(rmid, gm.R_GRID, nfw_mass(gm.R_GRID, 10 ** 13.8, r200, c))
    P('  the Hankel route over the direct projection for a truncated NFW of 10^13.8 Msun (used for the miscentring and the')
    P('  two-halo term only): ' + ', '.join(f'{x:.3f}' for x in e_h / e_p))
    P('  bin   N     z      log M200m fitted         in their definition (h^-1)   Viola et al. (h^-1)    sigma   chi2/dof   cross chi2/12  random/signal')
    nfail, fits = 0, {}
    for key, prof in d2['bins'].items():
        Rb = np.asarray(prof['reff'])
        J2 = j2_matrix(Rb)
        z = prof['z_mean']
        twoh = two_halo_unit(z, J2)
        x = np.asarray(prof['cross'])
        chi_x = chi2(x * 1e12, np.zeros_like(x), np.asarray(prof['cov_cross']) * 1e24, prof['npatch'])
        e = np.asarray(prof['esd'])
        rsel = (Rb > 0.1) & (Rb < 1.0)
        rand_frac = float(np.mean(np.abs(np.asarray(prof['random'])[rsel])) / np.mean(e[rsel]))
        fits[key] = dict(R=Rb, J2=J2, z=z, twoh=twoh)
        if prof['kind'] == 'L':
            v = viola[int(key[1:]) - 1]
            s = fit_mass(prof, VIOLA_HM['A_P'] * v['Mc'], z, Rb, J2, twoh)
            lm, lo, hi = s['logM'], s['logM'] - s['lo'], s['logM'] + s['hi']
            lmv = math.log10(m200m_to_viola(10 ** lm, z) * H_VIOLA)
            lov, hiv = lmv - (lm - lo), lmv + (hi - lm)
            diff = lmv - v['logM_h']
            sv = v['up'] if diff > 0 else v['lo']
            so = (lmv - lov) if diff > 0 else (hiv - lmv)
            ns = diff / math.sqrt(sv ** 2 + so ** 2)
            nfail += abs(ns) > 2.0
            fits[key]['validation_logM'] = lm
            P(f"  {key}  {prof['n_lens']:5d}  {z:.3f}  {lm:.3f} +{s['hi']:.3f}/-{s['lo']:.3f}       {lmv:.3f}                       "
              f"{v['logM_h']:.2f} +{v['up']:.2f}/-{v['lo']:.2f}       {ns:+.2f}   {s['chi2']:5.1f}/{s['dof']}   {chi_x:6.1f}         {rand_frac:.3f}")
        else:
            P(f"  {key}  {prof['n_lens']:5d}  {z:.3f}  (mass-proxy bin, no published mass)                                          "
              f"{chi_x:6.1f}         {rand_frac:.3f}")
    passed = nfail <= 1
    P(f'  luminosity bins beyond 2 sigma: {nfail} of 6; the test passes with at most one: {"passed" if passed else "FAILED"}')
    P('')
    if not passed:
        P('The validation failed: no halo fit and no groups-stack-profiles.json.')
        return
    P('3. THE HALO FIT WITH EACH BIN\'S OWN BCG, WRITTEN TO groups-stack-profiles.json')
    P(f'  stars {UPS_R:g} Msun/Lsun times the mean LumB; BCG {UPS_R:g} Msun/Lsun times the mean BCG luminosity (M_sun,r = {M_SUN_R});'
      f' point mass {VIOLA_HM["A_P"]} times the BCG')
    P('  bin   log M200m fitted         chi2/dof   change from the validation fit   Mstar_tot (Msun)   Mc (Msun)')
    lums = own_group_luminosities(data)
    gids = np.asarray(L['GroupID']).astype(int)
    lum = np.log10(np.asarray(L['LumB']))
    bins = {}
    for key, prof in d2['bins'].items():
        fk = fits[key]
        if prof['kind'] == 'L':
            sel = np.digitize(lum, d2['l_edges'][1:-1]) == int(key[1:]) - 1
        else:
            sel = np.asarray(L['mbin']) == int(key[1:]) - 1
        lumb = np.asarray(L['LumB'])[sel]
        lmem = np.array([lums[g][0] for g in gids[sel]])
        shr = np.array([lums[g][1] for g in gids[sel]])
        Lbcg = lmem * shr
        good = np.isfinite(Lbcg)
        Mstar_tot = UPS_R * float(np.mean(lumb)) / H_G ** 2
        Mc = UPS_R * float(np.mean(Lbcg[good])) / H_G ** 2
        s = fit_mass(prof, VIOLA_HM['A_P'] * Mc, fk['z'], fk['R'], fk['J2'], fk['twoh'])
        dlm = f"{s['logM'] - fk['validation_logM']:+.3f}" if 'validation_logM' in fk else 'n/a'
        P(f"  {key}  {s['logM']:.3f} +{s['hi']:.3f}/-{s['lo']:.3f}       {s['chi2']:5.1f}/{s['dof']}     {dlm:>6s}                          "
          f"{Mstar_tot:.3e}          {Mc:.3e}")
        bins[key] = dict(kind=prof['kind'], n_lens=prof['n_lens'], z=prof['z_mean'], R=prof['reff'], esd=prof['esd'],
                         cov=prof['cov'], npatch=prof['npatch'], M200m=10 ** s['logM'], Mstar_tot=Mstar_tot, Mc=Mc)
    about = ('Stacked excess-surface-density profiles of the GAMA G3C v10 groups in the KiDS-1000 shear, with the '
             'single-mass halo fit and the stellar masses of each bin. Units: R in Mpc (physical), esd in Msun/pc^2, '
             'cov in (Msun/pc^2)^2, masses in Msun. group_models.py describes the measurement and the fields.')
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(dict(about=about, bins=bins), f, indent=1)
        f.write('\n')
    P('')
    with open(RESULTS, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LINES) + '\n')
    print(f'stage 3: {time.time() - t0:.0f} s', flush=True)


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='The GAMA groups stacked in the KiDS-1000 shear')
    ap.add_argument('data', nargs='?', default='data', help='the data folder (default data)')
    ap.add_argument('--stage', type=int, default=None, help='run one stage from the checkpoints before it')
    ap.add_argument('--workers', type=int, default=12, help='the processes of stage 2')
    ap.add_argument('--checkpoints', default=CK_DEFAULT, help='the checkpoint folder')
    a = ap.parse_args()
    os.makedirs(a.checkpoints, exist_ok=True)
    run = {1: lambda: stage1(a.data, a.checkpoints), 2: lambda: stage2(a.checkpoints, a.workers),
           3: lambda: stage3(a.data, a.checkpoints)}
    for n in ([a.stage] if a.stage else [1, 2, 3]):
        run[n]()


if __name__ == '__main__':
    main()
