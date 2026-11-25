"""Population-synthesis ingredients for the stellar mass-to-light ratio Y_* of the 133 rich-kinematics clusters.

Every number below is quoted from the source named:
  KM09  Kruijssen & Mieske 2009, A&A 500, 785 (arXiv:0903.4683): Table 1 canonical (Kroupa IMF, SSP) M/L_V
        against [Fe/H]; Eq. 5 (Q against t/t_dis); Eq. 15-16 and Table 5 (MF slope alpha against Q).
  D20   Dalgleish et al. 2020, MNRAS 492, 3859 (arXiv:2001.01810), Fig. 10: FSPS-MIST Kroupa SSP at 12.59 Gyr,
        and the version adjusted for the age-metallicity relation of Kruijssen et al. 2019. Values ABOVE
        [Fe/H] = -0.76 are read off the figure by eye (rendered at 300 dpi), so they carry about +-0.1.
  B17   Bianchini et al. 2017, MNRAS 469, 4359 (arXiv:1705.02310), Eq. 3: M/L = a n_rel^-1 + b, with
        a = 3.25, b = 1.35 (11 Gyr, [Fe/H] = -1.3) and a = 4.97, b = 2.01 (11 Gyr, solar); n_rel = T_age / T_rc.
  BSH20 Baumgardt, Sollima & Hilker 2020, PASA 37, e046 (arXiv:2009.09611): M/L_V rises by about 0.45 (observed)
        or 0.3 (PARSEC) from 10 to 13.5 Gyr; mean M/L_V 1.83.
  H10   Harris 1996, 2010 edition (mwgc.dat): [Fe/H] (Part II) and lg(tc), the core relaxation time (Part III).
  BH18  Baumgardt & Hilker 2018, MNRAS 478, 1520: the Kroupa-equivalent single power-law slope over 0.2-0.8 Msun is
        alpha = -1.5 in the convention N(m) ~ m^alpha used by the catalog's MF column.

A module, with no output of its own, imported by globular-clusters/gc_radial_trends.py. It reads the Harris catalog,
data/harris-mwgc-2010-2026-11-15/mwgc.dat.txt, and the globular-cluster database's data/baumgardt-gc/combined_table.txt,
both fetched by the reader (not redistributed).
"""
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HARRIS = os.path.normpath(os.path.join(HERE, "..", "data", "harris-mwgc-2010-2026-11-15", "mwgc.dat.txt"))
CATALOG = os.path.normpath(os.path.join(HERE, "..", "data", "baumgardt-gc", "combined_table.txt"))

# ------------------------------------------------------------------ canonical SSP M/L_V against [Fe/H]
# KM09 Table 1, (M/L_V)_can for the 24 clusters, Kroupa IMF (verified in the text extraction).
KM09_CAN = [(-2.28, 1.99), (-2.22, 1.99), (-2.06, 2.00), (-1.94, 2.01), (-1.83, 2.03), (-1.81, 2.03),
            (-1.75, 2.04), (-1.64, 2.06), (-1.62, 2.06), (-1.57, 2.08), (-1.54, 2.08), (-1.52, 2.09),
            (-1.48, 2.10), (-1.29, 2.18), (-1.27, 2.19), (-1.24, 2.20), (-1.22, 2.21), (-1.20, 2.22),
            (-1.04, 2.34), (-1.01, 2.37), (-0.95, 2.42), (-0.76, 2.68)]
# D20 Fig. 10, blue curve (FSPS MIST, Kroupa, 12.59 Gyr), read off the figure: metal-rich extension only.
D20_KROUPA_HI = [(-0.49, 3.13), (-0.25, 3.65), (0.00, 4.54)]
# D20 Fig. 10, orange curve (the same with the Kruijssen et al. 2019 age-metallicity relation), read off.
D20_AGEADJ = [(-2.46, 2.12), (-2.02, 2.01), (-1.51, 2.20), (-1.23, 2.20), (-1.01, 2.43), (-0.76, 2.54),
              (-0.49, 2.99), (-0.25, 3.50), (-0.10, 3.69)]


def _interp_log(knots, feh):
    x = np.array([k[0] for k in knots])
    y = np.log10([k[1] for k in knots])
    return float(np.interp(feh, x, y))           # flat beyond the ends


def ssp_kroupa(feh):
    """log10 canonical M/L_V: KM09 Table 1 up to -0.76, D20 Fig. 10 (figure-read) above."""
    return _interp_log(KM09_CAN + D20_KROUPA_HI, feh)


def ssp_ageadj(feh):
    """log10 M/L_V of D20's age-adjusted curve (figure-read throughout)."""
    return _interp_log(D20_AGEADJ, feh)


# ------------------------------------------------------------------ dynamical depletion factor Q
def q_from_tfrac(f):
    """KM09 Eq. 5: ratio of predicted to canonical M/L_V against the elapsed fraction of the total lifetime."""
    if f < 0.2:
        return 1.0
    if f < 0.7:
        return 1.142 - 0.71 * f
    return max(1.471 - 1.18 * f, 0.05)


KM09_A = (-4.31, 17.76, -22.10, 13.17, -3.02)     # Table 5, a_n (Q <= 0.645)
KM09_B = (-17.04, 74.41, -116.76, 84.13, -23.06)  # Table 5, b_n (0.645 < Q < 1)


def alpha_of_q(q):
    """KM09 Eq. 16: low-mass MF slope (dN/dm ~ m^-alpha, 0.3-0.8 Msun) against Q."""
    c = KM09_A if q <= 0.645 else KM09_B
    return sum(cn * q ** n for n, cn in enumerate(c))


_QG = np.linspace(0.40, 0.9999, 6000)
_AG = np.array([alpha_of_q(q) for q in _QG])


def q_from_alpha_km(a_km):
    """Invert Eq. 16 on its monotonic branch. alpha >= 1.68 (the Q = 1 value) gives Q = 1."""
    if a_km >= 1.68:
        return 1.0
    if a_km <= _AG.min():
        return float(_QG[np.argmin(_AG)])
    i = np.argsort(_AG)
    return float(np.interp(a_km, _AG[i], _QG[i]))


DELTA_ALPHA = 1.74 - 1.5   # KM09 canonical 1.74 (0.3-0.8 Msun) against BH18's Kroupa-equivalent -1.5 (0.2-0.8)


def q_from_catalog_mf(alpha_cat):
    """The catalog's slope (N ~ m^alpha) mapped to KM09's convention by alpha_KM = 0.24 - alpha_cat."""
    return q_from_alpha_km(DELTA_ALPHA - alpha_cat)


# ------------------------------------------------------------------ remnant retention against relaxation state
B17_A_OVER_B = 3.25 / 1.35     # 11 Gyr, [Fe/H] = -1.3; the solar models give 4.97 / 2.01 = 2.47


def g_bianchini(n_rel):
    """B17 Eq. 3 normalised to the fully relaxed value: 1 + (a/b) / n_rel."""
    return 1.0 + B17_A_OVER_B / n_rel


# ------------------------------------------------------------------ Harris 2010
def harris_tables():
    lines = open(HARRIS, encoding="latin-1").read().splitlines()
    i2 = next(i for i, l in enumerate(lines) if l.strip().startswith("ID") and "[Fe/H]" in l)
    i3 = next(i for i, l in enumerate(lines) if l.strip().startswith("ID") and "lg(tc)" in l)
    hdr3 = lines[i3]
    ptc, pth = hdr3.index("lg(tc)"), hdr3.index("lg(th)")
    feh, ltc = {}, {}
    for l in lines[i2 + 1:]:
        if l.startswith("___") or "Part III" in l:
            break
        if not l.strip():
            continue
        idn = l[:12].strip()
        tok = l[12:].split()
        if tok and re.match(r"^-?\d+\.\d+$", tok[0]):
            feh[idn] = float(tok[0])
    for l in lines[i3 + 1:]:
        if l.startswith("___"):
            break
        if not l.strip():
            continue
        idn = l[:12].strip()
        s = l[ptc - 1:pth - 1].strip()
        if re.match(r"^-?\d+\.\d+$", s):
            ltc[idn] = float(s)
    return feh, ltc


SPECIAL = {"Ter": "Terzan", "Djor": "Djorg", "2MASS-GC01": "2MS-GC01", "2MASS-GC02": "2MS-GC02",
           "ESO_452-SC11": "1636-283", "ESO_280-SC06": "ESO-SC06", "Lynga_7": "Lynga 7"}


def harris_id(bname):
    if bname in SPECIAL:
        return SPECIAL[bname]
    parts = bname.split("_")
    if parts[0] in SPECIAL:
        parts[0] = SPECIAL[parts[0]]
    return " ".join(parts)


def catalog_extra():
    """T_Diss (remaining lifetime, Gyr), lg(Mini), and the MF mass range, per cluster, from the snapshot."""
    raw = open(CATALOG, encoding="utf-8").read().splitlines()
    hdr = raw[0].lstrip("#").split()
    col = {n: i for i, n in enumerate(hdr)}
    out = {}
    for ln in raw:
        if ln.startswith("#") or not ln.strip():
            continue
        c = ln.split()
        out[c[0]] = dict(tdiss=float(c[col["T_Diss"]]), lmini=float(c[col["lg(Mini)"]]),
                         mlow=float(c[col["M_Low"]]), mhigh=float(c[col["M_High"]]),
                         dmf=float(c[col["Delta_MF"]]))
    return out


if __name__ == "__main__":
    for q in (0.40, 0.45, 0.5, 0.6, 0.645, 0.7, 0.8, 0.9, 0.99):
        print("Q %.3f  alpha_KM %.3f" % (q, alpha_of_q(q)))
    print("min alpha on grid %.3f at Q %.3f" % (_AG.min(), _QG[np.argmin(_AG)]))
    for a in (-1.9, -1.5, -1.2, -1.0, -0.7, -0.5, -0.3, 0.0, 0.3, 0.9):
        print("catalog alpha %+.2f -> Q %.3f" % (a, q_from_catalog_mf(a)))
    for f in (-2.3, -2.0, -1.5, -1.0, -0.76, -0.5, -0.25, 0.0):
        print("[Fe/H] %+.2f  Kroupa M/L %.2f  age-adjusted %.2f" % (f, 10 ** ssp_kroupa(f), 10 ** ssp_ageadj(f)))
    feh, ltc = harris_tables()
    print(len(feh), "Harris [Fe/H];", len(ltc), "lg(tc)")
