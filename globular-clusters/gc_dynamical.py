"""The globular-cluster test of Paper II section 6.3 on dynamical data: Newton, which the scale-selective coupling
gives a cluster smaller than the catchment length, against ambient rescaling, the isotropic form of the
ambient-stretch coupling.

Under the scale-selective coupling a cluster smaller than the catchment length is Newtonian, and its
Newtonian-inferred mass-to-light ratio is the stellar-population value. Under ambient rescaling the ambient
Galactic stretch rescales its effective gravitational constant, and its Newtonian-inferred ratio is the population
value times

    P = nu(Xbar(l_c)) / nu_sun                                                                   (1)

where Xbar is the total field at the catchment length l_c from the cluster's own catalog mass and a Milky Way point
mass of 6.5e10 solar masses, on MOND's simple function nu(X) = 1 + 1/X at a0 = 1.2e-10 m/s^2, and nu_sun is the
same quantity at 8.2 kpc. Single values are quoted at the floor of the catchment length, l_c = 58.254 pc at
H0 = 67.4 km/s/Mpc (Paper II section 4.1), and ranges over the window of 58.254 to 200 pc. Two more lengths inside
Paper I's window of 30 to 200 pc, 49 and 90 pc, are run as well. P rises with l_c for every cluster, so the floor
is the conservative end for rescaling.

Parts
  A. The remote clusters (internal and Galactic fields below a0) with tabulated velocity dispersions: each binned
     dispersion of the database against sqrt(k) times the database's N-body curve, with a Barlow variable-width
     Gaussian likelihood and a log-normal population prior on k. Newton predicts k = Y_N / Y_cat, rescaling
     k = Y_LC P / Y_cat. Population values: anchored on the baseline of high-acceleration clusters, the stellar
     value 1.8, fitted to the group, and each faint cluster's own star-count ratio. Unresolved binaries enter as a
     stated systematic, a line-of-sight dispersion added in quadrature with a uniform prior on [0, cap].
  B. NGC 2419, whose profile is Newtonian-shaped under both accounts, since rescaling multiplies the whole cluster
     by one factor. Its normalization is tested with a per-star Jeans fit.
  C. The clusters whose catalog masses rest on rich kinematics (N_PM > 0 or N_RV >= 50, with a nonzero quoted
     error): the correlation of their dynamical mass-to-light ratios with P, which criterion F-II.4 of Paper II
     names, with errors raised to the sampling floor sqrt(2/N). NGC 2419 enters with its Jeans ratio from Part B.
     P tracks galactocentric radius almost exactly, so the sensitivity runs let the stellar ratio itself vary with
     radius (a free or bounded power law, a cubic), add population covariates, and change the definition of rich
     kinematics.
  D. The clusters that carry no dynamical evidence and are not counted.

Supports Paper II section 6.3, Figure 3 and Tables 2 and 4: P varies by factors of 3.2 to 6.2 across the 133
clusters with rich kinematics; slope -0.20 +- 0.08 against +1; Delta lnL of 60 to 72 over the window at a fitted
stellar value of 1.90, and 11 to 18 beyond 8 kpc; 13 to 15 with a free radial power law; the fivefold fall of the
stellar ratio that rescaling needs; median 694 velocities and proper motions per cluster; the 21 sparse remote
clusters, of which five show dispersions 2.0 to 3.7 times their star-count models, and their lean of 1.2 to 3.6
toward rescaling that binaries of up to 1 km/s reverse; NGC 2419's Jeans ratio of 2.15 and the stellar ratio of
0.5 to 1.4 that rescaling would need; and the lines of Figure 3 (2.16 for 25 inner clusters, their median P of
0.69, 3.15 P, 1.8 P, 1.90 and 2.26 P). The database's dispersion table lists Sagittarius II as "Sgr 2".

Inputs (fetched by the reader, not redistributed): data/baumgardt-gc/combined_table.txt and
data/baumgardt-gc/pages/ (veldis.html, rvfit/*.pdf, rv/ngc2419.txt). The figure reader and the dispersion-table
parser are imported from globular-clusters/figread.py. NGC 2419's photometry is typed in from Ibata et al. 2011,
Table 2.
Outputs: globular-clusters/gc_dynamical-results.txt (deposited); globular-clusters/gc_dynamical-figdata.tsv, the
data of Paper II's Figure 3 and the input of mond-classes/mond_gc.py; and globular-clusters/gc_dynamical-rich-sample.tsv,
the input of globular-clusters/gc_radial_trends.py. The last two copy values of the database and are not deposited.
Run from the repository root (three to six minutes; needs numpy):
    python globular-clusters/gc_dynamical.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
TABLE = os.path.join(ROOT, "data", "baumgardt-gc", "combined_table.txt")
INP = os.path.join(ROOT, "data", "baumgardt-gc", "pages")
FLOOR = 58.254                     # pc, the floor of the catchment length at H0 = 67.4 km/s/Mpc (Paper II section 4.1)
OUT = os.path.join(HERE, "gc_dynamical-results.txt")
FIGDATA = os.path.join(HERE, "gc_dynamical-figdata.tsv")
RICH_SAMPLE = os.path.join(HERE, "gc_dynamical-rich-sample.tsv")
for p in (TABLE, INP):
    if not os.path.exists(p):
        sys.exit("missing input: %s (fetch the globular-cluster database into data/baumgardt-gc/)" % p)
sys.path.insert(0, HERE)
import figread  # noqa: E402  (the reader of the database's figures and of its dispersion table)

LN10 = math.log(10.0)
Lout = []


def out(s=""):
    Lout.append(s)
    print(s, flush=True)


# =============================================================================================
# 0. Conventions
# =============================================================================================
# Eq. (1): the cluster's own field at the catchment length and the Milky Way's point-mass field, on MOND's simple
# function nu = 1 + 1/X of the total field X at a0 = 1.2e-10 m/s^2 (Paper II section 6.3)
G = 6.674e-11
A0 = 1.2e-10
MSUN = 1.989e30
PC = 3.0857e16
MB = 6.5e10 * MSUN


def amplify(y):
    return 0.5 * (y + math.sqrt(y * y + 4.0 * y))


def nu_simple(X):
    return 1.0 + 1.0 / X


YSUN = G * MB / ((8.2e3 * PC) ** 2) / A0
NUSUN = nu_simple(amplify(YSUN))


def P_lc(mass, rgc, lc):
    """Eq. (1): own catalog mass at the catchment length plus the Milky Way point mass at R_GC."""
    yamb = G * MB / ((rgc * 1e3 * PC) ** 2) / A0
    yown = G * mass * MSUN / ((lc * PC) ** 2) / A0
    return nu_simple(amplify(yown + yamb)) / NUSUN


LCS = tuple(sorted({49.0, FLOOR, 90.0, 100.0, 200.0}))   # 49 pc, the floor, 90 pc, 100 pc and the ceiling
LC49 = 49.0                        # a length below the floor, inside Paper I's window of 30 to 200 pc
LC0 = FLOOR                        # the floor of the window: single values are quoted here
WIN = (FLOOR, 100.0, 200.0)        # ranges are quoted over the window

# the flat-curve field of a sensitivity run
VF = 230e3


def P_lc_fieldA(mass, rgc, lc):
    """Sensitivity: Eq. (1) with a flat-curve field A (total field V^2/R, R floored at 3 kpc) in place of
    the point mass. The ambient total field X_A is carried as its Newtonian equivalent X_A^2/(1 + X_A) under
    nu = 1 + 1/X, so that amplify() returns X_A when the cluster's own field is zero."""
    def yA(R):
        XA = VF ** 2 / (max(R, 3.0) * 1e3 * PC) / A0
        return XA * XA / (1.0 + XA)
    yown = G * mass * MSUN / ((lc * PC) ** 2) / A0
    return nu_simple(amplify(yown + yA(rgc))) / nu_simple(amplify(yA(8.2)))


# ---------------------------------------------------------------- catalog
raw = open(TABLE, encoding="utf-8").read().splitlines()
hdr = raw[0].lstrip("#").split()
col = {n: i for i, n in enumerate(hdr)}
for need in ("Cluster", "R_GC", "N_RV", "N_PM", "Mass", "DM", "M/L_V", "DM/L", "rh,m", "lg(Trh)", "MF"):
    if need not in col:
        sys.exit("catalog header lacks %s" % need)
rows = []
for ln in raw:
    if ln.startswith("#") or not ln.strip():
        continue
    c = ln.split()
    r = dict(name=c[0], rgc=float(c[col["R_GC"]]), nrv=int(c[col["N_RV"]]), npm=int(c[col["N_PM"]]),
             mass=float(c[col["Mass"]]), dmass=float(c[col["DM"]]), ml=float(c[col["M/L_V"]]),
             dml=float(c[col["DM/L"]]), rh=float(c[col["rh,m"]]), trh=float(c[col["lg(Trh)"]]),
             mf=float(c[col["MF"]]))
    rows.append(r)
good = [r for r in rows if r["mass"] > 0 and r["ml"] > 0 and r["rh"] > 0]
for r in good:
    r["L"] = r["mass"] / r["ml"]
    r["xe"] = VF ** 2 / (max(r["rgc"], 3.0) * 1e3 * PC) / A0
    r["gN0"] = G * 0.5 * r["mass"] * MSUN / (r["rh"] * PC) ** 2 / A0
    n = r["nrv"] + 2 * r["npm"] if r["npm"] > 0 else r["nrv"]
    r["floor"] = math.sqrt(2.0 / n) if n > 0 else float("inf")
    r["P"] = {lc: P_lc(r["mass"], r["rgc"], lc) for lc in LCS}
    r["PA"] = P_lc_fieldA(r["mass"], r["rgc"], LC0)
byname = {r["name"]: r for r in good}

# the baseline: the high-acceleration clusters (own field >= 10 a0 and external field >= 2 a0), log-normal fit
# on a grid of 1201 values of the scatter
base = [r for r in good if r["dml"] > 0 and r["gN0"] >= 10 and r["xe"] >= 2]
lb = np.log10([r["ml"] for r in base])
eb = np.array([r["dml"] / r["ml"] / LN10 for r in base])
best = None
for s in np.linspace(0, 0.6, 1201):
    w = 1 / (s * s + eb * eb)
    m = (w * lb).sum() / w.sum()
    ll = -0.5 * ((lb - m) ** 2 * w).sum() + 0.5 * np.log(w).sum()
    if best is None or ll > best[0]:
        best = (ll, m, s)
YBASE, SBASE = 10 ** best[1], best[2]
PBASE = {lc: float(np.median([r["P"][lc] for r in base])) for lc in LCS}
YLC = {lc: YBASE / PBASE[lc] for lc in LCS}           # the population value under rescaling, anchored
PBASE_A = float(np.median([r["PA"] for r in base]))
YLC_A = YBASE / PBASE_A
YPAPER = 1.8                                          # the stellar value of the dotted line of Paper II's Figure 3(a)

remote = sorted([r for r in good if r["gN0"] < 1 and r["xe"] < 1], key=lambda r: r["rgc"])
remote_names = [r["name"] for r in remote]

out("THE GLOBULAR-CLUSTER TEST ON DYNAMICAL DATA: NEWTON AGAINST AMBIENT RESCALING")
out("=" * 108)
out("Catalog: data/baumgardt-gc/combined_table.txt (%d rows, %d usable). Dispersions and N-body curves:" % (len(rows), len(good)))
out("data/baumgardt-gc/pages/ (fetched 2026-10-10). Delta lnL = lnL(Newton) - lnL(ambient rescaling);")
out("positive favors Newton, that is the scale-selective coupling. Ambient rescaling is the isotropic form of the")
out("ambient-stretch coupling of Paper II section 6.3.")
out()
out("0. CONVENTIONS")
out("-" * 108)
out("Eq. (1): y_sun = %.3f, nu_sun = %.4f at the Sun's 8.2 kpc." % (YSUN, NUSUN))
out("Baseline of high-acceleration clusters: N = %d, Y_base = %.3f, s_base = %.3f dex." % (len(base), YBASE, SBASE))
out("Median P over the baseline: %s. The baseline's inner clusters sit at P of %.2f to %.2f (49 pc), because" % (
    ", ".join("%.4f at %g pc" % (PBASE[lc], lc) for lc in LCS), min(r["P"][49.0] for r in base), max(r["P"][49.0] for r in base)))
out("their ambient field exceeds the solar one. Anchored on the baseline, the population value under rescaling")
out("is Y_LC = Y_base / median P = %s." % ", ".join("%.3f at %g pc" % (YLC[lc], lc) for lc in LCS))
out("A second anchoring takes the stellar value 1.8 for both accounts, as Paper II's Figure 3(a) draws it: Newton")
out("flat at 1.8, rescaling 1.8 P.")
out("Remote set (internal and Galactic fields below a0): %d clusters." % len(remote))
hiP = [r for r in remote if r["P"][49.0] > 3.5]
richP = [r for r in good if r["dml"] > 0 and (r["npm"] > 0 or r["nrv"] >= 50)]
out("Range of the boost P at each catchment length, for the %d remote clusters with P(49 pc) > 3.5 (%s)" % (
    len(hiP), ", ".join(r["name"].replace("_", " ") for r in hiP)))
out("and across the %d clusters with rich kinematics (max / min):" % len(richP))
for lc in LCS:
    out("    l_c %3g pc: remote %.2f to %.2f; rich %.2f to %.2f, a factor %.2f" % (
        lc, min(r["P"][lc] for r in hiP), max(r["P"][lc] for r in hiP), min(r["P"][lc] for r in richP),
        max(r["P"][lc] for r in richP), max(r["P"][lc] for r in richP) / min(r["P"][lc] for r in richP)))
out()

# ---------------------------------------------------------------- dispersion bins and model curves
VD = figread.parse_veldis()
VN = {"NGC_6426": ("NGC 6426", "ngc6426"), "Pal_12": ("Pal 12", "pal12"), "IC_4499": ("IC 4499", "ic4499"),
      "NGC_5466": ("NGC 5466", "ngc5466"), "Ter_7": ("Ter 7", "ter7"), "Pal_5": ("Pal 5", "pal5"),
      "Pal_1": ("Pal 1", "pal1"), "NGC_5053": ("NGC 5053", "ngc5053"), "Rup_106": ("Rup 106", "rup106"),
      "Ter_8": ("Ter 8", "ter8"), "Arp_2": ("Arp 2", "arp2"), "NGC_7492": ("NGC 7492", "ngc7492"),
      "Pal_13": ("Pal 13", "pal13"), "Pal_15": ("Pal 15", "pal15"), "Pyxis": ("Pyxis", "pyxis"),
      "Laevens_3": ("Laevens 3", "lae3"), "Pal_14": ("Pal 14", "pal14"), "NGC_2419": ("NGC 2419", "ngc2419"),
      "Pal_3": ("Pal 3", "pal3"), "Pal_4": ("Pal 4", "pal4"), "Crater": ("Crater", "crater"),
      "AM_1": ("AM 1", "am1"), "AM_4": ("AM 4", "am4"), "Eridanus": ("Eridanus", "eri"),
      "Whiting_1": ("Whiting 1", "whiting1"), "IC_1257": ("IC 1257", "ic1257"),
      "Sagittarius_II": ("Sgr 2", "sgrii")}      # the dispersion table spells Sagittarius II "Sgr 2"
OVR = {"ngc5053": (2.5, 10.0), "lae3": (1.0, 1.0), "rup106": (3.0, 10.0)}   # the figure reader's overrides for three clusters
LK = np.linspace(-3.0, 3.0, 12001)


def lnl_k(bins, lr, sg, sb=0.0):
    """The likelihood in log10 k: each bin against sqrt(k sigma_mod^2 + sb^2), Barlow variance."""
    tot = np.zeros_like(LK)
    for b in bins:
        sm = float(np.interp(math.log10(b["r"]), lr, sg))
        s = np.sqrt(10 ** LK * sm * sm + sb * sb)
        V = b["lo"] * b["up"] + (b["up"] - b["lo"]) * (s - b["s"])
        V = np.where(V > 1e-10, V, 1e-10)
        tot += -0.5 * (s - b["s"]) ** 2 / V
    return tot


def lnl_k_binaries(bins, lr, sg, cap, n=61):
    """The same, marginalized over a binary dispersion sb uniform on [0, cap] (one sb per cluster)."""
    if cap <= 0:
        return lnl_k(bins, lr, sg)
    stack = np.array([lnl_k(bins, lr, sg, sb) for sb in np.linspace(0.0, cap, n)])
    m = stack.max(axis=0)
    return m + np.log(np.exp(stack - m).mean(axis=0))


def marg(lnl, mu, s, grid=LK):
    """ln of the integral of exp(lnl) times the population prior N(log10 k; mu, s)."""
    w = np.exp(-0.5 * ((grid - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))
    m = lnl.max()
    return m + math.log(max(np.trapezoid(np.exp(lnl - m) * w, grid), 1e-300))


def interval(lnl, grid=LK):
    i = int(np.argmax(lnl))
    ok = grid[lnl >= lnl[i] - 0.5]
    return grid[i], ok.min(), ok.max()


KIN = {}
for r in remote:
    vn, stem = VN[r["name"]]
    bins = [dict(b) for b in VD.get(vn, []) if b["type"] == "RV"]
    if not bins:
        continue
    ov = OVR.get(stem, (None, None))
    lr, sg, top, first = figread.model_curve(stem, VD.get(vn, []), top_override=ov[0], first_override=ov[1])
    for b in bins:
        b["smod"] = float(np.interp(math.log10(b["r"]), lr, sg))
    KIN[r["name"]] = dict(r=r, bins=bins, lr=lr, sg=sg, lnl=lnl_k(bins, lr, sg))


# =============================================================================================
# A. Remote clusters with tabulated dispersions
# =============================================================================================
out("A. THE REMOTE CLUSTERS WITH TABULATED DISPERSIONS")
out("-" * 108)
out("Each tabulated bin of the database is compared with sqrt(k) times the database's N-body curve at the bin")
out("radius. k is the mass the dispersion implies over the catalog mass, so the dispersion-based ratio is")
out("Y_kin = k Y_cat. Provenance: 'sc>' marks a catalog mass from star counts whose model dispersion lies well")
out("below the measured one, by the factor sqrt(k) that A1 gives; 'mix' a")
out("faint cluster whose model meets its dispersion within about 1.5 errors; 'rich' N_PM > 0 or N_RV >= 50;")
out("'lum' a mass estimated from the luminosity. The Y_kin interval is where lnL is within 0.5 of its peak.")
out("Single values are quoted at l_c = %g pc, the floor of the window and the conservative end for rescaling;" % LC0)
out("49, 90, 100 and 200 pc follow in A5.")
SC_MISS = ("Crater", "Pal_3", "Pal_5", "Pal_13", "Sagittarius_II")
MIX = ("Pal_4", "Pal_12", "Pal_14", "Pyxis", "NGC_7492")


def prov(r):
    if r["name"] in SC_MISS:
        return "sc>"
    if r["name"] in ("Laevens_3", "IC_1257"):
        return "lum"
    if r["npm"] > 0 or r["nrv"] >= 50:
        return "rich"
    if r["name"] in MIX:
        return "mix"
    return "few"


CAPS = (0.5, 1.0, 1.5)


def mu_N(r, Y=None):
    return math.log10((YBASE if Y is None else Y) / r["ml"])


def mu_L(r, lc=LC0, Y=None, P=None):
    P = r["P"][lc] if P is None else P
    return math.log10((YLC[lc] if Y is None else Y) * P / r["ml"])


for n, v in KIN.items():
    v["lnl_b"] = {cap: lnl_k_binaries(v["bins"], v["lr"], v["sg"], cap) for cap in CAPS}

order = sorted(KIN, key=lambda n: -KIN[n]["r"]["P"][LC0])
noN = [n for n in order if n != "NGC_2419"]
NSET = len(noN)
out()
out("A1. Anchored on the baseline (Newton Y = %.3f; rescaling Y_LC P with Y_LC = %.3f), l_c = %g pc." % (
    YBASE, YLC[LC0], LC0))
out("    Columns: P; catalog ratio and provenance; k and Y_kin with their intervals; the two predicted ratios;")
out("    Delta lnL with the dispersions as measured, then with a binary dispersion marginalized up to 0.5, 1.0 and")
out("    1.5 km/s; then Delta lnL at the stellar value 1.8 (1.8 and 1.8 P), as measured.")
out("    %-14s %5s %5s %-4s %4s %18s %20s %5s %6s | %7s %7s %7s %7s | %7s" % (
    "cluster", "P", "Ycat", "prov", "N", "k (68%)", "Y_kin (68%)", "Y_N", "Y_LC P", "meas", "b<0.5", "b<1.0", "b<1.5", "at 1.8"))


def dA_cluster(n, lc, key="meas", Y_N=None, Y_L=None):
    v = KIN[n]
    r = v["r"]
    lnl = v["lnl"] if key == "meas" else v["lnl_b"][key]
    return marg(lnl, mu_N(r, Y_N), SBASE) - marg(lnl, mu_L(r, lc, Y=Y_L), SBASE)


SUM = {k: 0.0 for k in ("meas", 0.5, 1.0, 1.5, "paper")}
SUMN = {k: 0.0 for k in SUM}
ROWA = {}
for n in order:
    v = KIN[n]
    r = v["r"]
    k0, klo, khi = interval(v["lnl"])
    nst = sum(b["n"] or 0 for b in v["bins"])
    d = {"meas": dA_cluster(n, LC0)}
    for cap in CAPS:
        d[cap] = dA_cluster(n, LC0, cap)
    d["paper"] = dA_cluster(n, LC0, "meas", YPAPER, YPAPER)
    d["first"] = dA_cluster(n, LC49)
    for key in SUM:
        SUM[key] += d[key]
        if n != "NGC_2419":
            SUMN[key] += d[key]
    ROWA[n] = dict(k=(10 ** k0, 10 ** klo, 10 ** khi), Ykin=(r["ml"] * 10 ** k0, r["ml"] * 10 ** klo, r["ml"] * 10 ** khi), d=d)
    out("    %-14s %5.2f %5.2f %-4s %4d %5.2f [%5.2f,%5.2f] %6.2f [%5.2f,%6.2f] %5.2f %6.2f | %+7.2f %+7.2f %+7.2f %+7.2f | %+7.2f" % (
        n, r["P"][LC0], r["ml"], prov(r), nst, 10 ** k0, 10 ** klo, 10 ** khi, r["ml"] * 10 ** k0, r["ml"] * 10 ** klo,
        r["ml"] * 10 ** khi, YBASE, YLC[LC0] * r["P"][LC0], d["meas"], d[0.5], d[1.0], d[1.5], d["paper"]))
out("    sum, all %2d:           %+8.2f %+7.2f %+7.2f %+7.2f | %+7.2f" % (len(order), SUM["meas"], SUM[0.5], SUM[1.0], SUM[1.5], SUM["paper"]))
out("    sum without NGC 2419:  %+8.2f %+7.2f %+7.2f %+7.2f | %+7.2f   (NGC 2419 is tested by its Jeans fit in B)" % (
    SUMN["meas"], SUMN[0.5], SUMN[1.0], SUMN[1.5], SUMN["paper"]))


def subsum(names, key="meas"):
    return sum(ROWA[n]["d"][key] for n in names)


grpN = [n for n in noN if ROWA[n]["d"]["meas"] > 0]
grpL = [n for n in noN if ROWA[n]["d"]["meas"] < 0]
out("    As measured, %d clusters lean toward Newton (%+.2f in total): %s." % (len(grpN), subsum(grpN), ", ".join(x.replace("_", " ") for x in grpN)))
out("    %d lean toward rescaling (%+.2f in total): %s." % (len(grpL), subsum(grpL), ", ".join(x.replace("_", " ") for x in grpL)))
for lab, ex in (("without the five 'sc>' clusters", SC_MISS), ("without Pal 4", ("Pal_4",)),
                ("without the 'sc>' five and Pal 4", SC_MISS + ("Pal_4",))):
    nm = [n for n in noN if n not in ex]
    out("    %-62s %+8.2f (binaries to 1.0 km/s: %+.2f)" % (lab, subsum(nm), subsum(nm, 1.0)))
out("    The five 'sc>' clusters: %s." % ", ".join("%s %+.2f (binaries to 1.0 km/s %+.2f)" % (
    x.replace("_", " "), ROWA[x]["d"]["meas"], ROWA[x]["d"][1.0]) for x in SC_MISS))
sq = [math.sqrt(ROWA[x]["k"][0]) for x in SC_MISS]
out("    Their measured dispersions over the dispersions of their star-count models, sqrt(k): %.2f to %.2f." % (min(sq), max(sq)))
out()

# binary dispersion that each account needs, at the anchored values
out("A2. The binary dispersion each account needs (anchored values, l_c = %g pc): the sb in km/s that maximizes" % LC0)
out("    the likelihood at the predicted k, and the measured dispersion's position relative to the prediction.")
out("    Binaries can only raise a measured dispersion, so they reconcile an account only where the data lie above it.")
for n in order:
    v = KIN[n]
    r = v["r"]
    cells = []
    for lab, mu in (("Newton", mu_N(r)), ("rescaling", mu_L(r))):
        kpred = 10 ** mu
        sbs = np.linspace(0.0, 4.0, 401)
        lls = []
        for sb in sbs:
            tot = 0.0
            for b in v["bins"]:
                s = math.sqrt(kpred * b["smod"] ** 2 + sb * sb)
                V = max(b["lo"] * b["up"] + (b["up"] - b["lo"]) * (s - b["s"]), 1e-10)
                tot += -0.5 * (s - b["s"]) ** 2 / V
            lls.append(tot)
        j = int(np.argmax(lls))
        pos = "above" if ROWA[n]["k"][0] > kpred else "below"
        cells.append("%s: data %s, sb %.2f" % (lab, pos, sbs[j]))
    out("    %-14s %s; %s" % (n, cells[0], cells[1]))
out("    'above' and 'below' compare the best-fitting k with the predicted one.")
out()

# fitted population values
out("A3. Population value fitted to the group (one common factor per account), without")
out("    NGC 2419, l_c = %g pc. First with the scatter held at s_base, then with the scatter free." % LC0)


def best_shift(f, a=-1.5, b=1.5, n=301):
    xs = np.linspace(a, b, n)
    vals = [f(x) for x in xs]
    i = int(np.argmax(vals))
    lo, hi_ = xs[max(i - 1, 0)], xs[min(i + 1, n - 1)]
    for _ in range(60):
        m1, m2 = lo + (hi_ - lo) / 3, hi_ - (hi_ - lo) / 3
        if f(m1) < f(m2):
            lo = m1
        else:
            hi_ = m2
    x = 0.5 * (lo + hi_)
    return x, f(x)


LKC = LK[::10]                                   # a coarser grid (0.005 dex) for the fitted population values


def fit_population(names, which, lc=LC0, cap=None, scat=None, fieldA=False):
    def lnl_of(n):
        return (KIN[n]["lnl"] if cap is None else KIN[n]["lnl_b"][cap])[::10]

    def mu_of(r):
        if which == "N":
            return mu_N(r)
        if fieldA:
            return math.log10(YLC_A * r["PA"] / r["ml"])
        return mu_L(r, lc)

    def f(sh, s):
        tot = 0.0
        for n in names:
            tot += marg(lnl_of(n), mu_of(KIN[n]["r"]) + sh, s, LKC)
        return tot
    if scat is not None:
        x, val = best_shift(lambda sh: f(sh, scat))
        return x, val, scat
    bestf = None
    for s in np.linspace(0.04, 0.60, 29):
        x, val = best_shift(lambda sh: f(sh, s), n=61)
        if bestf is None or val > bestf[1]:
            bestf = (x, val, s)
    return bestf


FIT = {}
for cap in (None, 1.0):
    for scat in (SBASE, None):
        xN, lN, sN = fit_population(noN, "N", cap=cap, scat=scat)
        xL, lL, sL = fit_population(noN, "L", cap=cap, scat=scat)
        FIT[(cap, scat)] = (xN, xL, sN, sL, lN - lL)
        out("    %-34s %-15s Delta lnL %+7.2f (Newton Y %.2f; rescaling Y %.2f, so Y P runs %.1f to %.1f; scatter %.2f / %.2f dex)" % (
            "as measured" if cap is None else "binaries marginalized to 1.0 km/s", "scatter s_base" if scat is not None else "scatter free",
            lN - lL, YBASE * 10 ** xN, YLC[LC0] * 10 ** xL, YLC[LC0] * 10 ** xL * min(KIN[n]["r"]["P"][LC0] for n in noN),
            YLC[LC0] * 10 ** xL * max(KIN[n]["r"]["P"][LC0] for n in noN), sN, sL))
xN, xL = FIT[(None, SBASE)][0], FIT[(None, SBASE)][1]
out("    Per cluster at the fitted values (scatter s_base, as measured):")
line = []
for n in noN:
    r = KIN[n]["r"]
    d = marg(KIN[n]["lnl"], mu_N(r) + xN, SBASE) - marg(KIN[n]["lnl"], mu_L(r) + xL, SBASE)
    ROWA[n]["d"]["fit"] = d
    line.append("%s %+.2f" % (n.replace("_", " "), d))
for i in range(0, len(line), 7):
    out("      " + "; ".join(line[i:i + 7]))
out()

# star-count anchor
out("A4. Each faint cluster's own catalog ratio as its stellar value, l_c = %g pc: Newton" % LC0)
out("    predicts k = 1 and rescaling k = P, with the quoted fractional error in dex as the width of the prior. The")
out("    catalog ratios of the 'sc>' clusters are star-count values. For Pal 4 and Pal 14 the anchor is partly circular,")
out("    since their models meet their dispersions. Six clusters, then Sagittarius II.")
SC7 = ("Crater", "Pal_3", "Pal_13", "Pal_5", "Pal_4", "Pal_14", "Sagittarius_II")
tot6 = {None: 0.0, 1.0: 0.0}
tot7 = {None: 0.0, 1.0: 0.0}
for n in SC7:
    v = KIN[n]
    r = v["r"]
    e = r["dml"] / r["ml"] / LN10
    cells = []
    for cap in (None, 1.0):
        lnl = v["lnl"] if cap is None else v["lnl_b"][cap]
        d = marg(lnl, 0.0, e) - marg(lnl, math.log10(r["P"][LC0]), e)
        tot7[cap] += d
        if n != "Sagittarius_II":
            tot6[cap] += d
        cells.append(d)
    ROWA[n]["d"]["own"] = cells[0]
    out("    %-15s k %5.2f, P %.2f: Delta lnL %+7.2f as measured, %+7.2f with binaries to 1.0 km/s" % (
        n, ROWA[n]["k"][0], r["P"][LC0], cells[0], cells[1]))
out("    The six together: %+.2f as measured, %+.2f with binaries to 1.0 km/s; with Sagittarius II: %+.2f and %+.2f." % (
    tot6[None], tot6[1.0], tot7[None], tot7[1.0]))
out()

# catchment and mass sensitivity
out("A5. The sums at each catchment length, 49 and 90 pc and the window's %g, 100 and 200 pc (without" % LC0)
out("    NGC 2419): anchored on the inner clusters, as measured and with binaries to 1.0 km/s; at the stellar value 1.8;")
out("    fitted with the scatter at s_base, as measured and with binaries to 1.0 km/s.")
A5 = {}
for lc in LCS:
    s_a = sum(dA_cluster(n, lc) for n in noN)
    s_b = sum(dA_cluster(n, lc, 1.0) for n in noN)
    s_p = sum(dA_cluster(n, lc, "meas", YPAPER, YPAPER) for n in noN)
    xN_, lN_, _ = fit_population(noN, "N", lc=lc, scat=SBASE)
    xL_, lL_, _ = fit_population(noN, "L", lc=lc, scat=SBASE)
    xNb, lNb, _ = fit_population(noN, "N", lc=lc, cap=1.0, scat=SBASE)
    xLb, lLb, _ = fit_population(noN, "L", lc=lc, cap=1.0, scat=SBASE)
    A5[lc] = dict(anch=s_a, anchb=s_b, paper=s_p, fit=lN_ - lL_, fitb=lNb - lLb)
    out("    l_c %3g pc%s: anchored %+7.2f (binaries %+7.2f); at 1.8 %+7.2f; fitted %+7.2f (binaries %+6.2f; rescaling Y %.2f)" % (
        lc, "", s_a, s_b, s_p, lN_ - lL_, lNb - lLb, YLC[lc] * 10 ** xL_))


def wrange(key):
    vals = [A5[lc][key] for lc in WIN]
    return min(vals), max(vals)


out("    Window ranges (%g to 200 pc): anchored %+.1f to %+.1f; binaries %+.1f to %+.1f; at 1.8 %+.1f to %+.1f;" % (
    (WIN[0],) + wrange("anch") + wrange("anchb") + wrange("paper")))
out("    fitted %+.1f to %+.1f; fitted with binaries %+.1f to %+.1f." % (wrange("fit") + wrange("fitb")))
# own field from the baryonic mass under rescaling, Y_LC L, instead of the catalog mass
s_m = 0.0
for n in noN:
    r = KIN[n]["r"]
    Pm = P_lc(YLC[LC0] * r["L"], r["rgc"], LC0)
    s_m += marg(KIN[n]["lnl"], mu_N(r), SBASE) - marg(KIN[n]["lnl"], mu_L(r, P=Pm), SBASE)
out("    own field from the baryonic mass Y_LC L under rescaling instead of the catalog mass (%g pc, anchored): %+7.2f" % (LC0, s_m))
s_fa = sum(marg(KIN[n]["lnl"], mu_N(KIN[n]["r"]), SBASE) -
           marg(KIN[n]["lnl"], math.log10(YLC_A * KIN[n]["r"]["PA"] / KIN[n]["r"]["ml"]), SBASE) for n in noN)
xN_, lN_, _ = fit_population(noN, "N", scat=SBASE)
xL_, lL_, _ = fit_population(noN, "L", scat=SBASE, fieldA=True)
out("    a flat-curve field A (230 km/s, R floored at 3 kpc) in place of the point mass (%g pc): baseline median P %.3f, Y_LC %.2f;" % (LC0, PBASE_A, YLC_A))
out("        anchored %+7.2f; fitted %+7.2f (rescaling Y %.2f)" % (s_fa, lN_ - lL_, YLC_A * 10 ** xL_))
out()

# the correlation with P within the remote clusters: log10 Y_kin = a + b log10 P, intrinsic scatter s
out("A6. The correlation with P (criterion F-II.4) within the remote clusters with dispersions (without NGC 2419):")
out("    log Y_kin = a + b log P, with the intrinsic scatter s and the intercept a fitted. b = 0 is Newton and b = 1")
out("    rescaling. Each cluster enters through its likelihood in k, smoothed by the scatter. Spearman's rho is over")
out("    the best-fitting Y_kin.")
SG_A6 = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80])
B_A6 = np.round(np.arange(-1.0, 2.0001, 0.025), 3)
A_A6 = np.linspace(-0.6, 1.6, 441)


def smoothed(lnl, s):
    """ln of the integral of exp(lnl(x)) N(x; mu, s) dx, as a function of mu on the coarse grid LKC."""
    l = lnl[::10]
    m = l.max()
    dx = LKC[1] - LKC[0]
    half = int(round(6 * s / dx))
    ker = np.exp(-0.5 * (np.arange(-half, half + 1) * dx / s) ** 2) / (s * math.sqrt(2 * math.pi)) * dx
    conv = np.convolve(np.exp(l - m), ker, mode="full")[half:half + len(l)]   # centered, same length as l
    return m + np.log(np.maximum(conv, 1e-300))


def spearman(x, y):
    def ranks(a):
        a = np.asarray(a, float)
        o = np.argsort(a, kind="mergesort")
        rk = np.empty(len(a))
        rk[o] = np.arange(1, len(a) + 1)
        return rk
    rx, ry = ranks(x), ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / math.sqrt((rx * rx).sum() * (ry * ry).sum()))


def corr_fit(names, lc, cap=None):
    x = np.array([math.log10(KIN[n]["r"]["P"][lc]) for n in names])
    lyc = np.array([math.log10(KIN[n]["r"]["ml"]) for n in names])
    lnls = [(KIN[n]["lnl"] if cap is None else KIN[n]["lnl_b"][cap]) for n in names]
    S = {s: np.array([smoothed(l, s) for l in lnls]) for s in SG_A6}
    prof = []
    for b in B_A6:
        best_b = -np.inf
        for s in SG_A6:
            mu = A_A6[:, None] + b * x[None, :] - lyc[None, :]          # predicted log k, (na, nclusters)
            tot = np.zeros(len(A_A6))
            for j in range(len(names)):
                tot += np.interp(mu[:, j], LKC, S[s][j], left=-1e9, right=-1e9)
            best_b = max(best_b, float(tot.max()))
        prof.append(best_b)
    prof = np.array(prof)
    i = int(np.argmax(prof))
    ok = B_A6[prof >= prof[i] - 0.5]
    l0 = float(prof[np.argmin(abs(B_A6 - 0.0))])
    l1 = float(prof[np.argmin(abs(B_A6 - 1.0))])
    yk = [math.log10(KIN[n]["r"]["ml"]) + LK[int(np.argmax(KIN[n]["lnl"]))] for n in names]
    return B_A6[i], ok.min(), ok.max(), l0 - l1, spearman(x, yk)


A6 = {}
noSC = [n for n in noN if n not in SC_MISS]
for lab, names, cap, lcs in (("all, as measured", noN, None, LCS), ("all, binaries to 1.0 km/s", noN, 1.0, LCS),
                             ("without the five 'sc>' clusters", noSC, None, (LC0, 100.0))):
    for lc in lcs:
        b, blo, bhi, d, rho = corr_fit(names, lc, cap)
        A6[(lab, lc)] = (b, blo, bhi, d, rho)
        out("    %-32s l_c %3g pc: slope %+.3f (68%% %+.3f to %+.3f); Delta lnL (slope 0 against 1) %+6.2f; rho %+.2f" % (
            lab, lc, b, blo, bhi, d, rho))
out("    Unresolved binaries add a dispersion of fixed size, so they raise k most in the clusters with the smallest")
out("    intrinsic dispersions, the low-mass remote clusters at the largest P. They can therefore make a positive slope.")
out()

# =============================================================================================
# B. NGC 2419: the normalization of a Newtonian-shaped profile (per-star Jeans fit)
# =============================================================================================
out("B. NGC 2419: THE NORMALIZATION OF ITS NEWTONIAN-SHAPED PROFILE")
out("-" * 108)
out("Under ambient rescaling the whole cluster is rescaled by one factor P, one P per cluster, so its profile is")
out("Newtonian-shaped with dynamical ratio Y_LC P. The shape cannot separate the two accounts and only the")
out("normalization can. The Newtonian Jeans likelihood in M/L is computed from the multi-Gaussian")
out("photometry of Ibata et al. 2011 Table 2, Osipkov-Merritt anisotropy from 2 to 500 pc and isotropy, the stability")
out("priors, the 158-star window sample with a uniform contaminant up to 10 percent) and profiled over the anisotropy.")

GPC = 4.3009e-3
A0PC = A0 * PC / 1e6
D2419 = 87.5e3
PCARC = D2419 * math.pi / (180 * 3600)
AV = 3.1 * 0.08
MSUNV = 4.83
T2 = [(3.202, 19.652, 0.097), (7.735, 19.857, 0.062), (12.572, 20.019, 0.053), (17.495, 20.233, 0.048),
      (24.822, 20.740, 0.036), (34.924, 21.176, 0.044), (50.962, 22.159, 0.040), (71.989, 23.090, 0.036),
      (89.259, 23.740, 0.038), (113.743, 24.475, 0.038), (152.306, 25.365, 0.039), (212.092, 26.660, 0.057),
      (304.503, 27.879, 0.069), (402.642, 29.250, 0.123), (528.313, 30.991, 0.292)]
Rsb = np.array([t[0] for t in T2]) * PCARC
Ssb = 10 ** (0.4 * (MSUNV + 21.572 - (np.array([t[1] for t in T2]) - AV)))
dS = Ssb * 0.4 * math.log(10) * np.array([t[2] for t in T2])


def nnls(A, b, it=1000):
    m, n = A.shape
    Pset = []
    x = np.zeros(n)
    w = A.T @ (b - A @ x)
    k = 0
    while k < it and len(Pset) < n and np.max(np.where(np.isin(np.arange(n), Pset), -np.inf, w)) > 1e-10 * np.abs(w).max():
        k += 1
        j = int(np.argmax(np.where(np.isin(np.arange(n), Pset), -np.inf, w)))
        Pset.append(j)
        while True:
            z = np.zeros(n)
            z[Pset] = np.linalg.lstsq(A[:, Pset], b, rcond=None)[0]
            if np.all(z[Pset] > 0):
                x = z
                break
            neg = [i for i in Pset if z[i] <= 0]
            al = min(x[i] / (x[i] - z[i]) for i in neg)
            x = x + al * (z - x)
            Pset = [i for i in Pset if x[i] > 1e-15]
        w = A.T @ (b - A @ x)
    return x


SIGG = np.geomspace(0.6, 450.0, 16)
Amat = np.exp(-0.5 * (Rsb[:, None] / SIGG) ** 2)
amp = nnls(Amat / dS[:, None], Ssb / dS)
Ltot = float((2 * math.pi * amp * SIGG ** 2).sum())
erfv = np.vectorize(math.erf)


def Sigma(R):
    R = np.asarray(R, float)
    return (amp * np.exp(-0.5 * (R[..., None] / SIGG) ** 2)).sum(-1)


def nuL(rr):
    rr = np.asarray(rr, float)
    return (amp / (math.sqrt(2 * math.pi) * SIGG) * np.exp(-0.5 * (rr[..., None] / SIGG) ** 2)).sum(-1)


def Lenc(rr):
    rr = np.asarray(rr, float)
    x = rr[..., None] / SIGG
    return (2 * math.pi * amp * SIGG ** 2 * (erfv(x / math.sqrt(2)) - math.sqrt(2 / math.pi) * x * np.exp(-x * x / 2))).sum(-1)


rgrid = np.geomspace(0.01, 6000.0, 3000)
lnr = np.log(rgrid)
NU = nuL(rgrid)
LE = Lenc(rgrid)
r3h = float(np.interp(0.5 * Ltot, LE, rgrid))


def cumint_from_r(y):
    f = y * rgrid
    seg = 0.5 * (f[1:] + f[:-1]) * np.diff(lnr)
    return np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])


def nus2(g, ra):
    if np.isfinite(ra):
        fac = rgrid * rgrid + ra * ra
        return cumint_from_r(fac * NU * g) / fac
    return cumint_from_r(NU * g)


TQ = np.linspace(0.0, 1.0, 401)


class Proj:
    def __init__(self, R):
        self.R = np.asarray(R, float)
        tmax = np.arccosh(np.maximum(rgrid[-1] / self.R, 1.0 + 1e-9))
        self.t = TQ[None, :] * tmax[:, None]
        self.rr = self.R[:, None] * np.cosh(self.t)
        self.jac = self.R[:, None] * np.cosh(self.t)
        self.dt = tmax / (len(TQ) - 1)
        self.l = np.log(self.rr)
        self.S = Sigma(self.R)

    def s2(self, ns, ra):
        val = np.exp(np.interp(self.l, lnr, np.log(np.maximum(ns, 1e-300))))
        if np.isfinite(ra):
            val = val * (1 - self.R[:, None] ** 2 / (self.rr ** 2 + ra * ra))
        f = val * self.jac
        integ = (0.5 * (f[:, 1:] + f[:, :-1])).sum(1) * self.dt
        return 2 * integ / self.S


rh_mask = rgrid <= r3h
gam = -np.gradient(np.log(np.maximum(NU, 1e-300)), lnr)
gmask = NU > NU[0] * 1e-8


def admissible(ns, ra):
    if not np.isfinite(ra):
        return True
    beta = rgrid * rgrid / (rgrid * rgrid + ra * ra)
    if np.any(2 * beta[gmask & (rgrid > 0.05)] > gam[gmask & (rgrid > 0.05)] + 1e-9):
        return False
    xi = np.trapezoid((ns * rgrid ** 3)[rh_mask], lnr[rh_mask]) / np.trapezoid(((1 - beta) * ns * rgrid ** 3)[rh_mask], lnr[rh_mask])
    return xi <= 1.5


stars = []
for ln in open(os.path.join(INP, "rv", "ngc2419.txt"), encoding="utf-8", errors="replace").read().splitlines()[2:]:
    t = ln.split()
    try:
        i = next(j for j, x in enumerate(t) if "." in x)
        dcen, vv, ee = float(t[i + 2]), float(t[i + 3]), float(t[i + 4])
        ps, prv, ppm = float(t[i + 7]), float(t[i + 8]), float(t[i + 9])
    except (StopIteration, ValueError, IndexError):
        continue
    stars.append((dcen, vv, ee, ps, prv, ppm))
win = [s for s in stars if abs(s[1] + 20.9) < 20 and (s[5] >= 0.05 or s[5] > 9) and s[3] >= 0.05]
proj = Proj(np.array([s[0] for s in win]) * PCARC)
vst = np.array([s[1] for s in win])
est = np.array([s[2] for s in win])
FG = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
LY = np.linspace(math.log10(0.4), math.log10(12.0), 361)       # log10 M/L grid


def lnl_stars_vec(s2y):
    """Per-star lnL for an array of models s2y (nY, nstar), mean by vectorized golden section in [-26, -16],
    maximized over the contaminant fraction grid (uniform over 40 km/s)."""
    var = s2y + est[None, :] ** 2
    norm = np.sqrt(2 * math.pi * var)
    bestv = np.full(s2y.shape[0], -np.inf)
    gr = (math.sqrt(5) - 1) / 2
    for f in FG:
        def ll(mu):
            g = np.exp(-0.5 * (vst[None, :] - mu[:, None]) ** 2 / var) / norm
            return np.log((1 - f) * g + f / 40.0).sum(1)
        a = np.full(s2y.shape[0], -26.0)
        b = np.full(s2y.shape[0], -16.0)
        while (b - a).max() > 1e-3:          # golden section for a maximum, both points evaluated each step
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            m = ll(c) > ll(d)
            b = np.where(m, d, b)
            a = np.where(m, a, c)
        bestv = np.maximum(bestv, ll(0.5 * (a + b)))
    return bestv


RAS = list(np.geomspace(2.0, 500.0, 60)) + [np.inf]
gN1 = GPC * 1.0 * LE / rgrid ** 2
prof = np.full(len(LY), -np.inf)
best_ra = None
for ra in RAS:
    ns1 = nus2(gN1, ra)
    if not admissible(ns1, ra):
        continue
    s21 = proj.s2(ns1, ra)
    vals = lnl_stars_vec(10 ** LY[:, None] * s21[None, :])
    if best_ra is None or vals.max() > best_ra[0]:
        best_ra = (vals.max(), ra, 10 ** LY[int(np.argmax(vals))])
    prof = np.maximum(prof, vals)
j0, jlo, jhi = interval(prof, LY)
YJ = 10 ** j0
out("Photometric luminosity %.4e Lsun. Window sample: %d stars." % (Ltot, len(win)))
out("Newtonian Jeans fit: lnL %.3f at M/L %.3f, r_a %.1f pc;" % (prof.max(), YJ, best_ra[1]))
out("68 percent interval of the dynamical ratio %.2f to %.2f." % (10 ** jlo, 10 ** jhi))
B2419 = {}
for lc in LCS:
    P = byname["NGC_2419"]["P"][lc]
    dA = marg(prof, math.log10(YBASE), SBASE, LY) - marg(prof, math.log10(YLC[lc] * P), SBASE, LY)
    dP = marg(prof, math.log10(YPAPER), SBASE, LY) - marg(prof, math.log10(YPAPER * P), SBASE, LY)
    B2419[lc] = (P, dA, dP)
    out("    l_c %3g pc: P %.2f; anchored: Newton %.2f, rescaling %.2f, Delta lnL %+6.2f; at the stellar value 1.8: 1.8 against %.2f, %+6.2f;" % (
        lc, P, YBASE, YLC[lc] * P, dA, YPAPER * P, dP))
    out("               rescaling needs a stellar ratio of %.2f (68 percent %.2f to %.2f)." % (YJ / P, 10 ** jlo / P, 10 ** jhi / P))
out("Globular clusters with well-measured masses show ratios of about 1.4 to 2.5 (Baumgardt, Sollima and Hilker 2020).")
out("Those are Newtonian-dynamical ratios, so under rescaling they are not an independent stellar range. The needed")
out("stellar ratio is therefore compared with the stellar value 1.8 and with the inner anchor of rescaling,")
out("Y_LC = %.2f. At 49 pc the needed ratio is consistent with 1.8 within its 68 percent range. Inside the window," % YLC[LC0])
out("%g to 200 pc, it lies below both 1.8 and the inner anchor." % LC0)
out("Two conditions qualify this. Rescaling of the whole cluster by one factor is assumed. Were only a sphere about")
out("the centre rescaled, the outskirts beyond 90 to 100 pc, about 8 percent of the light and the outer stars to 194 pc,")
out("would lie outside the catchment and be unscreened. And the inner anchor of 3.15 is the")
out("calibration of rescaling on the inner clusters, so it inherits the radial systematic of Part C.")
out()

# =============================================================================================
# C. Clusters whose catalog masses rest on rich kinematics: the correlation of dynamical masses with P
# =============================================================================================
out("C. THE CLUSTERS WITH RICH KINEMATICS: THE CORRELATION OF DYNAMICAL MASSES WITH P (CRITERION F-II.4)")
out("-" * 108)
out("Sample: N_PM > 0 or N_RV >= 50 ('kinematically rich'), with a nonzero quoted error (luminosity-based masses")
out("carry none). For these clusters the database's model reproduces the measured dispersions, so the catalog ratio")
out("is a dynamical one. Each error is raised to the sampling floor sqrt(2/N), N = N_RV + 2 N_PM. Rescaling predicts")
out("log Y = log Y_pop + 1.0 log P, Newton a slope of 0.")
out("NGC 2419 enters with its Jeans ratio from B (%.2f, 68 percent %.2f to %.2f), since the database's model misses" % (
    YJ, 10 ** jlo, 10 ** jhi))
out("its profile shape and its catalog ratio is %.2f. 'catalog NGC 2419' rows keep the catalog ratio," % (
    byname["NGC_2419"]["ml"]))
out("and 'without NGC 2419' rows leave it out.")
rich_cat = [r for r in good if r["dml"] > 0 and (r["npm"] > 0 or r["nrv"] >= 50)]


def with_jeans(sample):
    """NGC 2419 with its Jeans ratio and the half-width of its 68 percent interval in dex as the error."""
    res = []
    for r in sample:
        if r["name"] == "NGC_2419":
            q = dict(r)
            q["ml"] = YJ
            q["dml"] = YJ * 0.5 * (jhi - jlo) * LN10        # so that dml/ml/ln10 is the half-width in dex
            q["floor"] = 0.0
            res.append(q)
        else:
            res.append(r)
    return res


rich = with_jeans(rich_cat)
rich_no2419 = [r for r in rich_cat if r["name"] != "NGC_2419"]
med_n = float(np.median([r["nrv"] + r["npm"] for r in rich_cat]))
out("Velocities and proper motions per cluster in the sample: median %.0f; %d clusters have fewer than 100." % (
    med_n, sum(1 for r in rich_cat if r["nrv"] + r["npm"] < 100)))


def vnorm(n):
    return n.strip().upper().replace("_", "").replace(" ", "").replace("-", "").replace("2MASSGC", "2MSGC")


VDN = {vnorm(k): v for k, v in VD.items()}
VDN[vnorm("Sagittarius_II")] = VD.get("Sgr 2", [])
VDN[vnorm("Patchick_126")] = VD.get("Pat 126", [])
for r in good:
    bins_ = [b for b in VDN.get(vnorm(r["name"]), []) if b["s"] > 0]
    inv = sum(1.0 / ((b["up"] + b["lo"]) / b["s"]) ** 2 for b in bins_)
    r["prec"] = 1.0 / math.sqrt(inv) if inv > 0 else float("inf")
prec_cat = [r for r in good if r["dml"] > 0 and r["prec"] <= 0.20]
prec_rich = with_jeans(prec_cat)

SG_C = np.linspace(0.005, 0.5, 496)
BS_C = np.linspace(-2.0, 2.0, 4001)


def wls_scan(y, X, e, j, sgrid):
    """For each intrinsic scatter s: weighted least squares of y on the columns of X with weights
    1/(s^2 + e^2). The likelihood is Gaussian in the linear coefficients at fixed s, so the profile over the
    others at a fixed value b of coefficient j is ll_max(s) - (b - b_hat(s))^2 / (2 var_b(s)), exactly."""
    llm, bh, vb, cf = [], [], [], []
    for s in sgrid:
        w = 1.0 / (s * s + e * e)
        A = X * np.sqrt(w)[:, None]
        M = A.T @ A
        c = np.linalg.solve(M, A.T @ (y * np.sqrt(w)))
        res = y - X @ c
        llm.append(-0.5 * (w * res * res).sum() + 0.5 * np.log(w).sum())
        bh.append(c[j])
        vb.append(np.linalg.inv(M)[j, j])
        cf.append(c)
    return np.array(llm), np.array(bh), np.array(vb), cf


def arrays(sample, lc, fieldA=False):
    x = np.array([math.log10(r["PA"] if fieldA else r["P"][lc]) for r in sample])
    y = np.array([math.log10(r["ml"]) for r in sample])
    e = np.array([max(r["dml"] / r["ml"], r["floor"]) / LN10 for r in sample])
    return x, y, e


def slope_fits(sample, lc, covars=(), fieldA=False):
    """log10 Y = a + b log10 P (+ covariates); b = 0 is Newton, b = 1 rescaling; scatter free or held."""
    x, y, e = arrays(sample, lc, fieldA)
    cols = [np.ones_like(x), x] + [np.array([cv(r) for r in sample]) for cv in covars]
    X = np.vstack(cols).T
    llm, bh, vb, cf = wls_scan(y, X, e, 1, SG_C)
    prof = (llm[:, None] - 0.5 * (BS_C[None, :] - bh[:, None]) ** 2 / vb[:, None]).max(axis=0)
    i = int(np.argmax(prof))
    ok = BS_C[prof >= prof[i] - 0.5]

    def at_b(bv):
        v = llm - 0.5 * (bv - bh) ** 2 / vb
        k = int(np.argmax(v))
        return v[k], SG_C[k]
    l0, s0 = at_b(0.0)
    l1, s1 = at_b(1.0)
    lf, bf, vf, _ = wls_scan(y, X, e, 1, [SBASE])
    dfix = (lf[0] - 0.5 * (0.0 - bf[0]) ** 2 / vf[0]) - (lf[0] - 0.5 * (1.0 - bf[0]) ** 2 / vf[0])
    res = dict(n=len(sample), b=BS_C[i], blo=ok.min(), bhi=ok.max(), d=l0 - l1, s0=s0, s1=s1, dfix=dfix,
               rho=spearman(x, y), xmin=10 ** x.min(), xmax=10 ** x.max())
    if not covars:                                    # the population value at slopes 0 and 1
        for bv, key in ((0.0, "Y0"), (1.0, "Y1")):
            s = res["s0"] if bv == 0.0 else res["s1"]
            w = 1.0 / (s * s + e * e)
            res[key] = 10 ** ((w * (y - bv * x)).sum() / w.sum())
    return res


def bounded_fit(sample, lc, cmax, step=0.005):
    """A power-law radial trend of the stellar ratio, Y_* proportional to R_GC^c with |c| <= cmax, fitted under
    both accounts (intercept and scatter free); returns Delta lnL (slope 0 against 1)."""
    x, y, e = arrays(sample, lc)
    lr = np.array([math.log10(r["rgc"]) for r in sample])
    X = np.vstack([np.ones_like(x), x]).T
    best = {0.0: -np.inf, 1.0: -np.inf}
    for c in np.arange(-cmax, cmax + 1e-9, step):
        llm, bh, vb, _ = wls_scan(y - c * lr, X, e, 1, SG_C)
        for bv in (0.0, 1.0):
            best[bv] = max(best[bv], float(np.max(llm - 0.5 * (bv - bh) ** 2 / vb)))
    return best[0.0] - best[1.0]


lgR = lambda r: math.log10(r["rgc"])
lgR2 = lambda r: math.log10(r["rgc"]) ** 2
lgR3 = lambda r: math.log10(r["rgc"]) ** 3
lgM = lambda r: math.log10(r["mass"])
TRH = lambda r: r["trh"]
MFS = lambda r: r["mf"]

CRES = {}
out()
out("C1. The slope of log Y on log P, the population value and the scatter fitted (Delta lnL: slope 0 against 1).")
for lab, smp in (("all", rich), ("R_GC > 8 kpc", [r for r in rich if r["rgc"] > 8]),
                 ("R_GC > 15 kpc", [r for r in rich if r["rgc"] > 15])):
    for lc in LCS:
        fz = slope_fits(smp, lc)
        CRES[(lab, lc)] = fz
        out("  %-14s l_c %3g pc%s: N %3d, P %.2f to %.2f; Spearman %+.2f; slope %+.2f (68%% %+.2f to %+.2f);" % (
            lab, lc, "", fz["n"], fz["xmin"], fz["xmax"], fz["rho"], fz["b"], fz["blo"], fz["bhi"]))
        out("  %-14s              Delta lnL %+7.1f (Y %.2f, s %.3f; Y %.2f, s %.3f); with the scatter held at s_base %+7.1f" % (
            "", fz["d"], fz["Y0"], fz["s0"], fz["Y1"], fz["s1"], fz["dfix"]))
out("The boost tracks galactocentric radius: Spearman rank correlation of log P (%g pc) with log R_GC over the sample" % LC0)
xr = [math.log10(r["P"][LC0]) for r in rich]
lrr = [math.log10(r["rgc"]) for r in rich]
cub = np.polyfit(lrr, xr, 3)
out("%+.3f, and log P departs from a cubic in log R_GC by %.3f dex rms over a range of %.2f dex." % (
    spearman(lrr, xr), float(np.std(np.array(xr) - np.polyval(cub, lrr))), max(xr) - min(xr)))
out()

out("C2. Sensitivity to a radial trend of the stellar ratio, to population covariates and to the definition of rich")
out("    kinematics. Delta lnL (slope 0 against 1), population value and scatter fitted, at the window's %g, 100 and" % WIN[0])
out("    200 pc. A free power law is a covariate log R_GC; the cubic adds (log R_GC)^2 and (log R_GC)^3; the bounded")
out("    power laws hold |c| <= 0.1 and <= 0.2 in Y_* ~ R_GC^c, a factor of 1.6 and 2.7 across the sample's 0.7 to")
out("    96 kpc. 'MF' is the catalog's mass-function slope (column 29). 'Table precision' defines rich kinematics as")
out("    a mass precision of 0.20 or better from the database's own dispersion table, all bins combined.")
out("    %-46s %-24s %-24s" % ("", "all clusters", "R_GC > 8 kpc"))
out("    %-46s %7s %7s %7s  %7s %7s %7s" % ("run", "%g pc" % WIN[0], "100 pc", "200 pc", "%g pc" % WIN[0], "100 pc", "200 pc"))
SENS = {}


def sens_row(label, fn):
    vals = []
    for sub in ("all", "R8"):
        for lc in WIN:
            v = fn(sub, lc)
            vals.append(v)
            SENS[(label, sub, lc)] = v
    out("    %-46s %+7.1f %+7.1f %+7.1f  %+7.1f %+7.1f %+7.1f" % ((label,) + tuple(vals)))


def pick(sample, sub):
    return sample if sub == "all" else [r for r in sample if r["rgc"] > 8]


sens_row("none", lambda sub, lc: slope_fits(pick(rich, sub), lc)["d"])
sens_row("free power law in R_GC", lambda sub, lc: slope_fits(pick(rich, sub), lc, (lgR,))["d"])
sens_row("cubic in log R_GC", lambda sub, lc: slope_fits(pick(rich, sub), lc, (lgR, lgR2, lgR3))["d"])
sens_row("power law bounded, factor 1.6 (|c| <= 0.1)", lambda sub, lc: bounded_fit(pick(rich, sub), lc, 0.1))
sens_row("power law bounded, factor 2.7 (|c| <= 0.2)", lambda sub, lc: bounded_fit(pick(rich, sub), lc, 0.2))
sens_row("MF slope", lambda sub, lc: slope_fits(pick(rich, sub), lc, (MFS,))["d"])
sens_row("MF slope, log mass and lg Trh", lambda sub, lc: slope_fits(pick(rich, sub), lc, (MFS, lgM, TRH))["d"])
sens_row("log mass and lg Trh", lambda sub, lc: slope_fits(pick(rich, sub), lc, (lgM, TRH))["d"])
sens_row("table precision <= 0.20 (%d clusters)" % len(prec_rich), lambda sub, lc: slope_fits(pick(prec_rich, sub), lc)["d"])
sens_row("without NGC 2419", lambda sub, lc: slope_fits(pick(rich_no2419, sub), lc)["d"])
sens_row("catalog NGC 2419", lambda sub, lc: slope_fits(pick(rich_cat, sub), lc)["d"])
fA_all = slope_fits(rich, LC0, fieldA=True)
fA_8 = slope_fits([r for r in rich if r["rgc"] > 8], LC0, fieldA=True)
out("    flat-curve field A in place of the point mass, %g pc: %+.1f (slope %+.2f), beyond 8 kpc %+.1f" % (
    LC0, fA_all["d"], fA_all["b"], fA_8["d"]))
def trend_c(sample, lc, bfix):
    """The fitted power-law index c of a radial trend Y_* ~ R_GC^c at fixed slope b, intercept and scatter free."""
    x, y, e = arrays(sample, lc)
    lr = np.array([math.log10(r["rgc"]) for r in sample])
    X = np.vstack([np.ones_like(lr), lr]).T
    best = None
    for s in SG_C:
        w = 1.0 / (s * s + e * e)
        A = X * np.sqrt(w)[:, None]
        c = np.linalg.solve(A.T @ A, A.T @ ((y - bfix * x) * np.sqrt(w)))
        res = y - bfix * x - X @ c
        ll = -0.5 * (w * res * res).sum() + 0.5 * np.log(w).sum()
        if best is None or ll > best[0]:
            best = (ll, c[1])
    return best[1]


span = math.log10(max(r["rgc"] for r in rich)) - math.log10(min(r["rgc"] for r in rich))
cN, cR = trend_c(rich, LC0, 0.0), trend_c(rich, LC0, 1.0)
out("    With the free power law (%g pc) the fitted trend of the stellar ratio is R_GC^%+.2f under Newton and R_GC^%+.2f" % (
    LC0, cN, cR))
out("    under rescaling. Rescaling needs its stellar ratio to fall by a factor %.1f across the sample's %.1f to %.0f kpc." % (
    10 ** (-cR * span), min(r["rgc"] for r in rich), max(r["rgc"] for r in rich)))
out()

out("C4. Binned medians of the dynamical ratio against P (%g pc), all %d clusters with rich kinematics:" % (LC0, len(rich)))
BINS = ((0.0, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.1), (1.1, 1.5), (1.5, 2.5), (2.5, 99.0))
for lo, hi_ in BINS:
    sel = sorted(r["ml"] for r in rich if lo <= r["P"][LC0] < hi_)
    if sel:
        q = np.percentile(sel, [25, 50, 75])
        out("    P in [%.1f, %.1f): N %3d, median %.2f, quartiles %.2f to %.2f" % (lo, hi_, len(sel), q[1], q[0], q[2]))
out("Systematics: the catalog snapshot has no metallicity column, and P tracks galactocentric radius, so the slope")
out("inherits any radial trend of the stellar-population ratio. C2 shows how much of the preference a radial trend")
out("of a given size absorbs. The rise that rescaling predicts across the sample is a factor %.1f at %g pc and %.1f at 200 pc." % (
    max(r["P"][LC0] for r in rich) / min(r["P"][LC0] for r in rich), LC0,
    max(r["P"][200.0] for r in rich) / min(r["P"][200.0] for r in rich)))
out()

out("C5. The outer-halo clusters (R_GC > 15 kpc) with rich kinematics outside the remote set, one by one,")
out("    anchored as in A1, errors raised to the floor, Delta lnL from the log-normal likelihood.")
sec = sorted([r for r in rich if r["rgc"] > 15 and r["name"] not in remote_names], key=lambda r: -r["P"][LC0])
tsec = {lc: 0.0 for lc in LCS}
for r in sec:
    e = max(r["dml"] / r["ml"], r["floor"]) / LN10
    var = SBASE ** 2 + e * e
    o = math.log10(r["ml"])
    cells = {}
    for lc in LCS:
        d = (-0.5 * (o - math.log10(YBASE)) ** 2 / var) - (-0.5 * (o - math.log10(YLC[lc] * r["P"][lc])) ** 2 / var)
        tsec[lc] += d
        cells[lc] = d
    out("    %-10s R_GC %5.1f  M/L %.2f +- %.2f  P %.2f: Newton %.2f, rescaling %.2f; Delta lnL %+6.2f (100 pc %+6.2f, 200 pc %+6.2f)" % (
        r["name"], r["rgc"], r["ml"], max(r["dml"], r["floor"] * r["ml"]), r["P"][LC0], YBASE, YLC[LC0] * r["P"][LC0],
        cells[LC0], cells[100.0], cells[200.0]))
out("    sum over %d: %+.1f at %g pc, %+.1f at 100 pc, %+.1f at 200 pc (49 pc: %+.1f)." % (
    len(sec), tsec[LC0], LC0, tsec[100.0], tsec[200.0], tsec[LC49]))
sparse = sorted([r for r in good if r["rgc"] > 15 and r["name"] not in remote_names and r not in rich_cat],
                key=lambda r: -r["P"][LC0])
out("    Not counted (fewer than 50 velocities and no proper motions, or a luminosity mass): " +
    ", ".join("%s (N_RV %d, M/L %.2f +- %.2f)" % (r["name"], r["nrv"], r["ml"], r["dml"]) for r in sparse) + ".")
out()

# =============================================================================================
# D. Clusters that carry no dynamical evidence
# =============================================================================================
out("D. CLUSTERS NOT COUNTED")
out("-" * 108)
nodisp = [n for n in remote_names if n not in KIN]
out("Remote clusters with no tabulated dispersion (their catalog ratios rest on star counts or the luminosity):")
for n in nodisp:
    r = byname[n]
    out("    %-15s N_RV %3d, catalog M/L %.2f +- %.2f, P(%g pc) %.2f" % (n, r["nrv"], r["ml"], r["dml"], LC0, r["P"][LC0]))
out("Seven remote clusters of the lowest fields, on their dispersions (%g pc: anchored, fitted, at 1.8):" % LC0)
for n in ("AM_1", "Eridanus", "Pal_3", "Pal_4", "Crater", "Pal_14", "Sagittarius_II"):
    if n not in KIN:
        out("    %-15s no tabulated dispersion: not evidence either way" % n)
        continue
    d = ROWA[n]["d"]
    out("    %-15s Y_kin %.2f (%.2f to %.2f) against Newton %.2f and rescaling %.2f: anchored %+.2f, fitted %+.2f, at 1.8 %+.2f" % (
        n, ROWA[n]["Ykin"][0], ROWA[n]["Ykin"][1], ROWA[n]["Ykin"][2], YBASE, YLC[LC0] * KIN[n]["r"]["P"][LC0],
        d["meas"], d["fit"], d["paper"]))
out()

# =============================================================================================
# E. Summary
# =============================================================================================
out("E. SUMMARY (positive favors Newton, the scale-selective coupling; single values at %g pc, ranges over %g to 200 pc)" % (LC0, WIN[0]))
out("-" * 108)
out("Remote clusters with dispersions, without NGC 2419 (%d clusters, Sagittarius II included):" % NSET)
out("    anchored (Y_N %.2f, Y_LC %.2f): %+.1f (window %+.1f to %+.1f); binaries to 1.0 km/s %+.1f (%+.1f to %+.1f)" % (
    (YBASE, YLC[LC0], A5[LC0]["anch"]) + wrange("anch") + (A5[LC0]["anchb"],) + wrange("anchb")))
out("    at 1.8: %+.1f (%+.1f to %+.1f); fitted, scatter s_base: %+.1f (%+.1f to %+.1f); fitted with binaries: %+.1f (%+.1f to %+.1f)" % (
    (A5[LC0]["paper"],) + wrange("paper") + (A5[LC0]["fit"],) + wrange("fit") + (A5[LC0]["fitb"],) + wrange("fitb")))
out("    49 pc: anchored %+.1f, at 1.8 %+.1f, fitted %+.1f, fitted with binaries %+.1f" % (
    A5[LC49]["anch"], A5[LC49]["paper"], A5[LC49]["fit"], A5[LC49]["fitb"]))
out("    fitted with the scatter free: %+.1f; with binaries %+.1f" % (FIT[(None, None)][4], FIT[(1.0, None)][4]))
out("    own star-count ratios, the six of A4: %+.1f (binaries %+.1f); with Sagittarius II: %+.1f (%+.1f)" % (
    tot6[None], tot6[1.0], tot7[None], tot7[1.0]))
s90 = A6[("all, as measured", LC0)]
out("    within-sample slope of Y_kin on P: %+.2f (%+.2f to %+.2f) at %g pc; %s; with binaries to 1.0 km/s %+.2f;" % (
    s90[0], s90[1], s90[2], LC0, ", ".join("%+.2f at %g pc" % (A6[("all, as measured", lc)][0], lc) for lc in (100.0, 200.0)),
    A6[("all, binaries to 1.0 km/s", LC0)][0]))
out("    without the five 'sc>' clusters %+.2f." % A6[("without the five 'sc>' clusters", LC0)][0])
out("NGC 2419 (Jeans normalization): anchored %+.1f / %+.1f / %+.1f at %g / 100 / 200 pc; at 1.8 %+.1f / %+.1f / %+.1f;" % (
    B2419[LC0][1], B2419[100.0][1], B2419[200.0][1], LC0, B2419[LC0][2], B2419[100.0][2], B2419[200.0][2]))
out("    rescaling needs a stellar ratio of %.2f to %.2f across the window." % (YJ / B2419[200.0][0], YJ / B2419[LC0][0]))
fa = CRES[("all", LC0)]
out("Rich kinematics, %d clusters: slope %+.2f (%+.2f to %+.2f) at %g pc against +1; Delta lnL %+.1f (%+.1f to %+.1f over" % (
    fa["n"], fa["b"], fa["blo"], fa["bhi"], LC0, fa["d"], min(CRES[("all", lc)]["d"] for lc in WIN),
    max(CRES[("all", lc)]["d"] for lc in WIN)))
out("    the window); beyond 8 kpc %+.1f (%+.1f to %+.1f)." % (CRES[("R_GC > 8 kpc", LC0)]["d"],
    min(CRES[("R_GC > 8 kpc", lc)]["d"] for lc in WIN), max(CRES[("R_GC > 8 kpc", lc)]["d"] for lc in WIN)))
out("    radial condition, all / beyond 8 kpc at %g pc: free power law %+.1f / %+.1f; cubic %+.1f / %+.1f;" % (
    LC0, SENS[("free power law in R_GC", "all", LC0)], SENS[("free power law in R_GC", "R8", LC0)],
    SENS[("cubic in log R_GC", "all", LC0)], SENS[("cubic in log R_GC", "R8", LC0)]))
out("    bounded to a factor 1.6 %+.1f / %+.1f; to a factor 2.7 %+.1f / %+.1f; MF slope %+.1f; MF, mass, Trh %+.1f;" % (
    SENS[("power law bounded, factor 1.6 (|c| <= 0.1)", "all", LC0)], SENS[("power law bounded, factor 1.6 (|c| <= 0.1)", "R8", LC0)],
    SENS[("power law bounded, factor 2.7 (|c| <= 0.2)", "all", LC0)], SENS[("power law bounded, factor 2.7 (|c| <= 0.2)", "R8", LC0)],
    SENS[("MF slope", "all", LC0)], SENS[("MF slope, log mass and lg Trh", "all", LC0)]))
out("    table-precision definition (%d clusters) %+.1f." % (len(prec_rich), SENS[("table precision <= 0.20 (%d clusters)" % len(prec_rich), "all", LC0)]))

with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(Lout) + "\n")

# ---------------------------------------------------------------- figure data
# One block per catchment length: kind, lc, name, P, Y, lo, hi, flag, extra (tab-separated).
with open(FIGDATA, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("# gc_dynamical.py figure data. kind: disp (dispersion-based ratio, remote cluster with tabulated bins),\n")
    fh.write("# jeans (NGC 2419 per-star Jeans ratio), rich (dynamical ratio resting on rich kinematics, NGC 2419 at its\n")
    fh.write("# Jeans ratio; flag outer for R_GC > 15 kpc), nodisp (no dispersion: catalog ratio from star counts or\n")
    fh.write("# luminosity), line (prediction constants), med (binned medians of the rich sample).\n")
    fh.write("# Columns: kind lc name P Y lo hi flag extra\n")
    for lc in LCS:
        for n in order:
            r = KIN[n]["r"]
            y0, ylo, yhi = ROWA[n]["Ykin"]
            if n == "NGC_2419":
                continue
            fh.write("disp\t%g\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%s\t%.4f\n" % (lc, n, r["P"][lc], y0, ylo, yhi, prov(r), r["ml"]))
        fh.write("jeans\t%g\tNGC_2419\t%.4f\t%.4f\t%.4f\t%.4f\trich\t%.4f\n" % (lc, byname["NGC_2419"]["P"][lc], YJ, 10 ** jlo, 10 ** jhi,
                                                                            byname["NGC_2419"]["ml"]))
        for r in rich:
            err = max(r["dml"] / r["ml"], r["floor"])
            flag = "outer" if r["rgc"] > 15 else "inner"
            fh.write("rich\t%g\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%s\t%.2f\n" % (lc, r["name"], r["P"][lc], r["ml"], r["ml"] / (1 + err),
                                                                       r["ml"] * (1 + err), flag, r["rgc"]))
        for n in nodisp:
            r = byname[n]
            fh.write("nodisp\t%g\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%s\t%.2f\n" % (lc, n, r["P"][lc], r["ml"], r["ml"] - r["dml"], r["ml"] + r["dml"],
                                                                         "sc", r["rgc"]))
        for lo, hi_ in BINS:
            sel = [r for r in rich if lo <= r["P"][lc] < hi_]
            if sel:
                q = np.percentile([r["ml"] for r in sel], [25, 50, 75])
                fh.write("med\t%g\t%.1f-%.1f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t0\n" % (lc, lo, hi_, float(np.median([r["P"][lc] for r in sel])),
                                                                             q[1], q[0], q[2], len(sel)))
        fz = CRES[("all", lc)]
        fh.write("line\t%g\tY_base\t0\t%.4f\t0\t0\t-\t0\n" % (lc, YBASE))
        fh.write("line\t%g\tY_LC\t0\t%.4f\t0\t0\t-\t0\n" % (lc, YLC[lc]))
        fh.write("line\t%g\tY_paper\t0\t%.4f\t0\t0\t-\t0\n" % (lc, YPAPER))
        fh.write("line\t%g\trich_Y0\t0\t%.4f\t0\t0\t-\t0\n" % (lc, fz["Y0"]))
        fh.write("line\t%g\trich_Y1\t0\t%.4f\t0\t0\t-\t0\n" % (lc, fz["Y1"]))
        fh.write("line\t%g\trich_slope\t0\t%.4f\t%.4f\t%.4f\t-\t%.4f\n" % (lc, fz["b"], fz["blo"], fz["bhi"], fz["d"]))
print("wrote globular-clusters/gc_dynamical-results.txt and globular-clusters/gc_dynamical-figdata.tsv")

# the Part C sample with the boost at every catchment length, the input of globular-clusters/gc_radial_trends.py
with open(RICH_SAMPLE, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("name\trgc\tml\tdml\tfloor\tmass\ttrh\tmf\t" + "\t".join("P%g" % lc for lc in LCS) + "\n")
    for r in rich:
        fh.write("%s\t%.6g\t%.8g\t%.8g\t%.8g\t%.6g\t%.4f\t%.4f\t" % (r["name"], r["rgc"], r["ml"], r["dml"], r["floor"],
                 r["mass"], r["trh"], r["mf"]) + "\t".join("%.10g" % r["P"][lc] for lc in LCS) + "\n")

