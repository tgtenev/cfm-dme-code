"""The direction of a new layer, tested on the innermost points of strongly boosted SPARC galaxies against their
speeds through the fabric (Paper II, section 3.1 and Table 1).

Paper II section 3.1 adopts as an identification that each smoothing lays its new layer across the transmitted flux,
and this test selects that rule over laying the layer across the local interior gradient of the freshly smoothed
domain. The two are called the flux rule and the gradient rule here. A body that moves at speed v through the fabric
leaves its stored axes trailing by
    Delta = F 2 v T nu^p,    T = (1/Gamma) (nu - 1 + delta)/delta,
with T the mean interval between smoothings at stretch nu and 1/Gamma the mean interval between renewal events at a
domain (Paper II Eq. (8)). 1/Gamma = delta tau_eq spans 190 to 650 yr over the tightened kinetic window of Paper II
section 4.1, which gives T = 1,200 to 4,100 yr at the crossover field, as section 3.1 states. The flux rule has p = 0
and F = 0.85 to 0.92 (Paper II Appendix A.3). The gradient rule has p = 1 and F = 1.0 to 2.6, its logarithmic factor of
section 3.1 inside a body whose arrangement ends at 5.4 to 100 body radii. Both factors are the first-order solutions
of kinetics/steady_dipole.py. Under the gradient rule the boost at the innermost points of strongly boosted galaxies
falls where the trail nears the radius, so their residuals from the radial-acceleration relation would fall with the
speed and with the trail. The flux rule predicts no such fall.

Steps.
  1. The fit. The boost law of Paper I, Eq. (8) (code T3 in refit_core.py), at delta = 0.19 and a0 = 0.976e-10 m/s^2.
     The free volume is the constrained fit's, rounded to 0.19 as Paper I does for derived quantities, and the scale
     lies between the constrained fit's best scales of 0.973e-10 and 0.981e-10 m/s^2 at H0 = 67.4 and 73 km/s/Mpc
     (Paper I section 5.3). The law acts on SPARC's tabulated baryonic field. Every galaxy's mass-to-light ratios,
     distance and inclination are refitted by the fit machinery of refit_core.py, over all points with a positive
     baryonic field. The residual at a point is log10(g_obs/q^2) - log10(g_law).
  2. The sample. The galaxies whose innermost valid point the law boosts by more than 5 (the wide sample, the 32
     strongly boosted galaxies of Paper II) and, among them, those boosted by more than 10, the boost being
     nu(g_bar/a0) at the fitted ratios.
  3. The speeds. The heliocentric velocity and position of each galaxy, read from the replies in the data folder,
     converted to the frame of the microwave background with the Sun's motion of 369.82 km/s toward
     (l, b) = (264.021, 48.253) deg (Planck 2018). The Hubble flow at SPARC's distance is subtracted with
     H0 = 73 km/s/Mpc, the scale of SPARC's flow distances. That leaves the line-of-sight component u_los of the
     galaxy's motion through the fabric, with the error H0 e_D. The transverse component is not measured. The flow
     speed takes it equal to the transverse part of the Local Group's motion, 627 km/s toward (276, 30) deg (Kogut et
     al. 1993), and a Monte Carlo adds a relative transverse velocity drawn isotropically with 200 km/s per component.
     The line-of-sight component alone is a lower bound on the speed.
  4. The trails. Delta / r at the innermost point under each rule, at the flow speed, at the line-of-sight component
     and at the Monte Carlo's 16th and 84th percentiles, over the ranges of 1/Gamma and F.
  5. The boost left. boost_left.py: a sphere with the galaxy's baryons, the static aligned arrangement of the law
     about it, each shell displaced by its trail, the field solved with that tensor; the boost at every data radius
     over the undisplaced one, in the plane of motion and normal to it.
  6. The tests. (a) Spearman's rank correlation of the residual at the innermost point, and of the mean of the two
     innermost points, with the speed, with its line-of-sight component, with the gradient rule's trail and with two
     controls, with permutation p-values. (b) The correlation that each case of the gradient rule would itself
     produce, its predicted residuals plus noise, the noise either the observed residuals reshuffled or Gaussian with
     each point's sigma. (c) The weighted slope of the residual on each case's predicted residual, which the gradient
     rule requires to be 1 and the flux rule 0. (d) A level test on the innermost points, chi^2 with the scatter set
     so that the unmodified law's chi^2 per point is one. Then a refit of every galaxy of the wide sample with each
     rule's boost left at all its radii, Delta chi^2 against the unmodified law.
  7. Variants: the stretch that the observed field gives, nu(X_obs), in place of the law's, for the trails; the rigid
     displacement for the boost left; H0 = 67.4 km/s/Mpc and a relative transverse velocity of 100 and 300 km/s per
     component for the speeds.

The numbers of Paper II section 3.1 and Table 1, by section of the results file:
  1: the 32 galaxies of the wide sample and the 9 boosted by more than 10; the innermost errors sigma_obs of
     UGCA 444, DDO 64 and NGC 3109, 0.351, 0.639 and 0.232 dex.
  2: the flow speeds of 515 to 694 km/s, a factor of 1.35.
  3: at the flow speed, the flux rule's trail below the radius at every innermost point, and the gradient rule's trail
     above a tenth of the radius at 22 to 32 of the 32 galaxies and above the radius at 3 to 20.
  4: the boost left by the flux rule, 91 percent or more in its central case (flux-central, -0.042 dex at worst, at
     UGCA 444) and at least 69 percent in its most exposed case (flux-most, -0.163 dex at DDO 64).
  5: the slope of +0.143 +/- 0.197 on the gradient rule's central prediction (c, boost above 5), against the 1 that
     rule requires and the 0 of the flux rule. With the level test (d) and the correlations that rule would give (b)
     it bears the three to four standard deviations. The gradient rule's least cases fail only on the nine galaxies
     boosted by more than 10 (b).

Inputs: data/sparc/ (through aqual-refit-fit.py, with this folder's aqual-refit-gamma-*.csv and aqual-disk-points.csv)
and the data folder given as the first argument, data/sparc-velocities-2026-11-22. That folder is not redistributed.
The reader fetches it with fetch_sesame.py, fetch_vizier.py, fetch_vizier_hi.py and fetch_retry.py, in that order.
Its four manifests, sesame-manifest.tsv, vizier-manifest.tsv, vizier-hi-manifest.tsv and vizier-retry-manifest.tsv,
list every query with its address, time and the SHA-256 of its reply, and the folder's SHA256SUMS gives the
checksums of the replies that Paper II used. The script reads the replies as text only.
Output: rotation-curves/layer_direction_test-results.txt.

Run from the repository root (six to eight minutes; numpy and scipy):
    python rotation-curves/layer_direction_test.py data/sparc-velocities-2026-11-22 --out rotation-curves/layer_direction_test-results.txt
"""
import importlib.util
import math
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))                            # the repository root
sys.path.insert(0, HERE)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
_saved = sys.argv
sys.argv = [sys.argv[0]]                       # the refit modules read options from sys.argv at import
import refit_core as RC                        # noqa: E402
_spec = importlib.util.spec_from_file_location("aqual_refit_fit", os.path.join(HERE, "aqual-refit-fit.py"))
AF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AF)
sys.argv = _saved
import boost_left as BL                        # noqa: E402

DELTA, A0 = 0.19, 0.976e-10                    # the free volume and the scale of the fit (step 1 of the header)
LAM = math.log1p(DELTA)
T_EVENT = (190.0, 650.0)                       # yr, 1/Gamma over the tightened kinetic window (Paper II section 4.1)
RULE = {"grad": "the gradient rule", "flux": "the flux rule"}
FAC = {"grad": (1.0, 2.6), "flux": (0.85, 0.92)}   # the first-order factors F (kinetics/steady_dipole.py)
POW = {"grad": 1, "flux": 0}
PC_PER_YR_KMS = BL.PC_PER_YR_KMS
C_KMS = 299792.458
V_SUN, L_SUN, B_SUN = 369.82, 264.021, 48.253  # the Sun in the frame of the microwave background (Planck 2018)
V_LG, L_LG, B_LG = 627.0, 276.0, 30.0          # the Local Group in that frame (Kogut et al. 1993)
H0 = 73.0                                      # km/s/Mpc, the scale of SPARC's flow distances
SIG_T = 200.0                                  # km/s per component, the relative transverse velocity
NMC = 20000
NPERM = 20000
KGRID = 10.0 ** np.arange(3.3, 6.61, 0.3)      # the displacement scale K = v F/Gamma, km/s yr (boost_left.py)
THRESH = (10.0, 5.0)
SEED = 20261122
OUT = os.path.abspath(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else None
_lines = []


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


# ---------------------------------------------------------------- the law
def nu_of_X(X):
    """The stretch at the total field X (units of a0): 1/mu(X) = 1 + delta/(exp(lambda X) - 1)."""
    return 1.0 + DELTA / np.expm1(LAM * np.maximum(X, 1e-300))


def interval(nu, t_event):
    """T, the mean interval between smoothings at stretch nu, for a mean interval t_event = 1/Gamma between events."""
    return t_event * (nu - 1.0 + DELTA) / DELTA


def trail_ratio(v, nu, r_kpc, t_event, F, p):
    return F * 2.0 * v / PC_PER_YR_KMS * interval(nu, t_event) * nu ** p / (r_kpc * 1000.0)


# ---------------------------------------------------------------- the data folder
def sparc_fd(path):
    """SPARC Table 1: distance method f_D per galaxy (1 Hubble flow, 2 TRGB, 3 Cepheids, 4 Ursa Major, 5 SN)."""
    fd, rule = {}, 0
    for line in open(path, encoding="ascii", errors="replace"):
        s = line.rstrip("\r\n")
        if re.fullmatch(r"-+", s):
            rule += 1
            continue
        if rule < 4:
            continue
        f = s.split()
        if len(f) >= 18 and re.fullmatch(r"\d+", f[17]):
            fd[f[0]] = int(f[4])
    return fd


def read_tsv_rows(path, ncol):
    rows, started = [], False
    for ln in open(path, encoding="utf-8", errors="replace").read().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        if re.match(r"^-+\t", ln) or re.fullmatch(r"[-\t]+", ln):
            started = True
            continue
        if started:
            f = ln.split("\t")
            if len(f) >= ncol:
                rows.append([x.strip() for x in f])
    return rows


def read_velocities(dd):
    """Per galaxy: ra, dec, heliocentric velocity, its error, its source; and the catalog checks."""
    G = {}
    man = open(os.path.join(dd, "sesame-manifest.tsv"), encoding="utf-8").read().splitlines()[1:]
    for line in man:
        f = line.split("\t")
        gal, path = f[0], f[4]
        txt = open(os.path.join(dd, path), encoding="utf-8", errors="replace").read()
        g = G.setdefault(gal, dict(ra=None, dec=None, v=None, ev=None, src=None, oname=None))
        for block in re.findall(r"<Resolver name=\"([^\"]*)\">(.*?)</Resolver>", txt, flags=re.S):
            rname, body = block
            ra = re.search(r"<jradeg>\s*([-+0-9.]+)\s*</jradeg>", body)
            de = re.search(r"<jdedeg>\s*([-+0-9.]+)\s*</jdedeg>", body)
            on = re.search(r"<oname>\s*(.*?)\s*</oname>", body)
            if ra and de and g["ra"] is None:
                g["ra"], g["dec"] = float(ra.group(1)), float(de.group(1))
                g["oname"] = on.group(1) if on else None
            if g["v"] is None:
                mv = re.search(r"<Vel>(.*?)</Vel>", body, flags=re.S)
                mz = re.search(r"<z>(.*?)</z>", body, flags=re.S)
                if mv:
                    v = re.search(r"<v>\s*([-+0-9.eE]+)\s*</v>", mv.group(1))
                    e = re.search(r"<e>\s*([-+0-9.eE]+)\s*</e>", mv.group(1))
                    r = re.search(r"<r>\s*(.*?)\s*</r>", mv.group(1))
                    if v:
                        g["v"] = float(v.group(1))
                        g["ev"] = float(e.group(1)) if e else None
                        g["src"] = "Sesame " + rname.split("=")[0] + (" " + r.group(1) if r else "")
                elif mz:
                    v = re.search(r"<v>\s*([-+0-9.eE]+)\s*</v>", mz.group(1))
                    e = re.search(r"<e>\s*([-+0-9.eE]+)\s*</e>", mz.group(1))
                    r = re.search(r"<r>\s*(.*?)\s*</r>", mz.group(1))
                    if v:
                        g["v"] = C_KMS * float(v.group(1))
                        g["ev"] = C_KMS * float(e.group(1)) if e else None
                        g["src"] = "Sesame " + rname.split("=")[0] + " z" + (" " + r.group(1) if r else "")
    for man_name in ("vizier-manifest.tsv", "vizier-hi-manifest.tsv", "vizier-retry-manifest.tsv"):
        vman = os.path.join(dd, man_name)
        if not os.path.exists(vman):
            continue
        for line in open(vman, encoding="utf-8").read().splitlines()[1:]:
            f = line.split("\t")
            gal, cat, path = f[0], f[1], f[3]
            g = G[gal]
            if cat.startswith("J/AJ/145/101"):
                rows = read_tsv_rows(os.path.join(dd, path), 8)
                if rows and rows[0][4]:
                    g["ungc"] = dict(r=float(rows[0][0]), name=rows[0][1], v=float(rows[0][4]),
                                     D=float(rows[0][5]) if rows[0][5] else None, fD=rows[0][6])
            elif cat.startswith("J/ApJ/861/49"):
                rows = read_tsv_rows(os.path.join(dd, path), 11)
                if rows and rows[0][7]:
                    g["alfalfa"] = dict(r=float(rows[0][0]), agc=rows[0][1], name=rows[0][2], v=float(rows[0][7]),
                                        w50=float(rows[0][8]) if rows[0][8] else None)
            elif cat.startswith("VII/238"):
                rows = read_tsv_rows(os.path.join(dd, path), 7)
                rows = [r for r in rows if r[5]]
                if rows:
                    g["ledahi"] = dict(r=float(rows[0][0]), pgc=rows[0][1], v=float(rows[0][5]),
                                       ev=float(rows[0][6]) if rows[0][6] else 10.0)
    for gal, g in G.items():
        if g["v"] is None and "alfalfa" in g:
            g["v"], g["ev"], g["src"] = g["alfalfa"]["v"], 5.0, "ALFALFA AGC " + g["alfalfa"]["agc"]
        if g["v"] is None and "ledahi" in g:
            g["v"], g["ev"], g["src"] = g["ledahi"]["v"], g["ledahi"]["ev"], "VII/238 PGC " + g["ledahi"]["pgc"]
        if g["v"] is None and "ungc" in g:
            g["v"], g["ev"], g["src"] = g["ungc"]["v"], 5.0, "UNGC " + g["ungc"]["name"]
    return G


# ---------------------------------------------------------------- the frame
EQ2GAL = np.array([[-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
                   [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
                   [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669]])


def unit_lb(l_deg, b_deg):
    l, b = math.radians(l_deg), math.radians(b_deg)
    return np.array([math.cos(b) * math.cos(l), math.cos(b) * math.sin(l), math.sin(b)])


def radec_to_lb(ra, dec):
    a, d = math.radians(ra), math.radians(dec)
    x = EQ2GAL @ np.array([math.cos(d) * math.cos(a), math.cos(d) * math.sin(a), math.sin(d)])
    return math.degrees(math.atan2(x[1], x[0])) % 360.0, math.degrees(math.asin(max(-1.0, min(1.0, x[2]))))


def speeds(g, D, eD, h0=H0, sig_t=SIG_T, rng=None):
    """Line-of-sight component, the flow speed and the Monte Carlo of the speed through the fabric."""
    l, b = radec_to_lb(g["ra"], g["dec"])
    rhat = unit_lb(l, b)
    vsun = V_SUN * unit_lb(L_SUN, B_SUN)
    ulg = V_LG * unit_lb(L_LG, B_LG)
    cz_cmb = g["v"] + float(vsun @ rhat)
    u_los = cz_cmb - h0 * D
    s_los = math.sqrt((h0 * eD) ** 2 + (g["ev"] or 5.0) ** 2)
    lg_los = float(ulg @ rhat)
    lg_perp = ulg - lg_los * rhat
    v_flow = math.sqrt(u_los ** 2 + float(lg_perp @ lg_perp))
    # Monte Carlo: u_los with its error; transverse = LG's transverse part plus an isotropic relative velocity
    e1 = np.cross(rhat, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(rhat, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(rhat, e1)
    p1, p2 = float(lg_perp @ e1), float(lg_perp @ e2)
    ul = rng.normal(u_los, s_los, NMC)
    w1, w2 = rng.normal(0.0, sig_t, NMC), rng.normal(0.0, sig_t, NMC)
    vmc = np.sqrt(ul ** 2 + (p1 + w1) ** 2 + (p2 + w2) ** 2)
    return dict(l=l, b=b, cz_cmb=cz_cmb, u_los=u_los, s_los=s_los, lg_los=lg_los, w_los=u_los - lg_los,
                lg_perp=math.sqrt(p1 * p1 + p2 * p2), v_flow=v_flow, v_16=float(np.percentile(vmc, 16)),
                v_50=float(np.percentile(vmc, 50)), v_84=float(np.percentile(vmc, 84)))


# ---------------------------------------------------------------- statistics
def rankdata(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(1, len(x) + 1)
    # average ties
    xs = x[o]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / math.sqrt((rx * rx).sum() * (ry * ry).sum()))


def perm_p(x, y, rng, n=NPERM):
    """Spearman's rho, its two-sided permutation p-value, and the one-sided p of rho at or below the observed."""
    r0 = spearman(x, y)
    rx = rankdata(x); ry = rankdata(y)
    rx -= rx.mean(); ry -= ry.mean()
    den = math.sqrt((rx * rx).sum() * (ry * ry).sum())
    sims = np.array([(rx * ry[rng.permutation(len(ry))]).sum() / den for _ in range(n)])
    return r0, float(np.mean(np.abs(sims) >= abs(r0) - 1e-12)), float(np.mean(sims <= r0 + 1e-12))


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
        sys.exit(__doc__)
    dd = os.path.abspath(sys.argv[1])
    os.chdir(ROOT)
    rng = np.random.default_rng(SEED)
    T1, gals = AF.load()
    FD = sparc_fd("data/sparc/SPARC_Lelli2016c.mrt")
    VEL = read_velocities(dd)
    ds = RC.Dataset(gals)
    ctx = AF.ctx_of("sparc", "T3", delta=DELTA, use_gok=False)
    ctx["a0"] = A0
    res = ds.fit(ctx)
    pts = ds.points(ctx, res)
    nu_fun = RC.law_nu("T3", DELTA)
    out("THE DIRECTION OF A NEW LAYER AGAINST THE INNER ROTATION CURVES AND THE SPEEDS THROUGH THE FABRIC")
    out("=" * 118)
    out(f"law: the boost law of Paper I, Eq. (8), at delta = {DELTA}, a0 = {A0:.4g} m/s^2, on SPARC's tabulated field; the fit")
    out(f"machinery of refit_core.py refits every galaxy's ratios, distance and inclination over all points with g_bar > 0.")
    out(f"{len(ds.names)} galaxies, total chi^2 {res['F'].sum():.2f}.")
    out(f"trail: Delta = F 2 v T nu^p, T = (1/Gamma) (nu - 1 + delta)/delta, 1/Gamma the mean interval between renewal events,")
    out(f"{T_EVENT[0]:.0f} to {T_EVENT[1]:.0f} yr; the gradient rule p = 1, F {FAC['grad'][0]} to {FAC['grad'][1]}; "
        f"the flux rule p = 0, F {FAC['flux'][0]} to {FAC['flux'][1]}.")
    out(f"frame: the Sun {V_SUN} km/s toward (l, b) = ({L_SUN}, {B_SUN}); the Local Group {V_LG:.0f} km/s toward ({L_LG:.0f}, {B_LG:.0f});")
    out(f"H0 = {H0} km/s/Mpc for the Hubble flow at SPARC's distance; relative transverse velocity {SIG_T:.0f} km/s per component.")

    # --- per-galaxy quantities at the innermost points
    info = {}
    for i, n in enumerate(ds.names):
        p = pts[n]
        k = np.nonzero(p["valid"])[0]
        if len(k) < 2:
            continue
        q = float(res["q"][i])
        row = dict(name=n, i=i, k=k, q=q, Yd=10 ** float(res["u"][i]), Yb=(10 ** float(res["v"][i])) if gals[n]["hasb"] else 0.0,
                   D=gals[n]["T1"]["D"], eD=gals[n]["T1"]["eD"], fD=FD[n], R=p["R"], gbar=p["gbar"], r=p["r"],
                   Vo=p["Vo"], eV=gals[n]["eV"])
        sig = 2.0 / math.log(10.0) * np.sqrt(gals[n]["eV"] ** 2 + (0.05 * p["Vo"]) ** 2) / np.maximum(p["Vo"], 1e-6)
        row["sig"] = sig
        row["nu_alg"] = nu_fun(np.maximum(p["gbar"], 1e-300) / A0)
        Xobs = p["Vo"] ** 2 / p["R"] * RC.CONV / q ** 2 / A0
        row["nu_obs"] = nu_of_X(Xobs)
        info[n] = row
    samples, dropped = {}, []
    for th in THRESH:
        sel = sorted([n for n, rw in info.items() if rw["nu_alg"][rw["k"][0]] > th])
        dropped += [n for n in sel if VEL.get(n, {}).get("v") is None and n not in dropped]
        samples[th] = [n for n in sel if VEL.get(n, {}).get("v") is not None]
    wide = samples[5.0]
    strong = samples[10.0]
    out("")
    out("1. THE SAMPLE: THE LAW'S BOOST nu(g_bar/a0) AT THE INNERMOST VALID POINT, AT THE FITTED RATIOS")
    out(f"   boost above 10: {len(strong)} galaxies; above 5: {len(wide)} galaxies (of {len(info)} with two or more valid points)")
    if dropped:
        out(f"   left out for want of a velocity: {', '.join(dropped)}")
    out("   galaxy        f_D  D (Mpc)   r_in (kpc)  boost   nu(X_obs)  residual  sigma_obs | 2nd point: r, boost, residual")
    for n in wide:
        rw = info[n]; j0, j1 = rw["k"][0], rw["k"][1]
        out(f"   {n:12s} {rw['fD']:3d} {rw['D']:7.2f} +/- {rw['eD']:5.2f} {rw['R'][j0]:6.2f}  {rw['nu_alg'][j0]:7.2f} {rw['nu_obs'][j0]:8.2f}"
            f"   {rw['r'][j0]:+.3f}   {rw['sig'][j0]:.3f}   | {rw['R'][j1]:5.2f} {rw['nu_alg'][j1]:6.2f} {rw['r'][j1]:+.3f}"
            f"{'   *' if n in strong else ''}")
    out("   (* boost above 10)")

    # --- velocities and speeds
    out("")
    out("2. THE SPEEDS THROUGH THE FABRIC (km/s)")
    out("   v_hel and its source; cz_CMB; u_los = cz_CMB - H0 D with its error H0 e_D (+) the velocity error; the Local Group's")
    out("   line-of-sight and transverse parts there; w_los = u_los minus the Local Group's line-of-sight part; the flow speed")
    out("   (transverse part equal to the Local Group's); the Monte Carlo's 16th, 50th and 84th percentiles.")
    out("   galaxy        v_hel   cz_CMB   u_los +/- err   LG_los LG_perp   w_los  v_flow  v_16  v_50  v_84  source of v_hel")
    for n in wide:
        g = VEL[n]
        rw = info[n]
        sp = speeds(g, rw["D"], rw["eD"], rng=rng)
        rw["sp"] = sp
        out(f"   {n:12s} {g['v']:7.1f} {sp['cz_cmb']:8.1f} {sp['u_los']:7.0f} +/- {sp['s_los']:4.0f} {sp['lg_los']:7.0f} {sp['lg_perp']:6.0f}"
            f"  {sp['w_los']:6.0f} {sp['v_flow']:6.0f} {sp['v_16']:5.0f} {sp['v_50']:5.0f} {sp['v_84']:5.0f}  {g['src']}")
    wl = np.array([info[n]["sp"]["w_los"] for n in wide])
    sl = np.array([info[n]["sp"]["s_los"] for n in wide])
    fd_ok = np.array([info[n]["fD"] in (2, 3, 5) for n in wide])
    exc = lambda a, s: math.sqrt(max(float(np.mean(a ** 2) - np.mean(s ** 2)), 0.0))
    out(f"   relative line-of-sight velocity w_los over the wide sample: rms {math.sqrt(float(np.mean(wl ** 2))):.0f} km/s, "
        f"rms after the errors {exc(wl, sl):.0f} km/s;")
    out(f"   over the {int(fd_ok.sum())} galaxies with TRGB, Cepheid or supernova distances: rms {math.sqrt(float(np.mean(wl[fd_ok] ** 2))):.0f} km/s, "
        f"after the errors {exc(wl[fd_ok], sl[fd_ok]):.0f} km/s")
    for fdv, fdl in ((2, "TRGB"), (3, "Cepheids"), (5, "supernovae"), (4, "Ursa Major"), (1, "Hubble flow")):
        e = [info[n]["sp"]["s_los"] for n in wide if info[n]["fD"] == fdv]
        if e:
            out(f"   error of u_los, distances from {fdl}: {len(e)} galaxies, {min(e):.0f} to {max(e):.0f} km/s")
    srcs = {}
    for n in info:
        srcs[VEL[n]["src"].split()[0]] = srcs.get(VEL[n]["src"].split()[0], 0) + 1
    out("   source of the heliocentric velocity over the 153 galaxies of the fit: " + ", ".join(f"{k} {v}" for k, v in sorted(srcs.items())))
    vf = np.array([info[n]["sp"]["v_flow"] for n in wide])
    ul = np.array([abs(info[n]["sp"]["u_los"]) for n in wide])
    p16 = np.array([info[n]["sp"]["v_16"] for n in wide]); p84 = np.array([info[n]["sp"]["v_84"] for n in wide])
    out(f"   Monte Carlo over the wide sample: 16th percentiles {p16.min():.0f} to {p16.max():.0f} km/s, 84th percentiles {p84.min():.0f} to "
        f"{p84.max():.0f} km/s")
    out(f"   flow speed over the wide sample: {vf.min():.0f} to {vf.max():.0f} km/s, median {np.median(vf):.0f}; |u_los| {ul.min():.0f} to "
        f"{ul.max():.0f}, median {np.median(ul):.0f}")
    for key, lab in (("ungc", "the Updated Nearby Galaxy Catalog's"), ("ledahi", "the homogenized HI velocity of VII/238")):
        for grp_lab, names in (("all galaxies", list(info)), ("the wide sample", wide)):
            chk = [(n, VEL[n]["v"], VEL[n][key]["v"]) for n in names if VEL[n].get(key) and VEL[n]["src"].startswith("Sesame")]
            if chk:
                d = np.array([a - b for _, a, b in chk])
                j = int(np.argmax(np.abs(d)))
                out(f"   check, {grp_lab}: the resolver's velocity minus {lab} for {len(chk)} galaxies: median {np.median(d):+.1f} km/s, "
                    f"within 20 km/s for {int((np.abs(d) <= 20).sum())}, largest {d[j]:+.1f} km/s ({chk[j][0]})")

    # --- trails at the innermost point
    out("")
    out("3. THE TRAIL AT THE INNERMOST POINT, OVER THE RADIUS (each range over 1/Gamma and F; the law's stretch)")
    out("   galaxy        r_in  nu     T (kyr)    | gradient rule at v_flow, at |u_los|, at v_16-v_84 | flux rule at v_flow, at v_84")
    for n in wide:
        rw = info[n]; j0 = rw["k"][0]; sp = rw["sp"]
        r, nu = rw["R"][j0], rw["nu_alg"][j0]
        f = lambda v, ru, lo: trail_ratio(v, nu, r, T_EVENT[0] if lo else T_EVENT[1], FAC[ru][0] if lo else FAC[ru][1], POW[ru])
        rw["tr"] = {ru: dict(flow=(f(sp["v_flow"], ru, True), f(sp["v_flow"], ru, False)),
                             los=(f(abs(sp["u_los"]), ru, True), f(abs(sp["u_los"]), ru, False)),
                             mc=(f(sp["v_16"], ru, True), f(sp["v_84"], ru, False))) for ru in ("grad", "flux")}
        tb, tc = rw["tr"]["grad"], rw["tr"]["flux"]
        out(f"   {n:12s} {r:5.2f} {nu:6.2f} {interval(nu, T_EVENT[0]) / 1e3:5.1f}-{interval(nu, T_EVENT[1]) / 1e3:5.1f} | "
            f"{tb['flow'][0]:6.3f}-{tb['flow'][1]:6.3f} {tb['los'][0]:6.3f}-{tb['los'][1]:6.3f} {tb['mc'][0]:6.3f}-{tb['mc'][1]:6.3f} | "
            f"{tc['flow'][0]:.4f}-{tc['flow'][1]:.4f} {tc['mc'][1]:.4f}")
    for lab, key in (("at the flow speed", "flow"), ("at the line-of-sight component", "los")):
        for ru in ("grad", "flux"):
            lo = np.array([info[n]["tr"][ru][key][0] for n in wide]); hi = np.array([info[n]["tr"][ru][key][1] for n in wide])
            los = np.array([info[n]["tr"][ru][key][0] for n in strong]); his = np.array([info[n]["tr"][ru][key][1] for n in strong])
            out(f"   {RULE[ru]} {lab}: wide sample, trail above a tenth of the radius {int((lo > 0.1).sum())} to {int((hi > 0.1).sum())} of "
                f"{len(wide)}, above the radius {int((lo > 1).sum())} to {int((hi > 1).sum())}; boost above 10: above a tenth "
                f"{int((los > 0.1).sum())} to {int((his > 0.1).sum())} of {len(strong)}, above the radius {int((los > 1).sum())} to {int((his > 1).sum())}")
    out("   the same with the stretch that the observed field gives, nu(X_obs), at the flow speed:")
    for ru in ("grad", "flux"):
        lo, hi = [], []
        for n in wide:
            rw = info[n]; j0 = rw["k"][0]; v = rw["sp"]["v_flow"]
            lo.append(trail_ratio(v, rw["nu_obs"][j0], rw["R"][j0], T_EVENT[0], FAC[ru][0], POW[ru]))
            hi.append(trail_ratio(v, rw["nu_obs"][j0], rw["R"][j0], T_EVENT[1], FAC[ru][1], POW[ru]))
        lo, hi = np.array(lo), np.array(hi)
        out(f"   {RULE[ru]}: above a tenth of the radius {int((lo > 0.1).sum())} to {int((hi > 0.1).sum())} of {len(wide)}, above the radius "
            f"{int((lo > 1).sum())} to {int((hi > 1).sum())}; median {np.median(lo):.4f} to {np.median(hi):.4f}")

    # --- the boost left
    out("")
    out("4. THE BOOST LEFT (boost_left.py: sheared arrangement; in the plane of motion / normal to it), solved on a grid of")
    out("   the displacement scale K = v F/Gamma (km/s yr) and interpolated in log K. Static check: the undisplaced solve against")
    out("   the spherical law.")
    BLT = {}
    for n in wide:
        rw = info[n]; k = rw["k"]
        R = rw["R"][k[0]:]
        y = np.maximum(rw["gbar"][k[0]:] / A0, 1e-9)
        S = BL.GalaxySphere(R, y, DELTA, dxi=0.01, nt=128)
        ci, _ = S.static_check()
        tab = {}
        for ru in ("grad", "flux"):
            bi = np.zeros((len(KGRID), len(R))); bn = np.zeros_like(bi)
            for a, kd in enumerate(KGRID):
                bi[a], bn[a] = S.boost_left(kd, POW[ru])
            tab[ru] = (bi, bn)
        ri = np.zeros(len(KGRID)); rn = np.zeros(len(KGRID))
        for a, kd in enumerate(KGRID):
            x1, x2 = S.boost_left(kd, 1, rigid=True)
            ri[a], rn[a] = x1[0], x2[0]
        tab["grad-rigid"] = (ri, rn)
        BLT[n] = dict(S=S, tab=tab, R=R, k0=k[0], static=ci)
        out(f"   {n:12s} static check at the innermost point {ci[0]:+.4f}, largest over its radii {np.abs(ci).max():.4f}; "
            f"boost left at the innermost point, gradient rule, K = 1e4, 1e5, 1e6: "
            + ", ".join(f"{np.interp(math.log10(kk), np.log10(KGRID), tab['grad'][0][:, 0]):.2f}/{np.interp(math.log10(kk), np.log10(KGRID), tab['grad'][1][:, 0]):.2f}"
                        for kk in (1e4, 1e5, 1e6)))

    out(f"   static check over the wide sample: largest deviation {max(float(np.abs(BLT[n]['static']).max()) for n in wide):.4f} at the "
        f"data radii and {max(abs(float(BLT[n]['static'][0])) for n in wide):.4f} at the innermost points")

    def boost_at(n, ru, kd, which, rigid=False):
        """Boost left at all of a galaxy's radii from its innermost valid point (which: 0 in the plane, 1 normal)."""
        T = BLT[n]["tab"]["grad-rigid" if rigid else ru]
        lk = math.log10(max(kd, 1e-9))
        L = np.log10(KGRID)
        if rigid:
            vals = T[which]
            if lk < L[0]:
                return 1.0 + (vals[0] - 1.0) * kd / KGRID[0]
            return float(np.interp(lk, L, vals))
        M = T[which]
        if lk < L[0]:
            return 1.0 + (M[0] - 1.0) * kd / KGRID[0]
        return np.array([np.interp(lk, L, M[:, j]) for j in range(M.shape[1])])

    def kscale_of(v, ru, corner):
        """K = v F/Gamma at a corner: 'least' (190 yr, F low), 'mid' (geometric means), 'most' (650 yr, F high)."""
        if corner == "least":
            return v * T_EVENT[0] * FAC[ru][0]
        if corner == "most":
            return v * T_EVENT[1] * FAC[ru][1]
        return v * math.sqrt(T_EVENT[0] * T_EVENT[1]) * math.sqrt(FAC[ru][0] * FAC[ru][1])

    CASES = [
        dict(id="grad-central", ru="grad", sk="flow", corner="mid", mode="mean", rigid=False,
             lab="the gradient rule, central: the flow speed, the geometric means of 1/Gamma and F, the mean of the two directions"),
        dict(id="grad-rigid", ru="grad", sk="flow", corner="mid", mode="mean", rigid=True,
             lab="the gradient rule, central with the rigid displacement (the innermost point only)"),
        dict(id="grad-least-flow", ru="grad", sk="flow", corner="least", mode="kind", rigid=False,
             lab="the gradient rule, least at the flow speed: 190 yr, F = 1.0, the larger boost of the two directions"),
        dict(id="grad-least-los", ru="grad", sk="los", corner="least", mode="kind", rigid=False,
             lab="the gradient rule, least at |u_los|: 190 yr, F = 1.0, the larger boost of the two directions"),
        dict(id="flux-central", ru="flux", sk="flow", corner="mid", mode="mean", rigid=False,
             lab="the flux rule, central: the flow speed, the geometric means of 1/Gamma and F, the mean of the two directions"),
        dict(id="flux-most", ru="flux", sk="v_84", corner="most", mode="harsh", rigid=False,
             lab="the flux rule, most exposed: the 84th percentile of the speed, 650 yr, F = 0.92, the smaller boost of the two directions")]
    GCASES = [c for c in CASES if c["ru"] == "grad"]

    def speed_of(n, key):
        sp = info[n]["sp"]
        return {"flow": sp["v_flow"], "los": abs(sp["u_los"]), "v_84": sp["v_84"], "v_16": sp["v_16"]}[key]

    def case_pred(n, c, all_radii=False):
        """Predicted residual, log10 of the boost left, at the innermost valid point or at every radius from it."""
        kd = kscale_of(speed_of(n, c["sk"]), c["ru"], c["corner"])
        bi = np.atleast_1d(boost_at(n, c["ru"], kd, 0, rigid=c["rigid"]))
        bn = np.atleast_1d(boost_at(n, c["ru"], kd, 1, rigid=c["rigid"]))
        li, ln = np.log10(np.clip(bi, 0.01, None)), np.log10(np.clip(bn, 0.01, None))
        pr = {"mean": 0.5 * (li + ln), "kind": np.maximum(li, ln), "harsh": np.minimum(li, ln)}[c["mode"]]
        return pr if all_radii else float(pr[0])

    out("   predicted residual at the innermost point, log10 of the boost left, clipped at -2 (dex):")
    for c in CASES:
        out(f"      {c['id']:15s} {c['lab']}")
    out("   galaxy        observed | " + " ".join(f"{c['id']:>15s}" for c in CASES))
    for n in wide:
        rw = info[n]
        out(f"   {n:12s} {rw['r'][rw['k'][0]]:+.3f}   | " + " ".join(f"{case_pred(n, c):+15.3f}" for c in CASES))

    def null_sigma(r0, sobs):
        lo, hi = 0.0, 2.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if np.mean(r0 ** 2 / (sobs ** 2 + mid ** 2)) > 1.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def sim_rho(x, pred, resid, s, n=4000):
        """Spearman's rho that a rule gives: its predicted residuals plus noise, the noise either the observed
        residuals reshuffled or Gaussian with each point's sigma."""
        a = np.array([spearman(x, pred + resid[rng.permutation(len(resid))]) for _ in range(n)])
        b = np.array([spearman(x, pred + rng.normal(0.0, s)) for _ in range(n)])
        return a, b

    def beyond(sims, rho):
        med = float(np.median(sims))
        return float(np.mean(sims <= rho)) if rho < med else float(np.mean(sims >= rho))

    def wls_slope(x, y, w, nb=4000):
        """Weighted least squares of y on x with an intercept: the slope, its formal error, and the bootstrap
        16th and 84th percentiles over galaxies."""
        def fit(xx, yy, ww):
            W = ww.sum()
            xm, ym = (ww * xx).sum() / W, (ww * yy).sum() / W
            sxx = (ww * (xx - xm) ** 2).sum()
            if sxx <= 1e-300:
                return float("nan"), float("nan"), float("nan")
            b = (ww * (xx - xm) * (yy - ym)).sum() / sxx
            return b, math.sqrt(1.0 / sxx), ym - b * xm
        b, sb, a = fit(x, y, w)
        bs = []
        for _ in range(nb):
            i = rng.integers(0, len(x), len(x))
            bb, _, _ = fit(x[i], y[i], w[i])
            if np.isfinite(bb):
                bs.append(bb)
        return b, sb, a, float(np.percentile(bs, 16)), float(np.percentile(bs, 84))

    # --- the tests
    out("")
    out("5. THE TESTS")
    for lab, names in (("boost above 10", strong), ("boost above 5", wide)):
        r0 = np.array([info[n]["r"][info[n]["k"][0]] for n in names])
        r01 = np.array([np.average(info[n]["r"][info[n]["k"][:2]], weights=1.0 / info[n]["sig"][info[n]["k"][:2]] ** 2) for n in names])
        sobs = np.array([info[n]["sig"][info[n]["k"][0]] for n in names])
        sint = null_sigma(r0, sobs)
        s = np.sqrt(sobs ** 2 + sint ** 2)
        vflow = np.array([info[n]["sp"]["v_flow"] for n in names])
        trG = np.array([math.sqrt(info[n]["tr"]["grad"]["flow"][0] * info[n]["tr"]["grad"]["flow"][1]) for n in names])
        xs = (("the flow speed", vflow), ("|u_los|", np.array([abs(info[n]["sp"]["u_los"]) for n in names])),
              ("u_los, signed", np.array([info[n]["sp"]["u_los"] for n in names])),
              ("w_los", np.array([info[n]["sp"]["w_los"] for n in names])),
              ("the gradient rule's central trail over the radius", trG),
              ("the innermost radius, a control", np.array([info[n]["R"][info[n]["k"][0]] for n in names])),
              ("the law's boost there, a control", np.array([info[n]["nu_alg"][info[n]["k"][0]] for n in names])))
        out(f"   {lab} ({len(names)} galaxies): residual at the innermost point, mean {r0.mean():+.3f}, median {np.median(r0):+.3f}, "
            f"rms {math.sqrt(float(np.mean(r0 ** 2))):.3f} dex; the scatter beyond the observational error that sets chi^2 per point")
        out(f"      to one for the unmodified law, {sint:.3f} dex")
        out("   a. Spearman's rho of the residual with x, and its two-sided permutation p (no dependence, as the flux rule requires):")
        for xl, x in xs:
            ra, pa, _ = perm_p(x, r0, rng)
            rb, pb, _ = perm_p(x, r01, rng)
            out(f"      {xl:49s}: innermost point {ra:+.3f} (p {pa:.3f}); two innermost points {rb:+.3f} (p {pb:.3f})")
        out("   b. the rho that the gradient rule would give: its predicted residuals plus noise, the noise the observed residuals")
        out("      reshuffled or Gaussian with each point's sigma; median, 2.5 to 97.5 percent, and the share of draws at or beyond")
        out("      the observed rho on the observed side")
        for c in GCASES:
            pr = np.array([case_pred(n, c) for n in names])
            for xl, x in (("the flow speed", vflow), ("the gradient rule's central trail", trG), ("this case's predicted residual", pr)):
                rho = spearman(x, r0)
                a, b = sim_rho(x, pr, r0, s)
                out(f"      {c['id']:15s} x = {xl:33s}: observed {rho:+.3f}; reshuffled {np.median(a):+.3f} ({np.percentile(a, 2.5):+.3f} to "
                    f"{np.percentile(a, 97.5):+.3f}), beyond {beyond(a, rho):.4f}; Gaussian {np.median(b):+.3f} ({np.percentile(b, 2.5):+.3f} to "
                    f"{np.percentile(b, 97.5):+.3f}), beyond {beyond(b, rho):.4f}")
        c = GCASES[0]
        pr2 = []
        for n in names:
            k = info[n]["k"]
            pa = case_pred(n, c, all_radii=True)
            pr2.append(np.average(pa[[0, k[1] - k[0]]], weights=1.0 / info[n]["sig"][k[:2]] ** 2))
        pr2 = np.array(pr2)
        s01 = np.array([math.sqrt(1.0 / np.sum(1.0 / info[n]["sig"][info[n]["k"][:2]] ** 2)) for n in names])
        s01 = np.sqrt(s01 ** 2 + null_sigma(r01, s01) ** 2)
        for xl, x in (("the gradient rule's central trail", trG), ("this case's predicted residual", pr2)):
            rho = spearman(x, r01)
            a, b = sim_rho(x, pr2, r01, s01)
            out(f"      {c['id']:15s} two innermost points, x = {xl:33s}: observed {rho:+.3f}; reshuffled {np.median(a):+.3f} "
                f"({np.percentile(a, 2.5):+.3f} to {np.percentile(a, 97.5):+.3f}), beyond {beyond(a, rho):.4f}; Gaussian {np.median(b):+.3f}, "
                f"beyond {beyond(b, rho):.4f}")
        out("   c. the slope of the residual on each case's predicted residual, weighted least squares with an intercept;")
        out("      the gradient rule requires a slope of 1 and the flux rule a slope of 0")
        for c in GCASES:
            pr = np.array([case_pred(n, c) for n in names])
            b, sb, a, b16, b84 = wls_slope(pr, r0, 1.0 / s ** 2)
            out(f"      {c['id']:15s}: slope {b:+.3f} +/- {sb:.3f} (bootstrap 16 to 84 percent {b16:+.3f} to {b84:+.3f}), intercept {a:+.3f} dex; "
                f"predicted residuals {pr.min():+.2f} to {pr.max():+.2f}")
        out("   d. level test at the innermost points, chi^2 with the sigma above:")
        for c in CASES:
            pr = np.array([case_pred(n, c) for n in names])
            chi = float(np.sum((r0 - pr) ** 2 / s ** 2)); chi0 = float(np.sum(r0 ** 2 / s ** 2))
            con = (r0 - pr) ** 2 / s ** 2 - r0 ** 2 / s ** 2
            top = np.argsort(-np.abs(con))[:3]
            out(f"      {c['id']:15s}: mean predicted {pr.mean():+.3f} dex; chi^2 {chi:.1f} against {chi0:.1f} for no change, Delta chi^2 "
                f"{chi - chi0:+.1f} on {len(names)} points; largest terms " + ", ".join(f"{names[j]} {con[j]:+.1f}" for j in top))

    # --- refits with each rule's boost left at all radii
    out("")
    out("6. EVERY GALAXY OF THE WIDE SAMPLE REFITTED WITH EACH RULE'S BOOST LEFT AT ALL ITS RADII (the fit machinery of")
    out("   refit_core.py, the boost left entered as a fixed change of log g per point; Delta chi^2 against the unmodified law")
    out("   on the same sample)")
    dss = RC.Dataset(gals, names=wide)
    cref = AF.ctx_of("sparc", "T3", delta=DELTA, use_gok=False)
    cref["a0"] = A0
    rref = dss.fit(cref)
    out(f"   the unmodified law on the wide sample: chi^2 {rref['F'].sum():.1f}")
    for c in CASES:
        if c["rigid"]:
            continue
        arrs = []
        for grp in dss.groups:
            if grp.G == 0:
                arrs.append(np.zeros(0))
                continue
            parts = []
            for gdict, sel in zip(grp.gl, grp.sel):
                n = gdict["name"]
                full = np.zeros(len(gdict["R"]))
                k0 = BLT[n]["k0"]
                full[k0:] = case_pred(n, c, all_radii=True)
                parts.append(full[sel])
            arrs.append(np.concatenate(parts))
        cL = dict(cref)
        cL.update(full=True, gmode="exact", gexact=arrs, _cache={})
        rL = dss.fit(cL)
        d = rL["F"] - rref["F"]
        worst, best = np.argsort(-d)[:5], np.argsort(d)[:5]
        ds_ = [dss.names.index(n) for n in strong]
        out(f"   {c['id']}: Delta chi^2 summed {d.sum():+.1f} over {len(wide)} galaxies, {d[ds_].sum():+.1f} over the {len(strong)} with boost "
            f"above 10; worse by more than 4: {int((d > 4).sum())}, better by more than 4: {int((d < -4).sum())}")
        out("      most worsened: " + ", ".join(f"{dss.names[j]} {d[j]:+.1f}" for j in worst)
            + "; most improved: " + ", ".join(f"{dss.names[j]} {d[j]:+.1f}" for j in best))

    # --- variants
    out("")
    out("7. VARIANTS OF THE SPEED")
    for lab, h0, st in (("H0 = 67.4 km/s/Mpc", 67.4, SIG_T), ("relative transverse velocity 100 km/s", H0, 100.0),
                        ("relative transverse velocity 300 km/s", H0, 300.0)):
        vv = []
        for n in wide:
            sp = speeds(VEL[n], info[n]["D"], info[n]["eD"], h0=h0, sig_t=st, rng=rng)
            vv.append((sp["v_flow"], sp["v_16"], sp["v_84"], abs(sp["u_los"])))
        vv = np.array(vv)
        out(f"   {lab}: flow speed {vv[:, 0].min():.0f} to {vv[:, 0].max():.0f} km/s (median {np.median(vv[:, 0]):.0f}); 16th to 84th "
            f"percentiles, medians {np.median(vv[:, 1]):.0f} and {np.median(vv[:, 2]):.0f}; |u_los| median {np.median(vv[:, 3]):.0f}")
    out(f"\n[{time.time() - t0:.0f} s]")
    if OUT:
        with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(_lines) + "\n")


if __name__ == "__main__":
    main()
