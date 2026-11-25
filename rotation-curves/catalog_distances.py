"""The catalog-distance check of the two-constant fit: the saved ladders of catalog_distances_nodes.py analyzed (Paper I
section 4.2, Paper II section 6.1).

The law is the boost law of Paper I, Eq. (8) (code T3 in refit_core.py). For each case of catalog_distances_common.py
(input sparc or model, model full or rel, q fit or 1) the script prints the chi^2 curve over the free volume delta
and its minimum, by the rule of two_constant_fit.pl: the edge node when the
lowest node is at an end of the scan, else the parabola through the three lowest nodes. At the minimum it gives
kappa = (c H0/a0) ln(1 + delta) and kappa_dS = sqrt(Omega_L) kappa = D ln(1 + delta), the counting factor that the
fitted pair implies, which the scale relation of the constrained fit sets to 1. It then finds the delta where each
equals 1, and the cost of moving there with its galaxy-clustered SD from a galaxy bootstrap (2,000 draws, every model
refitted along its stored a0 profiles, as refit_core.py's bootstrap does).

The numbers of the papers, by case of the results file:
  Paper I section 4.2 and Paper II section 6.1, catalog distances and inclinations held fixed: the full solution no
  longer pays a penalty against the algebraic relation, chi^2 6678.68 at the minimum of sparc-full-q1 against
  6704.83 at that of sparc-rel-q1.
  Paper II section 6.1, the full solution on SPARC's field with q profiled (sparc-full-qfit): the minimum at the
  scan's lowest node, delta = 0.003, the bootstrap span of 0.003 to 0.15 (16 to 84 percent) and D ln(1 + delta),
  kappa_dS, of 0.027 and 0.030 at delta = 0.005.

Input: the saved ladders in rotation-curves/catalog-distances-nodes/, written by catalog_distances_nodes.py (not
deposited; its header gives the runs). Output: rotation-curves/catalog_distances-results.txt. A line
'TODO tag@delta' names a ladder still to be run, at a minimum or at a point where kappa or kappa_dS equals 1; the
deposited output has none.

Run from the repository root after the ladders (under a minute; numpy):
    python rotation-curves/catalog_distances.py --out rotation-curves/catalog_distances-results.txt
"""
import math, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import catalog_distances_common as C
RC = C.RC

A = sys.argv
arg = lambda n, d=None: A[A.index(n) + 1] if n in A else d
NBOOT = int(arg("--nboot", "2000"))
SCAN = [0.003, 0.005, 0.01, 0.03, 0.1, 0.2, 0.3, 0.4]
TARGETS = [("kappa", 67.4, False), ("kappa", 73.0, False), ("kappa_dS", 67.4, True), ("kappa_dS", 73.0, True)]
LINES = []


def out(s=""):
    print(s, flush=True); LINES.append(s)


def scan_min_rule(pts):
    """pts: sorted [(delta, chi2, la)]. The minimum rule of refit_core.Scan and two_constant_fit.pl: the edge node if
    the lowest is at an end, else the parabola through the three lowest nodes (parab3), falling back to the node and
    its neighbours."""
    n = pts
    ib = int(np.argmin([x[1] for x in n]))
    if ib == 0 or ib == len(n) - 1:
        return n[ib][0], n[ib][1], True
    n3 = sorted(sorted(n, key=lambda x: x[1])[:3])
    r = RC.parab3([(x[0], x[1]) for x in n3])
    if r is None or r[0] < n3[0][0] or r[0] > n3[2][0]:
        r = RC.parab3([(x[0], x[1]) for x in n[ib - 1:ib + 2]])
        if r is None or r[0] < n[ib - 1][0] or r[0] > n[ib + 1][0]:
            return n[ib][0], n[ib][1], False
    return r[0], r[2], False


def la_interp(pts, d):
    """log10 a0 at delta d, linear between the bracketing ladders (pts sorted by delta: (delta, chi2, la))."""
    ds = [p[0] for p in pts]; ls = [p[2] for p in pts]
    if d <= ds[0]:
        return ls[0] + (d - ds[0]) * (ls[1] - ls[0]) / (ds[1] - ds[0])
    if d >= ds[-1]:
        return ls[-1] + (d - ds[-1]) * (ls[-1] - ls[-2]) / (ds[-1] - ds[-2])
    return float(np.interp(d, ds, ls))


def solve_unity(pts, H0, dS):
    """delta where kappa(delta) = 1, with log10 a0 interpolated linearly between ladders; None if not bracketed."""
    f = lambda d: C.kappa(10 ** la_interp(pts, d), d, H0, dS) - 1.0
    lo, hi = pts[0][0], pts[-1][0]
    if f(lo) > 0 or f(hi) < 0:
        return None
    for _ in range(80):
        m = 0.5 * (lo + hi)
        if f(m) < 0:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def analyze(tag, W, todo):
    nodes = C.load_nodes(tag)
    if not nodes:
        out(f"\n{tag}: no ladders yet"); return None
    scan = sorted((d, nodes[d]["chi2"], nodes[d]["la"]) for d in nodes if any(abs(d - s) < 1e-9 for s in SCAN))
    allp = sorted((d, nodes[d]["chi2"], nodes[d]["la"]) for d in nodes)
    out(f"\n{tag}  ({len(scan)} scan nodes, {len(allp)} ladders in all)")
    if len(scan) < 3:
        return None
    cmin = min(p[1] for p in allp)
    out("   delta     a0(1e-10)    chi2      -min   kappa(67.4) kappa(73)  kappa_dS(67.4) kappa_dS(73)")
    for d, c, la in allp:
        a0 = 10 ** la
        mark = "" if any(abs(d - s) < 1e-9 for s in SCAN) else "  (extra ladder)"
        out(f"   {d:<8.4f}  {a0 * 1e10:.4f}   {c:9.2f}  {c - cmin:+7.2f}   {C.kappa(a0, d, 67.4):.3f}       {C.kappa(a0, d, 73):.3f}"
            f"      {C.kappa(a0, d, 67.4, True):.3f}          {C.kappa(a0, d, 73, True):.3f}{mark}")
    dmin, cpar, edge = scan_min_rule(scan)
    dmin = round(dmin, 4)
    out(f"   minimum over the scan nodes: delta = {dmin:g}" + (" (at the edge of the scan)" if edge else f" (three-lowest-node parabola, value {cpar:.2f})"))
    if dmin not in nodes:
        todo.append((tag, dmin))
        out(f"   TODO ladder at delta {dmin:g}"); recmin = None
        la_m = la_interp(allp, dmin); c_m = cpar
    else:
        recmin = nodes[dmin]; la_m = recmin["la"]; c_m = recmin["chi2"]
    a0m = 10 ** la_m
    out(f"   at the minimum (scan rule): a0 = {a0m * 1e10:.4f}e-10, chi2 = {c_m:.2f}; kappa {C.kappa(a0m, dmin, 67.4):.3f} (67.4), {C.kappa(a0m, dmin, 73):.3f} (73); "
        f"kappa_dS {C.kappa(a0m, dmin, 67.4, True):.3f} (67.4), {C.kappa(a0m, dmin, 73, True):.3f} (73)")
    # refinement: the same rule on every ladder (scan nodes, extra nodes, the unity ladders); the cost below is
    # measured from the lowest ladder of all
    dref, cref_par, edge_ref = scan_min_rule(allp)
    dlow = min(nodes, key=lambda d: nodes[d]["chi2"])
    out(f"   refined over all {len(allp)} ladders: parabola minimum delta = {dref:.4f}" + (" (edge)" if edge_ref else f" (value {cref_par:.2f})")
        + f"; lowest ladder delta {dlow:g}, chi2 {nodes[dlow]['chi2']:.2f}, a0 {nodes[dlow]['a0'] * 1e10:.4f}e-10, "
        f"kappa {C.kappa(nodes[dlow]['a0'], dlow, 67.4):.3f}/{C.kappa(nodes[dlow]['a0'], dlow, 73):.3f}, "
        f"kappa_dS {C.kappa(nodes[dlow]['a0'], dlow, 67.4, True):.3f}/{C.kappa(nodes[dlow]['a0'], dlow, 73, True):.3f} (67.4/73)")
    if recmin is not None and nodes[dlow]["chi2"] < c_m - 1e-6:
        out(f"   the cost reference is the lowest ladder (delta {dlow:g}), {c_m - nodes[dlow]['chi2']:.2f} below the scan rule's ladder")
        recmin = nodes[dlow]; c_m = recmin["chi2"]
    # the galaxy bootstrap of the minimum itself, with kappa per draw: each draw's minimum by the same rule on its
    # reweighted scan nodes
    scan_recs = [nodes[d] for d, _, _ in scan]
    res = dict(tag=tag, dmin=dmin, edge=edge, a0min=a0m, chimin=c_m, unity={})
    dd, kk = [], {t: [] for t in TARGETS}
    for b in range(NBOOT):
        pts_b = [(r["delta"], *C.node_min(r, W[b])) for r in scan_recs]       # (delta, chi, la)
        pts_b = sorted(pts_b)
        db, _, _ = scan_min_rule(pts_b)
        lab = la_interp(pts_b, db)
        dd.append(db)
        for t in TARGETS:
            kk[t].append(C.kappa(10 ** lab, db, t[1], t[2]))
    dd = np.array(dd)
    out(f"   bootstrap of the minimum ({NBOOT} draws): delta 16-84 percent {np.percentile(dd, 16):.4f} to {np.percentile(dd, 84):.4f}, "
        f"median {np.median(dd):.4f}; draws at the lowest node {np.mean(dd <= SCAN[0] + 1e-9) * 100:.1f} percent, at the highest {np.mean(dd >= SCAN[-1] - 1e-9) * 100:.1f} percent")
    for t in TARGETS:
        k = np.array(kk[t])
        out(f"      {t[0]} (H0 {t[1]:g}) at the draws' minima: 16-84 percent {np.percentile(k, 16):.3f} to {np.percentile(k, 84):.3f}; "
            f"draws reaching 1: {np.mean(k >= 1.0) * 100:.1f} percent" if np.median(k) < 1 else
            f"      {t[0]} (H0 {t[1]:g}) at the draws' minima: 16-84 percent {np.percentile(k, 16):.3f} to {np.percentile(k, 84):.3f}; "
            f"draws reaching 1: {np.mean(k <= 1.0) * 100:.1f} percent")
    # unity: where each product equals 1, and the cost of moving there
    for t in TARGETS:
        du = solve_unity(allp, t[1], t[2])
        if du is None:
            out(f"   {t[0]} = 1 at H0 {t[1]:g}: not bracketed by the ladders"); continue
        du4 = round(du, 4)
        near = [d for d in nodes if abs(d - du4) <= 0.0015]
        if not near:
            todo.append((tag, du4))
            out(f"   {t[0]} = 1 at H0 {t[1]:g}: delta ~ {du:.4f} (interpolated); TODO ladder at {du4:g}")
            continue
        dn = min(near, key=lambda d: abs(d - du4))
        rn = nodes[dn]
        kn = C.kappa(rn["a0"], dn, t[1], t[2])
        cost = rn["chi2"] - c_m
        if recmin is not None and abs(rn["delta"] - recmin["delta"]) < 1e-9:
            sd = 0.0
            sdtxt = "the unity point is the minimum's own ladder"
        elif recmin is not None:
            cb =np.array([C.node_min(rn, W[b])[0] - C.node_min(recmin, W[b])[0] for b in range(NBOOT)])
            sd = float(np.std(cb, ddof=1))
            frac = float(np.mean(cb <= 0))
            sdtxt = (f"clustered SD {sd:.1f} ({cost / sd:.2f} SD; {frac * 100:.1f} percent of draws favour the unity point); "
                     f"statistical {math.sqrt(max(cost, 0)):.2f} sigma")
        else:
            sd = float("nan"); sdtxt = "clustered SD pending the minimum's ladder"
        res["unity"][t] = dict(delta=du, dnode=dn, kappa_node=kn, cost=cost, sd=sd)
        out(f"   {t[0]} = 1 at H0 {t[1]:g}: delta = {du:.4f} (ladder at {dn:g}: a0 {rn['a0'] * 1e10:.4f}e-10, {t[0]} {kn:.3f}); "
            f"cost Delta chi^2 = {cost:+.2f}; {sdtxt}")
    return res


if __name__ == "__main__":
    W = RC.boot_weights(153, NBOOT, 20261013)
    todo = []
    out("The boost law (T3), fixed-delta ladders (a0 refitted at each), corrected Gamma, mass-to-light ratios fitted within 0.10 dex priors.")
    out("q fit = distances and inclinations profiled within their catalog priors (as in two_constant_fit.pl); q 1 = catalog values held fixed.")
    out(f"kappa = (c H0/a0) ln(1 + delta); kappa_dS = sqrt(Omega_L) kappa, Omega_L = {C.OMEGA_L}, sqrt = {C.SQ_OL:.4f}. Galaxy bootstrap: {NBOOT} draws, seed 20261013.")
    out("kappa_dS = D ln(1 + delta), with D = c H_dS/a0, is the counting factor that the fitted pair implies; the constrained fit holds it at 1.")
    for q in ("fit", "1"):
        for inp in ("sparc", "model"):
            for m in ("full", "rel"):
                analyze(C.case_tag(inp, m, q), W, todo)
    out("\nTODO " + ";".join(f"{t}@{d:g}" for t, d in todo) if todo else "\nTODO none")
    if arg("--out"):
        open(arg("--out"), "w", encoding="utf-8", newline="\n").write("\n".join(LINES) + "\n")
