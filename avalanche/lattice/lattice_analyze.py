"""Tables of the lattice simulation of the avalanche kinetics (Paper II, Appendix C) from its 398 runs.

Command (from the repository root; seconds; run after lattice_sim.pl, before rebuild_crossings.py):

    py -3 -X utf8 -I avalanche/lattice/lattice_analyze.py > avalanche/lattice/lattice_analyze-results.txt

Inputs: the run files avalanche/lattice/run-<grid>-<i>.tsv written by lattice_sim.pl (whose header defines the
columns), and the mean-field toy's row at ca = 1.2, eta = 0.05 in avalanche/mean-field-scan-signed-modulus.tsv
for the validation. Outputs: the tables on standard output, and avalanche/lattice/crossings.tsv with one row per
family of runs.

For each family of runs that differ only in the efficiency eta, the crossing of the marginal identity
A * d_emit^2 = 2 is located by linear interpolation of ln(A d_emit^2) in ln(eta) between the two bracketing
grid points, and every other measured quantity is interpolated at the same place. The statistical error of
the crossing comes from the batch-means standard error of A (10 batches) at the bracketing points.
The avalanche-size cutoff is estimated two ways: the moment estimator s_c = 2<s^2>/<s> (exact for the
mean-field form P(s) ~ s^-3/2 exp(-s/s_c), whose cutoff is s_c = 2 A^2 in the mean-field theory) and a binned
maximum-likelihood fit of that form (tau = 3/2 fixed, s >= 4) to the logarithmic size histogram.

Paper II results it supports (App. C): the count of 398 runs (first line); the self-organized state from
every cold start, with the branching ratio n_r below one; the compaction efficiencies at the crossing,
0.13 to 2.0 percent, and one-signed annealing, d_emit/d0 from 0.60 to 0.97 (the table of the main scan); the
match with random-neighbour delivery, A 23.3 against 23.0 and d_emit 0.2939 against 0.2947 (the table of the
reference family R2, row nn/on of the main grid against row rn/on of the modes grid, L = 16, seed 42), within
1.3 percent in A and 0.001 in d_emit;
the 16^3 working size checked from 12^3 to 32^3 (the finite-size table); and the divisors 4.68 and 4.82 of
the random loose packing at ca = 1.0 (the CROSSING lines of the ca1 grid).

Codes: nn and rn are the lattice and random-neighbour deliveries, on and off the labels' feedback into the
laying (lattice_sim.pl). R1 and R2 are the two reference families, q = 6 with d0 = 0.283, ca = 1.2 (R1) and
d0 = 0.334, ca = 1.7 (R2). A_B1, A_toy and A_ex are three forms of the flux balance, A_ex being Eq. (15) of
Paper II before its drift term (see lattice_sim.pl). Only numpy is used.
"""
import csv, glob, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LN2 = math.log(2.0)
STR = {"grid", "deliv", "labels", "hist", "g_E", "g_R"}


def load(pattern="run-*-*.tsv"):
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, pattern))):
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if not r.get("A"):
                    continue
                for k, v in list(r.items()):
                    if k in STR or v is None:
                        continue
                    try:
                        r[k] = float(v)
                    except ValueError:
                        pass
                for k in ("g_E", "g_R"):
                    try:
                        r[k] = float(r.get(k, "NA"))
                    except (TypeError, ValueError):
                        r[k] = float("nan")
                r["file"] = os.path.basename(p)
                rows.append(r)
    return rows


FAMKEY = ("grid", "L", "q", "d0", "ca", "deliv", "labels", "dreg", "k", "th", "eps", "seed")


def families(rows, key=FAMKEY):
    """Group runs differing only in eta; the extension grid 'ext' continues the main scan's families."""
    fam = {}
    for r in rows:
        kk = tuple(("main" if (k == "grid" and r.get(k) == "ext") else r.get(k)) for k in key)
        fam.setdefault(kk, []).append(r)
    for k in fam:
        fam[k].sort(key=lambda r: r["eta"])
    return fam


INTERP = ("A", "d_emit", "d_bar", "d_live", "d_snap", "wall", "f_rw", "f_dead", "f_loose", "rep_loose", "n_r",
          "n_p", "sc_mom_A2", "sc_mom", "wdom", "G1", "G4", "mP", "fmaxdom", "etaAd2", "dw_mean", "s_max",
          "nd_mean", "nd_c", "nd_max", "self_mean", "xi_int")

TOY_SCAN = os.path.normpath(os.path.join(HERE, "../mean-field-scan-signed-modulus.tsv"))


def toy_row(ca=1.2, eta=0.05):
    """The mean-field toy's row at (ca, eta) in its scan at d0 = 0.5, the corner that the validation runs and
    the corner grid of lattice_sim.pl repeat."""
    with open(TOY_SCAN, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if abs(float(r["ca"]) - ca) < 1e-9 and abs(float(r["eta"]) - eta) < 1e-9:
                return r
    return None

REF = {"R1": (6.0, 0.283, 1.2), "R2": (6.0, 0.334, 1.7)}


def crossing(sel, target=2.0):
    """Crossing of A d_emit^2 = target in ln(eta); returns a dict or a reason string."""
    ok = [r for r in sel if r["overflow"] == 0 and r["Ad2"] > 0]
    lt = math.log(target)
    for a, b in zip(ok, ok[1:]):
        ya, yb = math.log(a["Ad2"]), math.log(b["Ad2"])
        if (ya - lt) * (yb - lt) <= 0 and ya != yb:
            t = (lt - ya) / (yb - ya)
            xa, xb = math.log(a["eta"]), math.log(b["eta"])
            c = {"eta": math.exp(xa + t * (xb - xa))}
            for q in INTERP:
                if q in a and q in b and isinstance(a[q], float) and isinstance(b[q], float):
                    c[q] = a[q] + t * (b[q] - a[q])
            c["D_emit"] = 1.0 / math.log(1.0 + c["d_emit"])
            slope = (yb - ya) / (xb - xa)
            sa, sb = a["seA_b"] / a["A"], b["seA_b"] / b["A"]
            sy = math.hypot((1 - t) * sa, t * sb)
            c["slope"] = slope
            c["eta_err"] = c["eta"] * sy / abs(slope)
            c["A_err"] = c["A"] * sy
            c["bracket"] = (a["eta"], b["eta"])
            return c
    if not ok:
        return "no valid points"
    vals = [r["Ad2"] for r in ok]
    if min(vals) > target:
        return f"no crossing: A d_emit^2 > {target} at every eta (min {min(vals):.2f} at eta = {ok[-1]['eta']:g})"
    return f"no crossing: A d_emit^2 < {target} at every eta (max {max(vals):.3f} at eta = {ok[0]['eta']:g}, A = {ok[0]['A']:.2f})"


# ---- avalanche-size histogram: bins of 4 per octave, b(s) = int(4 log2 s + 1e-9), as in lattice_sim.pl ----
def binof(s):
    return int(4 * math.log(s) / math.log(2) + 1e-9)


_BR = {}


def bin_range(b):
    if b in _BR:
        return _BR[b]
    lo = max(1, int(math.floor(2 ** (b / 4.0))) - 2)
    while binof(lo) < b:
        lo += 1
    if binof(lo) != b:
        _BR[b] = None
        return None
    hi = int(math.floor(2 ** ((b + 1) / 4.0))) + 2
    while binof(hi) > b:
        hi -= 1
    _BR[b] = (lo, hi)
    return _BR[b]


def parse_hist(h):
    out = {}
    if not h or h == "NA":
        return out
    for item in h.split(","):
        b, n = item.split(":")
        out[int(b)] = int(n)
    return out


def hist_stats(h):
    tot = sum(h.values())
    if not tot:
        return {}
    st = {"P1": h.get(0, 0) / tot}
    cum = 0
    for b in sorted(h):
        cum += h[b]
        rng = bin_range(b)
        for qn, qq in (("q50", 0.5), ("q90", 0.9), ("q99", 0.99)):
            if qn not in st and cum >= qq * tot:
                st[qn] = rng[1]
    big = {b: n for b, n in h.items() if b >= binof(4)}
    st["mode_s"] = bin_range(max(big, key=lambda b: big[b] / (bin_range(b)[1] - bin_range(b)[0] + 1)))[0] if big else 1
    return st


def tau_eff(h, smax):
    """Least-squares slope of ln(density per unit s) against ln(s) over the bins with 4 <= s <= smax (sign flipped)."""
    xs, ys = [], []
    for b, n in h.items():
        rng = bin_range(b)
        if rng is None or rng[0] < 4 or rng[1] > smax or n < 5:
            continue
        lo, hi = rng
        xs.append(math.log(math.sqrt(lo * hi)))
        ys.append(math.log(n / (hi - lo + 1)))
    if len(xs) < 3:
        return float("nan")
    return -np.polyfit(np.array(xs), np.array(ys), 1)[0]


_NODES = {}


def _nodes(bmin, bmax):
    key = (bmin, bmax)
    if key in _NODES:
        return _NODES[key]
    S, W, I = [], [], []
    for b in range(bmin, bmax + 1):
        rng = bin_range(b)
        if rng is None:
            continue
        lo, hi = rng
        if hi - lo + 1 <= 200:
            s = np.arange(lo, hi + 1, dtype=float)
            w = np.ones_like(s)
        else:   # Simpson on a geometric grid for the sum over a wide bin
            n = 129
            u = np.linspace(math.log(lo - 0.5), math.log(hi + 0.5), n)
            s = np.exp(u)
            du = u[1] - u[0]
            w = np.ones(n)
            w[1:-1:2] = 4
            w[2:-1:2] = 2
            w = w * du / 3 * s
        S.append(s); W.append(w); I.append(np.full(s.shape, b))
    out = (np.concatenate(S), np.concatenate(W), np.concatenate(I))
    _NODES[key] = out
    return out


def fit_cutoff(h, tau=1.5, smin=4):
    """Binned ML fit of P(s) ~ s^-tau exp(-s/s_c) for s >= smin; returns s_c (or nan)."""
    bmin = binof(smin)
    data = {b: n for b, n in h.items() if b >= bmin}
    if sum(data.values()) < 50:
        return float("nan")
    bmax = binof(1e8)
    S, W, I = _nodes(bmin, bmax)
    bins = np.array(sorted(data))
    counts = np.array([data[b] for b in bins], dtype=float)

    def nll(lsc):
        sc = math.exp(lsc)
        f = W * S ** (-tau) * np.exp(-S / sc)
        m = np.bincount(I - bmin, weights=f, minlength=bmax - bmin + 1)
        tot = m.sum()
        p = m[bins - bmin] / tot
        return -np.sum(counts * np.log(np.maximum(p, 1e-300)))

    a, b = math.log(2.0), math.log(1e8)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a); fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a); fd = nll(d)
    return math.exp((a + b) / 2)


def fmt(x, nd=3):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def famlabel(k):
    d = dict(zip(FAMKEY, k))
    return (f"L={int(d['L'])} q={int(d['q'])} d0={d['d0']:.3f} ca={d['ca']:.1f} {d['deliv']}/{d['labels']} k={int(d['k'])} "
            f"dreg={d['dreg']:g} th={d['th']:g} eps={d['eps']:g} seed={int(d['seed'])}")


def per_point_table(sel, out):
    out.write("   eta     A      seA_b  d_emit  d_bar   d_live  A.d2    eta.A.d2 A/A_B1-1 A/A_toy-1 A/A_ex-1  n_p   n_r    1-1/A  "
              "sc/A^2 s_max  wall   f_rw   f_dead f_loose rep_loose wdom   mP    G1    t_s  ovf  A_h1/A_h2\n")
    for r in sel:
        out.write(f"   {r['eta']:<7g} {r['A']:7.2f} {r['seA_b']:6.2f}  {r['d_emit']:.4f}  {r['d_bar']:.4f}  {r['d_live']:.4f}  "
                  f"{r['Ad2']:6.3f}  {r['etaAd2']:.4f}  {r['A'] / r['A_B1'] - 1 if r['A_B1'] else float('nan'):+7.3f}  "
                  f"{r['A'] / r['A_toy'] - 1 if r['A_toy'] else float('nan'):+7.3f}  {r['A'] / r['A_ex'] - 1 if r['A_ex'] else float('nan'):+8.4f}  "
                  f"{r['n_p']:5.2f} {r['n_r']:.4f} {1 - 1 / r['A']:.4f}  {r['sc_mom_A2']:6.3f} {int(r['s_max']):6d} {r['wall']:.4f} "
                  f"{r['f_rw']:.3f}  {r['f_dead']:.3f}  {r['f_loose']:.4f}  {r['rep_loose']:.3f}    {r['wdom']:7.0f} {r['mP']:.3f} "
                  f"{r['G1']:.3f} {r['t_s']:5.0f} {int(r['overflow'])}  {r['A_h1'] / r['A_h2']:.3f}\n")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = sys.stdout
    rows = load()
    if not rows:
        print("no run-*-*.tsv files found"); return
    grids = sorted({r["grid"] for r in rows})
    cpu = sum(r["t_s"] for r in rows)
    out.write(f"lattice simulation of the avalanche kinetics: {len(rows)} runs in grids {', '.join(grids)}; total run time {cpu / 3600:.2f} CPU-hours "
              f"(sum of per-run wall-clock times of single-threaded Perl processes)\n")
    for g in grids:
        gr = [r for r in rows if r["grid"] == g]
        out.write(f"   grid {g}: {len(gr)} runs, {sum(r['t_s'] for r in gr) / 3600:.2f} CPU-hours, longest run {max(r['t_s'] for r in gr):.0f} s, "
                  f"seeds {sorted({int(r['seed']) for r in gr})}\n")
    fam = families(rows)
    cross_rows = []

    # ---------------- validation against the toy ----------------
    val = [r for r in rows if r["grid"] == "valid"]
    if val:
        out.write("\n=== Validation: lattice_sim.pl in the mean-field toy's mode (random delivery, labels off, k = 12, the toy's burn-in cap and budgets) ===\n")
        out.write("| sites | d0 | eta | seed | A | batch s.e. | naive s.e. | d_emit | d_rec | f_rec | A d_emit^2 | s_c(mom)/A^2 | run time (s) |\n")
        out.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in sorted(val, key=lambda r: (r["d0"], r["seed"])):
            out.write(f"| {int(r['N'])} | {r['d0']:.2f} | {r['eta']:g} | {int(r['seed'])} | {r['A']:.2f} | {r['seA_b']:.2f} | {r['seA']:.2f} | "
                      f"{r['d_emit']:.4f} | {r['d_rec']:.4f} | {r['f_rec']:.3f} | {r['Ad2']:.3f} | {r['sc_mom_A2']:.3f} | {r['t_s']:.0f} |\n")
        t = toy_row()
        if t:
            out.write(f"   mean_field_toy.pl at d0 = {float(t['d0']):g}, eta = {float(t['eta']):g} (mean-field-scan-signed-modulus.tsv), "
                      f"N = {int(float(t['N']))}, seed 42: A {t['A']} +/- {t['seA']}, d_emit {t['d_emit']}, d_rec {t['d_rec']}, "
                      f"f_rec {t['f_rec']}, A.d2 {t['A.d2']}\n")

    # ---------------- every family: per-point listing and crossing ----------------
    out.write("\n=== Per-family listings (A.d2 = A d_emit^2; A_B1 = 1 + (d0 - d_bar)/(eta d_bar), a cruder form of the flux balance; "
              "A_toy = 1 + (d0 - d_rec)/(eta d_emit), the mean-field toy's form; A_ex = 1 + (<d_w> - d_bar)/(eta d_emit), Eq. (15) "
              "of Paper II before the drift term) ===\n")
    for k in sorted(fam, key=lambda k: tuple(str(x) for x in k)):
        sel = fam[k]
        out.write(f"\n{famlabel(k)}  [grid {k[0]}]\n")
        per_point_table(sel, out)
        c = crossing(sel)
        if isinstance(c, dict):
            out.write(f"   CROSSING A.d2 = 2 at eta = {c['eta']:.5f} +/- {c['eta_err']:.5f} (bracket {c['bracket'][0]:g}-{c['bracket'][1]:g}, "
                      f"local slope d ln(A.d2)/d ln eta = {c['slope']:.3f}): A = {c['A']:.1f} +/- {c['A_err']:.1f}, d_emit = {c['d_emit']:.4f}, "
                      f"D_emit = {c['D_emit']:.2f}, d_bar = {c['d_bar']:.4f}, wall = {c['wall']:.4f}, f_rw = {c['f_rw']:.3f}, "
                      f"n_r = {c['n_r']:.4f} (1 - 1/A = {1 - 1 / c['A']:.4f}), sc/A^2 = {c['sc_mom_A2']:.3f}\n")
        else:
            out.write(f"   {c}\n")
        d = dict(zip(FAMKEY, k))
        cross_rows.append((d, c, sel))

    # ---------------- crossing table, main grid ----------------
    out.write("\n=== Table: crossings of A d_emit^2 = 2 in the main scan (L = 16, lattice delivery, labels on, dreg = 0) ===\n")
    out.write("| q | d0 | ca | eta at crossing (%) | A | d_emit | d_emit/d0 | D_emit | wall fraction | aligned writes | d_bar | n_r | 1-1/A | s_c/A^2 |\n")
    out.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    with open(os.path.join(HERE, "crossings.tsv"), "w", newline="") as fh:
        cols = ["grid", "L", "q", "d0", "ca", "deliv", "labels", "dreg", "k", "th", "eps", "seed", "eta", "eta_err", "A", "A_err",
                "d_emit", "D_emit", "d_bar", "d_live", "wall", "f_rw", "f_dead", "f_loose", "rep_loose", "n_p", "n_r", "sc_mom_A2",
                "wdom", "mP", "G1", "etaAd2", "slope", "note"]
        fh.write("\t".join(cols) + "\n")
        for d, c, sel in cross_rows:
            if isinstance(c, dict):
                vals = [d[x] for x in FAMKEY] + [c.get(x, "") for x in cols[12:-1]] + ["ok"]
            else:
                vals = [d[x] for x in FAMKEY] + [""] * (len(cols) - 13) + [c]
            fh.write("\t".join(fmt(v, 6) if isinstance(v, float) else str(v) for v in vals) + "\n")
    # the main scan with its extension (L = 16, nn, labels on, seed 42), one row per (q, d0, ca)
    for d, c, sel in sorted(cross_rows, key=lambda t: (t[0]["q"], t[0]["d0"], t[0]["ca"])):
        if d["grid"] != "main":
            continue
        q, d0, ca = d["q"], d["d0"], d["ca"]
        if isinstance(c, dict):
            out.write(f"| {int(q)} | {d0:.3f} | {ca:.1f} | {100 * c['eta']:.3f} ± {100 * c['eta_err']:.3f} | {c['A']:.1f} | "
                      f"{c['d_emit']:.3f} | {c['d_emit'] / d0:.2f} | {c['D_emit']:.2f} | {c['wall']:.3f} | {c['f_rw']:.3f} | {c['d_bar']:.3f} | "
                      f"{c['n_r']:.3f} | {1 - 1 / c['A']:.3f} | {c['sc_mom_A2']:.3f} |\n")
        else:
            mx = max(sel, key=lambda r: r["Ad2"])
            out.write(f"| {int(q)} | {d0:.3f} | {ca:.1f} | none: {c} | {mx['A']:.1f} (at eta {mx['eta']:g}) | {mx['d_emit']:.3f} | "
                      f"{mx['d_emit'] / d0:.2f} | {1 / math.log(1 + mx['d_emit']):.2f} | {mx['wall']:.3f} | {mx['f_rw']:.3f} | {mx['d_bar']:.3f} | "
                      f"{mx['n_r']:.3f} | {1 - 1 / mx['A']:.3f} | {mx['sc_mom_A2']:.3f} |\n")

    # ---------------- reference families across delivery, labels, size, seed, sensitivity ----------------
    def ref_table(match, title):
        sel = [(d, c, s) for d, c, s in cross_rows if match(d) and d["eps"] == 0]
        if not sel:
            return
        out.write(f"\n=== {title} ===\n")
        out.write("| grid | L | delivery | labels | k | th | dreg | seed | eta at crossing (%) | A | d_emit | D_emit | wall fraction | "
                  "aligned writes | s_c/A^2 | distinct sites | primary-site repacks | A at eta 0.0075 | A at eta 0.01 |\n")
        out.write("|" + "---|" * 19 + "\n")
        def at(s, e):
            m = [r for r in s if abs(r["eta"] - e) < 1e-9]
            return f"{m[0]['A']:.2f}" if m else "-"
        for d, c, s in sorted(sel, key=lambda t: (t[0]["deliv"], t[0]["labels"], t[0]["k"], t[0]["L"], t[0]["th"], t[0]["dreg"], t[0]["seed"])):
            head = (f"| {d['grid']} | {int(d['L'])} | {d['deliv']} | {d['labels']} | {int(d['k'])} | {d['th']:g} | {d['dreg']:g} | "
                    f"{int(d['seed'])} | ")
            if isinstance(c, dict):
                out.write(head + f"{100 * c['eta']:.3f} ± {100 * c['eta_err']:.3f} | {c['A']:.1f} ± {c['A_err']:.1f} | {c['d_emit']:.4f} | "
                          f"{c['D_emit']:.2f} | {c['wall']:.4f} | {c['f_rw']:.3f} | {c['sc_mom_A2']:.3f} | "
                          f"{fmt(c.get('nd_mean'), 1)} | {fmt(c.get('self_mean'), 1)} | {at(s, 0.0075)} | {at(s, 0.01)} |\n")
            else:
                out.write(head + f"{c} | | | | | | | | | {at(s, 0.0075)} | {at(s, 0.01)} |\n")

    for name, (q, d0, ca) in REF.items():
        ref_table(lambda d, q=q, d0=d0, ca=ca: d["q"] == q and abs(d["d0"] - d0) < 1e-9 and abs(d["ca"] - ca) < 1e-9,
                  f"Reference family {name} (q = {int(q)}, d0 = {d0}, ca = {ca}): every delivery, label, size, seed and sensitivity run")
    ref_table(lambda d: abs(d["d0"] - 0.5) < 1e-9 and abs(d["ca"] - 1.2) < 1e-9,
              "The mean-field toy's corner (d0 = 0.5, ca = 1.2, q = 6) on and off the lattice")
    corner = [r for r in rows if abs(r["d0"] - 0.5) < 1e-9 and abs(r["ca"] - 1.2) < 1e-9 and r["grid"] == "corner"]
    if corner:
        t = toy_row()
        out.write(f"\nThe corner at eta = 0.05 (mean_field_toy.pl: A {t['A']}, d_emit {t['d_emit']}, A.d2 {t['A.d2']}, "
                  f"eta.A.d2 {0.05 * float(t['A.d2']):.3f}):\n")
        out.write("| L | delivery | labels | k | A | d_emit | A d_emit^2 | eta A d_emit^2 | s_c/A^2 | wall fraction | aligned writes |\n")
        out.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in sorted(corner, key=lambda r: (r["L"], r["deliv"], r["labels"], r["k"])):
            if abs(r["eta"] - 0.05) < 1e-9:
                out.write(f"| {int(r['L'])} | {r['deliv']} | {r['labels']} | {int(r['k'])} | {r['A']:.2f} | {r['d_emit']:.4f} | {r['Ad2']:.3f} | "
                          f"{r['etaAd2']:.4f} | {r['sc_mom_A2']:.3f} | {r['wall']:.4f} | {r['f_rw']:.3f} |\n")

    # ---------------- finite size: the reference families at L = 12, 16, 24, 32 ----------------
    out.write("\n=== Finite-size check: reference families, lattice delivery, labels on (seed 42 unless stated; "
              "L = 16 seeds 7 and 11 are replicates) ===\n")
    out.write("| family | L | seed | eta at crossing (%) | A | d_emit | s_c/A^2 | distinct sites | primary-site repacks | wall fraction | "
              "domain size (sites, weight avg) | largest cluster fraction | G(1) | correlation length | A at 0.0075 | A at 0.01 |\n")
    out.write("|" + "---|" * 16 + "\n")
    for name, (q, d0, ca) in REF.items():
        sel = [(d, c, s) for d, c, s in cross_rows if d["q"] == q and abs(d["d0"] - d0) < 1e-9 and abs(d["ca"] - ca) < 1e-9
               and d["deliv"] == "nn" and d["labels"] == "on" and d["th"] == 0.01 and d["dreg"] == 0 and d["eps"] == 0
               and d["grid"] in ("main", "fsize")]
        for d, c, s in sorted(sel, key=lambda t: (t[0]["L"], t[0]["seed"])):
            def at(e, key="A"):
                m = [r for r in s if abs(r["eta"] - e) < 1e-9]
                return f"{m[0][key]:.2f}" if m else "-"
            near = min(s, key=lambda r: abs(r["eta"] - 0.0075))
            if isinstance(c, dict):
                core = (f"{100 * c['eta']:.3f} ± {100 * c['eta_err']:.3f} | {c['A']:.1f} | {c['d_emit']:.4f} | {c['sc_mom_A2']:.3f} | "
                        f"{fmt(c.get('nd_mean'), 1)} | {fmt(c.get('self_mean'), 1)} | {c['wall']:.4f} | {c['wdom']:.0f} | {c['fmaxdom']:.3f} | {c['G1']:.3f} | "
                        f"{fmt(c.get('xi_int'), 2)}")
            else:
                core = (f"{c} | | | | {fmt(near.get('nd_mean'), 1)} | {fmt(near.get('self_mean'), 1)} | {near['wall']:.4f} | "
                        f"{near['wdom']:.0f} | {near['fmaxdom']:.3f} | {near['G1']:.3f} | {near['xi_int']:.2f}")
            out.write(f"| {name} | {int(d['L'])} | {int(d['seed'])} | {core} | {at(0.0075)} | {at(0.01)} |\n")

    # ---------------- mean-field laws, family summaries ----------------
    out.write("\n=== Family summaries: the invariant eta A d_emit^2, d_emit's spread over eta, the flux balance, the cutoff and law 3 of Paper II, App. C "
              "(points with A >= 3) ===\n")
    out.write("family | n | eta.A.d2 mean (min-max) | d_emit mean (min-max) | A/A_B1-1 (min..max) | A/A_toy-1 (min..max) | "
              "A/A_ex-1 max | n_r max | overflows | sc_mom/A^2 (min-max) | slope d ln s_c/d ln A | f_loose | rep_loose | wall (min-max)\n")
    for k in sorted(fam, key=lambda k: tuple(str(x) for x in k)):
        sel = [r for r in fam[k] if r["A"] >= 3]
        if len(sel) < 1:
            continue
        ia = np.array([r["etaAd2"] for r in sel]); de = np.array([r["d_emit"] for r in sel])
        b1 = np.array([r["A"] / r["A_B1"] - 1 for r in sel]); tt = np.array([r["A"] / r["A_toy"] - 1 for r in sel])
        ex = np.array([abs(r["A"] / r["A_ex"] - 1) for r in sel])
        sc = np.array([r["sc_mom_A2"] for r in sel]); AA = np.array([r["A"] for r in sel])
        slope = np.polyfit(np.log(AA), np.log(sc * AA ** 2), 1)[0] if len(sel) >= 3 else float("nan")
        out.write(f"{famlabel(k)} | {len(sel)} | {ia.mean():.4f} ({ia.min():.4f}-{ia.max():.4f}) | {de.mean():.4f} ({de.min():.4f}-{de.max():.4f}) | "
                  f"{b1.mean():+.3f} ({b1.min():+.3f}..{b1.max():+.3f}) | {tt.mean():+.3f} ({tt.min():+.3f}..{tt.max():+.3f}) | {ex.max():.4f} | "
                  f"{max(r['n_r'] for r in sel):.4f} | {int(sum(r['overflow'] for r in fam[k]))} | {sc.min():.3f}-{sc.max():.3f} | {slope:.2f} | "
                  f"{np.mean([r['f_loose'] for r in sel]):.4f} | {np.mean([r['rep_loose'] for r in sel]):.3f} | "
                  f"{min(r['wall'] for r in sel):.4f}-{max(r['wall'] for r in sel):.4f}\n")

    # ---------------- avalanche-size distributions ----------------
    out.write("\n=== Avalanche-size distributions: moment cutoff s_c = 2<s^2>/<s>, ML cutoff (tau = 3/2, s >= 4), quantiles ===\n")
    out.write("family | eta | A | P(s=1) | median | 90% | 99% | s_max | tau_eff (4 <= s <= A) | s_c(mom)/A^2 | s_c(ML)/A^2 | "
              "distinct sites mean | 2<n_d^2>/<n_d> | max | primary-site repacks\n")
    for k in sorted(fam, key=lambda k: tuple(str(x) for x in k)):
        for r in fam[k]:
            if r["A"] < 5:
                continue
            h = parse_hist(r.get("hist", ""))
            if not h:
                continue
            st = hist_stats(h)
            scml = fit_cutoff(h)
            r["sc_ml_A2"] = scml / r["A"] ** 2
            r["tau_eff"] = tau_eff(h, max(r["A"], 8))
            out.write(f"{famlabel(k)} | {r['eta']:g} | {r['A']:.1f} | {st['P1']:.3f} | {st['q50']} | {st['q90']} | {st['q99']} | "
                      f"{int(r['s_max'])} | {r['tau_eff']:.2f} | {r['sc_mom_A2']:.3f} | {r['sc_ml_A2']:.3f} | {fmt(r.get('nd_mean'), 1)} | "
                      f"{fmt(r.get('nd_c'), 1)} | {fmt(r.get('nd_max'), 0)} | {fmt(r.get('self_mean'), 2)}\n")

    # ---------------- gain runs ----------------
    gr = [r for r in rows if r["eps"] != 0]
    if gr:
        out.write("\n=== Drive-asymmetry runs: the gain g = ln(response ratio)/ln(drive ratio) between the halves x < L/2 and x >= L/2 ===\n")
        out.write("Per half h, free-volume bookkeeping is exact: R_h eta <d_emit>_h = P_h (<d_w>_h - <d_old>_h) - dS_h, so\n"
                  "g_R - 1 = [ln(inj0/inj1) - ln(demit0/demit1) + (drift term)] / ln(P0/P1), with inj = <d_w> - <d_old> per primary.\n")
        out.write("The drift part is the change of each half's total free volume over the finite window; it averages to zero over long\n"
                  "windows and seeds, so g_inf = 1 + injection part + emission part estimates the long-window gain. The injection part\n"
                  "splits into a write part (more aligned writes in the driven half) and an overwrite part (looser sites overwritten).\n")
        out.write("run | eta | A | P0/P1 | g_E (events incl. primaries) | g_R (repacks) | injection part (write, overwrite) | emission part | "
                  "drift part | g_inf | 1 - 1/A | dbar0 dbar1 | demit0 demit1\n")
        summ = {}
        for r in sorted(gr, key=lambda r: (r["d0"], r["deliv"], r["L"], r["eps"], r["seed"])):
            lp = math.log(r["P0"] / r["P1"])
            inj0, inj1 = r["dw0"] - r["dold0"], r["dw1"] - r["dold1"]
            injm = 0.5 * (inj0 + inj1)
            pi = math.log(inj0 / inj1) / lp
            pw = ((r["dw0"] - r["dw1"]) / injm) / lp
            po = (-(r["dold0"] - r["dold1"]) / injm) / lp
            pe = -math.log(r["demit0"] / r["demit1"]) / lp
            pdr = float("nan")
            if isinstance(r.get("dS0"), float):
                pdr = (math.log((1 - r["dS0"] / (r["P0"] * inj0)) / (1 - r["dS1"] / (r["P1"] * inj1)))) / lp
            ginf = 1 + pi + pe
            r["g_inf"] = ginf
            summ.setdefault((r["d0"], r["ca"], r["deliv"], r["L"], r["eps"]), []).append(r)
            out.write(f"{famlabel(tuple(r.get(x) for x in FAMKEY))} | {r['eta']:g} | {r['A']:.2f} | {r['P0'] / r['P1']:.4f} | {r['g_E']:.4f} | "
                      f"{r['g_R']:.4f} | {pi:+.4f} ({pw:+.4f}, {po:+.4f}) | {pe:+.4f} | {pdr:+.4f} | {ginf:.4f} | {1 - 1 / r['A']:.4f} | "
                      f"{r['dbar0']:.4f} {r['dbar1']:.4f} | {r['demit0']:.4f} {r['demit1']:.4f}\n")
        out.write("\nGain summary (mean over seeds; the spread is the standard deviation over seeds, or n/a for one seed):\n")
        out.write("| d0 | ca | delivery | L | eps | seeds | A | 1 - 1/A | g_E | g_R | g_inf |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
        for kk in sorted(summ):
            rs = summ[kk]
            def ms(key):
                v = np.array([x[key] for x in rs])
                return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f} (n/a)"
            Am = np.mean([x["A"] for x in rs])
            out.write(f"| {kk[0]:.3f} | {kk[1]:.1f} | {kk[2]} | {int(kk[3])} | {kk[4]:g} | {len(rs)} | {Am:.2f} | {1 - 1 / Am:.3f} | "
                      f"{ms('g_E')} | {ms('g_R')} | {ms('g_inf')} |\n")
        out.write("\nPooled over every run of a family and delivery (all L and eps): mean, standard deviation, standard error, and the\n"
                  "distance of the mean from the window's floor 1 - 1/A and from unity in standard errors:\n")
        pool = {}
        for r in gr:
            pool.setdefault((r["d0"], r["ca"], r["deliv"]), []).append(r)
        for kk in sorted(pool):
            for key in ("g_R", "g_E"):
                v = np.array([x[key] for x in pool[kk]])
                Am = np.mean([x["A"] for x in pool[kk]])
                se = v.std(ddof=1) / math.sqrt(len(v))
                out.write(f"   d0 {kk[0]:.3f} ca {kk[1]:.1f} {kk[2]} {key}: n = {len(v)}, mean {v.mean():.4f}, sd {v.std(ddof=1):.4f}, "
                          f"se {se:.4f}; floor 1-1/A = {1 - 1 / Am:.4f} ({(v.mean() - (1 - 1 / Am)) / se:+.1f} se), "
                          f"unity ({(v.mean() - 1) / se:+.1f} se)\n")


if __name__ == "__main__":
    main()
