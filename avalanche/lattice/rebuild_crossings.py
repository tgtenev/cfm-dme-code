"""An independent check of the lattice simulation's crossings and of its flux balance, from the raw run rows.

Command (from the repository root; seconds; run after lattice_analyze.py, which writes crossings.tsv):

    py -3 -X utf8 -I avalanche/lattice/rebuild_crossings.py > avalanche/lattice/rebuild_crossings-results.txt

Inputs: the run files avalanche/lattice/run-<grid>-<i>.tsv of lattice_sim.pl (whose header defines the columns),
avalanche/lattice/crossings.tsv of lattice_analyze.py, and the mean-field toy's row at ca = 1.2, eta = 0.05 in
avalanche/mean-field-scan-signed-modulus.tsv. Output: the report on standard output.

It groups runs that differ only in the efficiency eta and locates the crossing of the marginal identity
A * d_emit^2 = 2 under three rules, written independently of lattice_analyze.py:

  rule A  lattice_analyze.py's rule: linear interpolation of ln(A d_emit^2) against ln(eta) between the two
          bracketing points; every other quantity interpolated linearly at the same fraction t.
  rule B  an alternative that interpolates only the slowly varying parts. By the exact flux bookkeeping,
          (A - 1) eta = (<d_w> - d_bar)/d_emit - drift*eta varies slowly with eta, and d_emit hardly varies.
          C(eta) = (A - 1) eta is interpolated linearly in ln C against ln eta, d_emit linearly against ln eta,
          and (1 + C/eta) d_emit^2 = 2 is solved by bisection inside the bracket. A* = 2/d_emit*^2 exactly.
  rule C  a local quadratic in ln(eta) for ln(A d_emit^2), least squares through the bracket and its two
          outer neighbours (three or four points), solved inside the bracket; d_emit by rule A's t.

It also checks: the flux balance of Paper II, Eq. (15), row by row, as the identity A = A_ex - drift (the
line "max |(A_ex - drift)/A - 1|", the 8e-6 of Paper II, App. C) and the cruder forms A_toy and A_B1; the
validation rows against the mean-field toy; a necessary condition for marginality on c_a; the scaling of the
avalanche-size cutoff with A; and the spread of d_emit over each family's eta grid.
Codes: nn and rn are the lattice and random-neighbour deliveries, on and off the labels' feedback into the
laying (see lattice_sim.pl). Only the standard library is used.
"""
import csv, glob, math, os, sys

SRC = os.path.dirname(os.path.abspath(__file__))
TOY_SCAN = os.path.normpath(os.path.join(SRC, "../mean-field-scan-signed-modulus.tsv"))
STRCOLS = {"grid", "deliv", "labels", "hist", "g_E", "g_R"}


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(SRC, "run-*-*.tsv"))):
        with open(p, newline="") as fh:
            rd = csv.reader(fh, delimiter="\t")
            hdr = next(rd)
            for rec in rd:
                if len(rec) != len(hdr):
                    print(f"# skipped malformed row in {os.path.basename(p)} ({len(rec)} fields)")
                    continue
                r = {}
                for k, v in zip(hdr, rec):
                    if k in STRCOLS:
                        r[k] = v
                    else:
                        try:
                            r[k] = float(v)
                        except ValueError:
                            r[k] = v
                r["file"] = os.path.basename(p)
                rows.append(r)
    return rows


def famkey(r):
    g = "main" if r["grid"] == "ext" else r["grid"]
    return (g, int(r["L"]), int(r["q"]), round(r["d0"], 4), round(r["ca"], 4), r["deliv"], r["labels"],
            round(r["dreg"], 5), int(r["k"]), round(r["th"], 5), round(r["eps"], 4), int(r["seed"]))


def lab(k):
    g, L, q, d0, ca, dv, lb, dreg, kk, th, eps, seed = k
    return f"{g:6s} L={L:2d} q={q:2d} d0={d0:.3f} ca={ca:.1f} {dv}/{lb:3s} k={kk:2d} dreg={dreg:g} th={th:g} eps={eps:g} seed={seed}"


def bracket(sel, target=2.0):
    lt = math.log(target)
    for i in range(len(sel) - 1):
        a, b = sel[i], sel[i + 1]
        ya, yb = math.log(a["A"] * a["d_emit"] ** 2), math.log(b["A"] * b["d_emit"] ** 2)
        if (ya - lt) * (yb - lt) <= 0 and ya != yb:
            return i
    return None


def rule_a(sel, i, target=2.0):
    a, b = sel[i], sel[i + 1]
    lt = math.log(target)
    ya, yb = math.log(a["A"] * a["d_emit"] ** 2), math.log(b["A"] * b["d_emit"] ** 2)
    xa, xb = math.log(a["eta"]), math.log(b["eta"])
    t = (lt - ya) / (yb - ya)
    eta = math.exp(xa + t * (xb - xa))
    lin = lambda key: a[key] + t * (b[key] - a[key])
    A, de = lin("A"), lin("d_emit")
    slope = (yb - ya) / (xb - xa)
    sa, sb = a["seA_b"] / a["A"], b["seA_b"] / b["A"]
    sy = math.hypot((1 - t) * sa, t * sb)
    return dict(eta=eta, eta_err=eta * sy / abs(slope), A=A, A_err=A * sy, d_emit=de, D=1 / math.log(1 + de), t=t,
                slope=slope, wall=lin("wall"), f_rw=lin("f_rw"), n_r=lin("n_r"), sc=lin("sc_mom_A2"),
                Aself=2 / de ** 2, bracket=(a["eta"], b["eta"]))


def rule_b(sel, i, target=2.0):
    a, b = sel[i], sel[i + 1]
    xa, xb = math.log(a["eta"]), math.log(b["eta"])
    Ca, Cb = (a["A"] - 1) * a["eta"], (b["A"] - 1) * b["eta"]

    def at(x):
        t = (x - xa) / (xb - xa)
        C = math.exp(math.log(Ca) + t * (math.log(Cb) - math.log(Ca)))
        de = a["d_emit"] + t * (b["d_emit"] - a["d_emit"])
        eta = math.exp(x)
        return (1 + C / eta) * de * de - target, de, C, eta

    lo, hi = xa, xb
    flo = at(lo)[0]
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = at(mid)[0]
        if (fm < 0) == (flo < 0):
            lo, flo = mid, fm
        else:
            hi = mid
    _, de, C, eta = at(0.5 * (lo + hi))
    return dict(eta=eta, A=1 + C / eta, d_emit=de, D=1 / math.log(1 + de))


def rule_c(sel, i, target=2.0):
    lo, hi = max(0, i - 1), min(len(sel) - 1, i + 2)
    pts = sel[lo:hi + 1]
    xs = [math.log(r["eta"]) for r in pts]
    ys = [math.log(r["A"] * r["d_emit"] ** 2) for r in pts]
    if len(pts) < 3:
        return None
    # least-squares quadratic y = c0 + c1 x + c2 x^2 (normal equations, centred x)
    xm = sum(xs) / len(xs)
    u = [x - xm for x in xs]
    S = [[sum(ui ** (p + q) for ui in u) for q in range(3)] for p in range(3)]
    T = [sum(yi * ui ** p for yi, ui in zip(ys, u)) for p in range(3)]
    # solve 3x3 by Cramer's rule
    def det(M):
        return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    D0 = det(S)
    c = []
    for j in range(3):
        M = [row[:] for row in S]
        for p in range(3):
            M[p][j] = T[p]
        c.append(det(M) / D0)
    f = lambda x: c[0] + c[1] * (x - xm) + c[2] * (x - xm) ** 2 - math.log(target)
    a, b = math.log(sel[i]["eta"]), math.log(sel[i + 1]["eta"])
    fa, fb = f(a), f(b)
    if fa * fb > 0:      # the fitted curve does not cross inside the bracket: report the nearest root
        # fall back to scanning a wider interval
        a, b = xs[0], xs[-1]
        fa = f(a)
    lo_, hi_ = a, b
    for _ in range(200):
        m = 0.5 * (lo_ + hi_)
        if (f(m) < 0) == (fa < 0):
            lo_, fa = m, f(m)
        else:
            hi_ = m
    x = 0.5 * (lo_ + hi_)
    r0, r1 = sel[i], sel[i + 1]
    t = (x - math.log(r0["eta"])) / (math.log(r1["eta"]) - math.log(r0["eta"]))
    de = r0["d_emit"] + t * (r1["d_emit"] - r0["d_emit"])
    return dict(eta=math.exp(x), A=2 / de ** 2, d_emit=de, D=1 / math.log(1 + de), npts=len(pts))


def load_crossings_tsv():
    out = {}
    with open(os.path.join(SRC, "crossings.tsv"), newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            g = r["grid"]
            k = (g, int(float(r["L"])), int(float(r["q"])), round(float(r["d0"]), 4), round(float(r["ca"]), 4), r["deliv"],
                 r["labels"], round(float(r["dreg"]), 5), int(float(r["k"])), round(float(r["th"]), 5), round(float(r["eps"]), 4),
                 int(float(r["seed"])))
            out[k] = r
    return out


def toy_row(ca=1.2, eta=0.05):
    """The mean-field toy's row at (ca, eta) of its scan at d0 = 0.5, N = 5e4, seed 42."""
    with open(TOY_SCAN, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if abs(float(r["ca"]) - ca) < 1e-9 and abs(float(r["eta"]) - eta) < 1e-9:
                return r
    return None


def main():
    out = sys.stdout
    rows = load()
    out.write(f"rows read: {len(rows)} from {len(set(r['file'] for r in rows))} files\n")
    fam = {}
    for r in rows:
        fam.setdefault(famkey(r), []).append(r)
    for k in fam:
        fam[k].sort(key=lambda r: r["eta"])
    out.write(f"families (runs differing only in eta): {len(fam)}\n")
    ctsv = load_crossings_tsv()
    out.write(f"families in crossings.tsv: {len(ctsv)}; keys missing there: "
              f"{[lab(k) for k in fam if k not in ctsv]}; extra there: {[lab(k) for k in ctsv if k not in fam]}\n")

    # ---------- consistency of the printed columns with each other ----------
    bad = 0
    for r in rows:
        ad2 = r["A"] * r["d_emit"] ** 2
        if abs(ad2 / r["Ad2"] - 1) > 2e-5 or abs(1 / math.log(1 + r["d_emit"]) / r["D_emit"] - 1) > 2e-5:
            bad += 1
            out.write(f"   column mismatch: {lab(famkey(r))} eta={r['eta']}: A d^2 {ad2:.6g} vs {r['Ad2']}\n")
    out.write(f"rows whose Ad2 or D_emit column disagrees with A and d_emit beyond print precision: {bad}\n")
    dup = {}
    for r in rows:
        dup.setdefault((famkey(r), r["eta"]), []).append(r["file"])
    dd = {k: v for k, v in dup.items() if len(v) > 1}
    out.write(f"duplicate (family, eta) points: {len(dd)} {[(lab(k[0]), k[1], v) for k, v in dd.items()]}\n")
    nonmono = []
    for k, sel in fam.items():
        y = [r["A"] * r["d_emit"] ** 2 for r in sel]
        if any(y[i + 1] >= y[i] for i in range(len(y) - 1)):
            nonmono.append(lab(k))
    out.write(f"families where A d^2 is not strictly decreasing in eta: {nonmono}\n")
    out.write(f"overflowed avalanches (cap 2e6) over all rows: {sum(r['overflow'] for r in rows):.0f}\n")

    # ---------- crossings ----------
    out.write("\n=== Crossings of A d_emit^2 = 2: rule A (lattice_analyze.py's), rule B (slowly varying parts), rule C (local quadratic) "
              "against crossings.tsv ===\n")
    out.write("family | n | bracket | ruleA eta% +- err | A | d_emit | D | A(2/d^2) | ruleB eta% | A | d_emit | D | "
              "ruleC eta% | A | D | tsv eta% | tsv A | tsv d_emit | tsv D | flags\n")
    res = {}
    for k in sorted(fam, key=lambda k: tuple(str(x) for x in k)):
        sel = [r for r in fam[k] if r["overflow"] == 0 and r["A"] > 0]
        i = bracket(sel)
        if i is None:
            mx = max(sel, key=lambda r: r["A"] * r["d_emit"] ** 2)
            res[k] = None
            note = ctsv.get(k, {}).get("note", "")
            out.write(f"{lab(k)} | {len(sel)} | none: max A d^2 = {mx['A'] * mx['d_emit'] ** 2:.3f} at eta {mx['eta']:g} "
                      f"(A = {mx['A']:.2f}) | tsv: {note[:90]}\n")
            continue
        a = rule_a(sel, i)
        b = rule_b(sel, i)
        c = rule_c(sel, i)
        res[k] = (a, b, c)
        t = ctsv.get(k)
        flags = []
        if t:
            te, tA, td, tD = float(t["eta"]), float(t["A"]), float(t["d_emit"]), float(t["D_emit"])
            if abs(te - a["eta"]) > 1e-6 * max(1, te) * 10 or abs(tA - a["A"]) > 1e-4 * tA or abs(td - a["d_emit"]) > 1e-5:
                flags.append("TSV-MISMATCH")
        if abs(b["eta"] / a["eta"] - 1) * a["eta"] > 2 * a["eta_err"]:
            flags.append(f"B-A eta diff {100 * (b['eta'] - a['eta']):+.4f}% vs err {100 * a['eta_err']:.4f}%")
        if c and abs(c["eta"] / a["eta"] - 1) * a["eta"] > 2 * a["eta_err"]:
            flags.append(f"C-A eta diff {100 * (c['eta'] - a['eta']):+.4f}% vs err {100 * a['eta_err']:.4f}%")
        out.write(f"{lab(k)} | {len(sel)} | {a['bracket'][0]:g}-{a['bracket'][1]:g} | {100 * a['eta']:.4f} +- {100 * a['eta_err']:.4f} | "
                  f"{a['A']:.2f} | {a['d_emit']:.5f} | {a['D']:.4f} | {a['Aself']:.2f} | {100 * b['eta']:.4f} | {b['A']:.2f} | "
                  f"{b['d_emit']:.5f} | {b['D']:.4f} | "
                  + (f"{100 * c['eta']:.4f} | {c['A']:.2f} | {c['D']:.4f}" if c else "n/a | | ")
                  + (f" | {100 * float(t['eta']):.4f} | {float(t['A']):.2f} | {float(t['d_emit']):.5f} | {float(t['D_emit']):.4f}" if t else " | not in tsv")
                  + f" | {'; '.join(flags)}\n")

    # ---------- validation rows against the mean-field toy ----------
    out.write("\n=== Validation rows (run-valid-0.tsv) against the mean-field toy (mean-field-scan-signed-modulus.tsv) ===\n")
    toy = toy_row()
    for r in sorted([r for r in rows if r["grid"] == "valid"], key=lambda r: (r["d0"], r["seed"])):
        ad2 = r["A"] * r["d_emit"] ** 2
        rtxt = ""
        if toy and abs(r["d0"] - float(toy["d0"])) < 1e-9 and int(r["seed"]) == 42:
            tA, tse, tde = float(toy["A"]), float(toy["seA"]), float(toy["d_emit"])
            z = (r["A"] - tA) / math.hypot(tse, r["seA_b"])
            rtxt = (f" | mean_field_toy.pl A {toy['A']} +- {toy['seA']} (z = {z:+.2f} with the toy's naive error; "
                    f"z = {(r['A'] - tA) / math.hypot(tse / 3, r['seA_b']):+.2f} with a third of it), d_emit {toy['d_emit']} "
                    f"(diff {r['d_emit'] - tde:+.4f}), A d^2 {toy['A.d2']}, d_rec {toy['d_rec']} vs {r['d_rec']:.4f}")
        out.write(f"N={r['N']:.0f} d0={r['d0']} eta={r['eta']} seed={r['seed']:.0f}: burn {r['burn']:.0f}, M {r['M']:.0f}, A {r['A']:.3f} "
                  f"(batch {r['seA_b']:.3f}, naive {r['seA']:.3f}), d_emit {r['d_emit']:.5f}, A d^2 {ad2:.4f}, sc/A^2 {r['sc_mom_A2']:.3f}, "
                  f"f_rec {r['f_rec']:.3f}{rtxt}\n")
    val30 = [r for r in rows if r["grid"] == "valid" and abs(r["d0"] - 0.30) < 1e-9]
    val50 = [r for r in rows if r["grid"] == "valid" and abs(r["d0"] - 0.50) < 1e-9]
    for nm, vv in (("d0 = 0.30", val30), ("d0 = 0.50", val50)):
        As = [r["A"] for r in vv]
        m = sum(As) / len(As)
        sd = math.sqrt(sum((x - m) ** 2 for x in As) / (len(As) - 1))
        out.write(f"   {nm}: mean A over three seeds {m:.3f}, seed sd {sd:.3f}, mean batch s.e. {sum(r['seA_b'] for r in vv) / 3:.3f}, "
                  f"mean naive s.e. {sum(r['seA'] for r in vv) / 3:.3f}\n")

    # ---------- flux balance, Eq. (15) of Paper II ----------
    out.write("\n=== Flux-law bookkeeping: A against A_ex - drift (identity), A_ex recomputed, and the other forms ===\n")
    worst = 0
    worst_ex = 0
    for r in rows:
        if r["d_emit"] <= 0:
            continue
        aex = 1 + (r["dw_mean"] - r["d_bar"]) / (r["eta"] * r["d_emit"])
        worst_ex = max(worst_ex, abs(aex / r["A_ex"] - 1))
        dev = abs((r["A_ex"] - r["drift"]) / r["A"] - 1)
        worst = max(worst, dev)
    out.write(f"max |(A_ex - drift)/A - 1| over {len(rows)} rows: {worst:.2e}   (print precision is 6 significant figures)\n")
    out.write(f"max |A_ex(recomputed from dw_mean, d_bar, eta, d_emit)/A_ex(column) - 1|: {worst_ex:.2e}\n")
    big = sorted(rows, key=lambda r: -abs(r["drift"] / r["A"]))[:5]
    out.write("largest |drift|/A: " + "; ".join(f"{lab(famkey(r))} eta={r['eta']:g}: drift/A = {r['drift'] / r['A']:+.4f}" for r in big) + "\n")
    refrows = [r for r in rows if r["grid"] in ("main", "modes", "fsize") and abs(r["eta"] - 0.0075) < 1e-9 and r["eps"] == 0
               and ((abs(r["d0"] - 0.283) < 1e-9 and abs(r["ca"] - 1.2) < 1e-9) or (abs(r["d0"] - 0.334) < 1e-9 and abs(r["ca"] - 1.7) < 1e-9))
               and int(r["q"]) == 6 and int(r["L"]) == 16 and int(r["seed"]) == 42]
    out.write("The reference families at eta = 0.0075 (L = 16, seed 42): A | n_p | n_r | 1+n_p/(1-n_r) | A/A_ex-1 | A/(A_ex-drift)-1 | "
              "A/A_toy-1 | A/A_B1-1 | A_B1 with d_emit in both places: A/A'-1 | A_ex with d0 in place of <d_w> | eta A d^2\n")
    for r in sorted(refrows, key=lambda r: (r["d0"], r["deliv"], r["labels"], r["k"])):
        b1alt = 1 + (r["d0"] - r["d_emit"]) / (r["eta"] * r["d_emit"])
        exd0 = 1 + (r["d0"] - r["d_bar"]) / (r["eta"] * r["d_emit"])
        out.write(f"   d0={r['d0']} {r['deliv']}/{r['labels']} k={r['k']:.0f}: {r['A']:.2f} | {r['n_p']:.3f} | {r['n_r']:.4f} | "
                  f"{1 + r['n_p'] / (1 - r['n_r']):.2f} | {r['A'] / r['A_ex'] - 1:+.4f} | {r['A'] / (r['A_ex'] - r['drift']) - 1:+.2e} | "
                  f"{r['A'] / r['A_toy'] - 1:+.3f} | {r['A'] / r['A_B1'] - 1:+.3f} | {r['A'] / b1alt - 1:+.3f} | {r['A'] / exd0 - 1:+.3f} | "
                  f"{r['eta'] * r['A'] * r['d_emit'] ** 2:.4f}\n")
    # ranges of the cruder form's offset over all rows with A >= 3, by labels
    for lb in ("off", "on"):
        sel = [r for r in rows if r["labels"] == lb and r["A"] >= 3 and r["grid"] != "valid"]
        v = [r["A"] / r["A_B1"] - 1 for r in sel]
        o = [r["A_B1"] / r["A"] - 1 for r in sel]
        out.write(f"   labels {lb}, all non-validation rows with A >= 3 (n = {len(sel)}): A/A_B1 - 1 from {min(v):+.3f} to {max(v):+.3f}; "
                  f"overprediction A_B1/A - 1 from {min(o):+.3f} to {max(o):+.3f}\n")
    for lb in ("off", "on"):
        sel = [r for r in rows if r["labels"] == lb and r["A"] >= 3 and r["grid"] in ("modes", "corner") and r["eps"] == 0]
        if sel:
            o = [r["A_B1"] / r["A"] - 1 for r in sel]
            out.write(f"   labels {lb}, modes and corner grids only (n = {len(sel)}): overprediction from {min(o):+.3f} to {max(o):+.3f}\n")
    for lb in ("off", "on"):
        sel = [r for r in rows if r["labels"] == lb and r["A"] >= 3 and r["grid"] != "valid" and r["eps"] == 0]
        v = [r["A"] / r["A_toy"] - 1 for r in sel]
        out.write(f"   labels {lb}: toy's form A/A_toy - 1 from {min(v):+.3f} to {max(v):+.3f} (n = {len(sel)})\n")
    # identity n_r: A = 1 + n_p/(1 - n_r)
    w = max(abs((1 + r["n_p"] / (1 - r["n_r"])) / r["A"] - 1) for r in rows if r["n_r"] < 1)
    out.write(f"max |(1 + n_p/(1 - n_r))/A - 1| over all rows: {w:.2e}; max n_r: {max(r['n_r'] for r in rows):.4f} "
              f"(at A = {max(rows, key=lambda r: r['n_r'])['A']:.1f})\n")

    # ---------- a necessary condition for marginality ----------
    # A kick of a primary triggers a site at the raw free volume d0 with probability exp(-ca/d0)/th, and a kick of a
    # repack (d_j <= d0) triggers less often; a site with d = 0 never triggers. A repack's k kicks therefore branch
    # at most k (1 - f_dead) exp(-ca/d0)/th times, and the cascade can become marginal only if this bound is at least one.
    out.write("\n=== A necessary condition for marginality: c_a <= d0 ln[k (1 - f_dead)/th], with f_dead the family's mean over its "
              "eta grid ===\n")
    for k in sorted(fam, key=lambda k: tuple(str(x) for x in k)):
        g, L, q, d0, ca, dv, lb, dreg, kk, th, eps, seed = k
        if g not in ("main",) or L != 16:
            continue
        sel = fam[k]
        fd = sum(r["f_dead"] for r in sel) / len(sel)
        b0 = d0 * math.log(kk / th)
        b1 = d0 * math.log(kk * (1 - fd) / th)
        maxA = max(r["A"] for r in sel)
        cr = res.get(k) is not None
        out.write(f"   d0={d0} ca={ca} q={q}: f_dead {fd:.3f}, bound without labels {b0:.3f}, with f_dead {b1:.3f}; "
                  f"ca {'<=' if ca <= b1 else '> '} bound; max A {maxA:.1f} at eta {min(r['eta'] for r in sel):g}; crossing {cr}\n")
    out.write(f"   coefficient ln(k/th): k=6 {math.log(600):.3f}, k=12 {math.log(1200):.3f}\n")

    # ---------- cutoff scaling ----------
    out.write("\n=== Cutoff scaling: slope of ln s_c (moment) against ln A per family, least squares; points with A >= 3 "
              "(as in lattice_analyze.py) and with A >= 10 ===\n")
    def slope(sel):
        xs = [math.log(r["A"]) for r in sel]
        ys = [math.log(r["sc_mom"]) for r in sel]
        n = len(xs)
        if n < 3:
            return float("nan")
        mx, my = sum(xs) / n, sum(ys) / n
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    for k in sorted(fam, key=lambda k: tuple(str(x) for x in k)):
        if k[0] not in ("main", "modes", "corner", "ca1"):
            continue
        s3 = [r for r in fam[k] if r["A"] >= 3]
        s10 = [r for r in fam[k] if r["A"] >= 10]
        if len(s3) < 3:
            continue
        scA2 = [r["sc_mom_A2"] for r in s3]
        out.write(f"   {lab(k)}: n={len(s3)} slope(A>=3) {slope(s3):.3f}; n={len(s10)} slope(A>=10) {slope(s10):.3f}; "
                  f"A {min(r['A'] for r in s3):.1f}-{max(r['A'] for r in s3):.1f}; sc/A^2 {min(scA2):.3f}-{max(scA2):.3f}\n")

    # ---------- d_emit spread over the eta grid ----------
    out.write("\n=== Spread of d_emit across each main-scan family's eta grid (max/min - 1), and the invariant eta A d^2 ===\n")
    for k in sorted(fam, key=lambda k: tuple(str(x) for x in k)):
        if k[0] != "main" or res.get(k) is None:
            continue
        sel = [r for r in fam[k] if r["A"] >= 3]
        de = [r["d_emit"] for r in sel]
        inv = [r["eta"] * r["A"] * r["d_emit"] ** 2 for r in sel]
        a = res[k][0]
        near = min(fam[k], key=lambda r: abs(math.log(r["eta"] / a["eta"])))
        Inear = near["eta"] * near["A"] * near["d_emit"] ** 2
        out.write(f"   {lab(k)}: d_emit {min(de):.4f}-{max(de):.4f} ({100 * (max(de) / min(de) - 1):.2f}%), A {min(r['A'] for r in sel):.1f}-"
                  f"{max(r['A'] for r in sel):.1f}; invariant {min(inv):.4f}-{max(inv):.4f} (spread {100 * (max(inv) / min(inv) - 1):.0f}%); "
                  f"I/2 at the nearest grid point {near['eta']:g} predicts eta* = {100 * Inear / 2:.3f}% against {100 * a['eta']:.3f}%\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
