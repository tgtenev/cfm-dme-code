"""The crossings of the runs with pair-specific turning holes (Paper II, Appendix C).

Command (from the repository root; seconds; after the runs of lattice_sim_pair.pl):

    py -3 -X utf8 -I avalanche/unit-shapes/pair-holes/summarize.py > avalanche/unit-shapes/pair-holes/summary.txt

It reads the run files <family>-eta<eta>.tsv beside this script, one family per shape, same-label convention
(min, mean or max; see lattice_sim_pair.pl) and trigger scale, and locates each family's crossing of
A d_emit^2 = 2 by the rule of build_results.py (ln(A d_emit^2) linear in ln eta between the bracketing runs,
d_emit at the same fraction, A = 2/d_emit^2, D = 1/ln(1 + d_emit)). It writes pair-holes-results.tsv beside
this script and prints one line per family, with its standing: target for D in the target divisor window
5.70 to 6.19 (Paper II, Sec. 5.2), empirical for the empirical function's divisors 4.52 to 5.61 (App. C), and
above, between or below otherwise. It then prints the trigger-scale windows of the rod of four sites and of
the W pentomino (p5-11), by log-linear interpolation of the divisor against the trigger scale.

Paper II results it supports (App. C): only the rod lands in the target window under every definition of the
turning hole (rod4-mean, D = 5.802); four other shapes enter under minimum-cost reorientation (brick124-min
5.927, box145-min 5.945, box234-min 5.723, p6-05-min 5.741), and still others under the largest change or a
plain mean (brick123-max 6.116, p5-04, p5-06, p5-07, p5-08, p5-11, p5-12 and p5-28 under max at 6.01 to 6.11,
p5-11-mean 6.155); the rod's trigger-scale window, 0.0033 to 0.0129 (Sec. 5.2: 0.0033 to 0.013).
"""
import csv, glob, math, os, re, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = (5.70, 6.19)      # the target divisor window of Paper II, Sec. 5.2
EMPIRICAL = (4.52, 5.61)   # the empirical function's divisors, Paper II, App. C
fam = defaultdict(list)
for p in glob.glob(os.path.join(HERE, "*-eta*.tsv")):
    m = re.match(r"(.+)-eta([\d.]+)\.tsv", os.path.basename(p))
    if not m:
        continue
    with open(p, encoding="utf-8-sig") as f:
        lines = [l.rstrip("\r\n").split("\t") for l in f if l.strip()]
    if len(lines) < 2:
        continue
    r = dict(zip(lines[0], lines[1]))
    fam[m.group(1)].append((float(r["eta"]), float(r["A"]), float(r["d_emit"]), float(r["Ad2"]), r))


def crossing(rows):
    rows = sorted(rows, key=lambda t: t[0])
    for a, b in zip(rows, rows[1:]):
        if (a[3] - 2) * (b[3] - 2) <= 0 and a[3] != b[3]:
            t = (math.log(2) - math.log(a[3])) / (math.log(b[3]) - math.log(a[3]))
            eta = math.exp(math.log(a[0]) + t * (math.log(b[0]) - math.log(a[0])))
            dem = a[2] + t * (b[2] - a[2])
            return eta, dem, 2 / dem ** 2, 1 / math.log(1 + dem)
    return None


def standing(D):
    return ("target" if TARGET[0] <= D <= TARGET[1] else "empirical" if EMPIRICAL[0] <= D <= EMPIRICAL[1] else
            "above" if D > TARGET[1] else "between" if D > EMPIRICAL[1] else "below")


sys.stdout.reconfigure(encoding="utf-8")
out = []
for name in sorted(fam):
    c = crossing(fam[name])
    r0 = fam[name][0][4]
    out.append(dict(family=name, q=r0["q"], d0=r0["d0"], th=r0["th"], same=r0.get("same", "-"),
                    c_min=r0.get("c_min", ""), c_mean=r0.get("c_mean", ""), c_max=r0.get("c_max", ""),
                    eta=round(c[0], 5) if c else "", d_emit=round(c[1], 4) if c else "",
                    A=round(c[2], 1) if c else "", D=round(c[3], 3) if c else "",
                    standing=standing(c[3]) if c else "no crossing", runs=len(fam[name])))
with open(os.path.join(HERE, "pair-holes-results.tsv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()), delimiter="\t")
    w.writeheader(); w.writerows(out)
for r in out:
    print(f"{r['family']:22s} q={r['q']:>2} th={r['th']:<6} same={r['same']:<4} D={r['D']!s:<6} {r['standing']}")


def window(points, lo=5.70, hi=6.19):
    """points: [(th, D)], D falling with th; returns (th at D = hi, th at D = lo) by log-linear interpolation,
    extrapolating from the nearest pair when the edge lies outside the scanned range."""
    pts = sorted(points)
    def at(Dt):
        for (t1, d1), (t2, d2) in zip(pts, pts[1:]):
            if (d1 - Dt) * (d2 - Dt) <= 0:
                f = (math.log(d1) - math.log(Dt)) / (math.log(d1) - math.log(d2))
                return math.exp(math.log(t1) + f * (math.log(t2) - math.log(t1))), False
        (t1, d1), (t2, d2) = (pts[0], pts[1]) if Dt > pts[0][1] else (pts[-2], pts[-1])
        f = (math.log(d1) - math.log(Dt)) / (math.log(d1) - math.log(d2))
        return math.exp(math.log(t1) + f * (math.log(t2) - math.log(t1))), True
    return at(hi), at(lo)


# The rod's divisors at trigger scales 0.003, 0.01 and 0.03, with lattice_sim.pl and its single hole 3/4: the
# crossings of checks/L4-th0.003-eta*.tsv, rod/L4-eta*.tsv and checks/L4-th0.03-eta*.tsv, as section 6 of
# catalogue/rebuild/rebuild_catalogue-results.txt prints them.
rodD = {0.003: 6.222, 0.01: 5.807, 0.03: 5.363}
wD = {}
for name, th in (("p5-11-mean-th0.003", 0.003), ("p5-11-mean", 0.01), ("p5-11-mean-th0.03", 0.03)):
    c = crossing(fam[name]); wD[th] = 1 / math.log(1 + c[1])
for label, pts in (("rod of four sites", rodD), ("W pentomino (same 'mean')", wD)):
    (a, ea), (b, eb) = window(list(pts.items()))
    print(f"trigger-scale window, {label}: {a:.4f}{' (extrapolated)' if ea else ''} to {b:.4f}{' (extrapolated)' if eb else ''}"
          f"; divisors {', '.join(f'{k}: {v:.3f}' for k, v in sorted(pts.items()))}")
