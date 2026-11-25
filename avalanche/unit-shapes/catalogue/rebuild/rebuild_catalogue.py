"""An independent check of the catalogue of unit shapes (Paper II, Sec. 5.2 and Appendix C), from the run files.

Command (from the repository root; seconds; after build_results.py, before pair-holes/gen_holes.py):

    py -3 -X utf8 -I avalanche/unit-shapes/catalogue/rebuild/rebuild_catalogue.py

It writes rebuild_catalogue-results.txt (the report, also printed) and rebuild-crossings.tsv (one row per
shape with a crossing, read by pair-holes/gen_holes.py) beside this script. It reads every avalanche run
<shape>-eta<eta>.tsv in avalanche/unit-shapes/catalogue, the rod runs L<L>-eta<eta>.tsv in
avalanche/unit-shapes/rod and the checks <name>-eta<eta>.tsv in avalanche/unit-shapes/checks (all written by
lattice_sim.pl), the jamming runs rsa-*.tsv and lattice-L*-s*.tsv, the shape lists pieces2.json and pieces3.json,
and catalogue-results.tsv. Written with the standard library only, independently of build_results.py.

Sections: 1, the run files and the Ad2 column against its factors; 2, every crossing of A d_emit^2 = 2 by the
rule of build_results.py (ln(A d^2) linear in ln(eta) between the bracketing pair, every other column at the
same fraction, A = 2/d_emit^2 and D = 1/ln(1 + d_emit) at the crossing), with an error for each divisor,
against catalogue-results.tsv; 3, the counts over the distinct shapes (Paper II: 59 distinct shapes with a
crossing); 4, the divisor scatter between chiral twins and duplicate entries; 5, the five shapes in the target
window at the crossing; 6, the checks of the rod of four sites and of the 1 x 2 x 4 brick (Paper II, Sec. 5.2 and
App. C: the trigger-scale window 0.0033 to 0.013, at about 7 percent per factor of three in the trigger scale;
full-hole activation, since half the hole gives D = 7.378, outside the window; bulk random addition, since the
one-draw front gives D = 4.236); 6b, derived numbers (the site counts, up to 36, and the scale a0 = cH_dS/D of
Sec. 5.2).

Standings: target for D in the target divisor window 5.70 to 6.19 (Paper II, Sec. 5.2), empirical for the
empirical function's divisors 4.52 to 5.60 (App. C), and above, between or below otherwise. Shape names: rodL
is the 1 x 1 x L rod, pN-MM a generated piece of N sites, boxLMN an L x M x N block, and the other names are
those of polycubes.py. The checks: L4-th<th> is the rod at trigger scale th; L4-th0.01-ca0.375 the rod with
half its hole; L4front1-th0.01 the rod at the raw packing error of the one-draw front; brick124-th<th> the
1 x 2 x 4 brick at trigger scale th.
"""
import csv
import glob
import json
import math
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAT = os.path.dirname(HERE)
UM = os.path.dirname(CAT)

TARGET = (5.70, 6.19)      # the target divisor window of Paper II, Sec. 5.2
EMPIRICAL = (4.52, 5.60)   # the empirical function's divisors, Paper II, App. C
OUT = []


# ---------------- rotations and canonical forms of the shapes (independent of polycubes.py) ----------------

def matmul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)) for i in range(3))


ID = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
QZ = ((0, -1, 0), (1, 0, 0), (0, 0, 1))      # quarter turn about z
QX = ((1, 0, 0), (0, 0, -1), (0, 1, 0))      # quarter turn about x


def closure(gens):
    group, frontier = {ID}, [ID]
    while frontier:
        nxt = []
        for g in frontier:
            for h in gens:
                k = matmul(h, g)
                if k not in group:
                    group.add(k)
                    nxt.append(k)
        frontier = nxt
    return sorted(group)


ROT = closure([QZ, QX])                      # the 24 proper rotations, as the closure of two quarter turns
assert len(ROT) == 24
MIRROR = ((-1, 0, 0), (0, 1, 0), (0, 0, 1))


def act(M, c):
    return (M[0][0] * c[0] + M[0][1] * c[1] + M[0][2] * c[2],
            M[1][0] * c[0] + M[1][1] * c[1] + M[1][2] * c[2],
            M[2][0] * c[0] + M[2][1] * c[1] + M[2][2] * c[2])


def normalize(cells):
    m = [min(c[i] for c in cells) for i in range(3)]
    return tuple(sorted((c[0] - m[0], c[1] - m[1], c[2] - m[2]) for c in cells))


def orientation_set(cells, group=ROT):
    return sorted({normalize([act(M, c) for c in cells]) for M in group})


def canon(cells, group=ROT):
    return orientation_set(cells, group)[0]


BUILTIN = {   # the shapes built into polycubes.py, and the rods of latticebricks.py
    "rod2": [(0, 0, 0), (1, 0, 0)],
    "rod3": [(i, 0, 0) for i in range(3)],
    "rod4": [(i, 0, 0) for i in range(4)],
    "rod5": [(i, 0, 0) for i in range(5)],
    "rod6": [(i, 0, 0) for i in range(6)],
    "rod8": [(i, 0, 0) for i in range(8)],
    "plate22": [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
    "plate23": [(x, y, 0) for x in range(2) for y in range(3)],
    "plate33": [(x, y, 0) for x in range(3) for y in range(3)],
    "brick123": [(0, x, y) for x in range(2) for y in range(3)],
    "brick124": [(0, x, y) for x in range(2) for y in range(4)],
    "brick223": [(x, y, z) for x in range(2) for y in range(2) for z in range(3)],
    "Ltro": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
    "Ltet": [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0)],
    "Ttet": [(0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0)],
    "Stet": [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],
    "corner": [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
    "screw": [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)],
}


# ---------------- reading the runs and locating the crossings ----------------

def say(s=""):
    print(s)
    OUT.append(s)


def read_rows(path):
    with open(path, encoding="utf-8-sig") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[1:]]


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_families(folder, pattern, rename=None):
    fam = {}
    for p in glob.glob(os.path.join(folder, pattern)):
        m = re.match(r"(.+)-eta([\d.]+)\.tsv$", os.path.basename(p))
        if not m:
            continue
        name = rename(m.group(1)) if rename else m.group(1)
        for r in read_rows(p):
            r["_file"] = os.path.basename(p)
            fam.setdefault(name, []).append(r)
    for k in fam:
        fam[k].sort(key=lambda r: float(r["eta"]))
    return fam


INTERP = ["d_emit", "d_bar", "wall", "wall_f", "wall_l", "xi_int", "s_max", "sc_mom", "n_r", "f_rw",
          "d_rec", "f_loose", "A_ex", "nb_tot"]


def crossing(rows):
    pts = []
    for r in rows:
        A, d, eta = float(r["A"]), float(r["d_emit"]), float(r["eta"])
        pts.append((eta, A * d * d, r))
    signs = [(a[1] - 2) * (b[1] - 2) <= 0 and a[1] != b[1] for a, b in zip(pts, pts[1:])]
    nchg = sum(signs)
    if not nchg:
        return dict(n_sign_changes=0, maxA=max(float(r["A"]) for r in rows),
                    need=2 / min(float(r["d_emit"]) for r in rows) ** 2)
    i = signs.index(True)
    a, b = pts[i], pts[i + 1]
    la, lb = math.log(a[1]), math.log(b[1])
    t = (math.log(2) - la) / (lb - la)
    dl = math.log(b[0]) - math.log(a[0])
    eta = math.exp(math.log(a[0]) + t * dl)
    res = dict(n_sign_changes=nchg, t=t, eta=eta, eta_a=a[0], eta_b=b[0])
    for k in INTERP:
        va, vb = num(a[2].get(k)), num(b[2].get(k))
        if va is not None and vb is not None:
            res[k] = va + t * (vb - va)
    d = res["d_emit"]
    res["A"] = 2 / d ** 2
    res["A_lin"] = float(a[2]["A"]) + t * (float(b[2]["A"]) - float(a[2]["A"]))
    res["D"] = 1 / math.log(1 + d)
    # errors: ln(A d^2) from the batch-means error of A; d_emit from the two halves of each run
    sa, sb = float(a[2]["seA_b"]) / float(a[2]["A"]), float(b[2]["seA_b"]) / float(b[2]["A"])
    s_t = math.hypot((1 - t) * sa, t * sb) / abs(lb - la)
    res["se_eta_rel"] = s_t * abs(dl)
    sda = abs(float(a[2]["demit0"]) - float(a[2]["demit1"])) / 2
    sdb = abs(float(b[2]["demit0"]) - float(b[2]["demit1"])) / 2
    db = float(b[2]["d_emit"]) - float(a[2]["d_emit"])
    sd = math.sqrt(((1 - t) * sda) ** 2 + (t * sdb) ** 2 + (db * s_t) ** 2)
    res["se_d"] = sd
    res["se_D"] = sd / ((1 + d) * math.log(1 + d) ** 2)
    res["inputs"] = (float(a[2]["d0"]), float(a[2]["ca"]), int(float(a[2]["q"])), float(a[2]["th"]))
    return res


def standing(D):
    if TARGET[0] <= D <= TARGET[1]:
        return "target"
    if EMPIRICAL[0] <= D <= EMPIRICAL[1]:
        return "empirical"
    return "above" if D > TARGET[1] else ("between" if D > EMPIRICAL[1] else "below")


def shape_cells():
    cells = dict(BUILTIN)
    for fn in ("pieces2.json", "pieces3.json"):
        with open(os.path.join(CAT, fn), encoding="utf-8") as f:
            for k, v in json.load(f).items():
                cells[k] = [tuple(c) for c in v]
    return cells


def is_box(c):
    ext = [max(p[i] for p in c) - min(p[i] for p in c) + 1 for i in range(3)]
    return ext[0] * ext[1] * ext[2] == len(c)


def main():
    cat = load_families(CAT, "*-eta*.tsv")
    rods = load_families(os.path.join(UM, "rod"), "L*-eta*.tsv", rename=lambda s: "rod" + s[1:])
    fam = dict(cat)
    fam.update(rods)
    say(f"1. Run files: catalogue folder {sum(len(v) for v in cat.values())} rows in {len(cat)} families; "
        f"rod folder {sum(len(v) for v in rods.values())} rows in {len(rods)} families")
    say("   rows per catalogue family: " + str(sorted(set((len(v)) for v in cat.values()))) + "; families with 5 rows: "
        + ", ".join(sorted(k for k, v in cat.items() if len(v) == 5)))
    # check the Ad2 column against its factors
    worst = max(abs(float(r["Ad2"]) - float(r["A"]) * float(r["d_emit"]) ** 2) for v in fam.values() for r in v)
    say(f"   largest |Ad2 - A d_emit^2| over every row: {worst:.2e}")

    tab = {r["shape"]: r for r in read_rows(os.path.join(CAT, "catalogue-results.tsv"))}
    jam = {}
    for p in glob.glob(os.path.join(CAT, "rsa-*.tsv")):
        for r in read_rows(p):
            jam.setdefault(r["piece"], []).append(float(r["delta0"]))
    for p in glob.glob(os.path.join(UM, "lattice-L*-s*.tsv")):
        for r in read_rows(p):
            jam.setdefault("rod" + r["L"], []).append(float(r["delta0"]))

    cells = shape_cells()
    say("\n2. Crossings against catalogue-results.tsv (the rule of build_results.py)")
    say("shape      d0(run)  ca      q   eta*%    d_emit   A=2/d^2  D       se(D)  standing   | table: eta*%  d_emit  A      D       | flags")
    res = {}
    maxdiff = dict(eta=0, d=0, A=0, D=0)
    rows_out = ["shape\tsites\tq\td0_run\td0_jam_mean\tca\teta\td_emit\tA\tA_lin\tD\tse_D\tstanding\td_bar\twall\twall_f\twall_l\txi_int\ts_max\tsc_mom\tn_r\tf_rw\td_rec\tn_sign_changes"]
    for name in sorted(fam, key=lambda k: -(crossing(fam[k]).get("D") or 0)):
        c = crossing(fam[name])
        res[name] = c
        r = tab.get(name)
        flags = []
        d0_jam = st.mean(jam[name]) if name in jam else float("nan")
        if c["n_sign_changes"] == 0:
            say(f"{name:10s} no crossing; largest A {c['maxA']:.1f}, identity needs about {c['need']:.0f}; "
                f"table: {r['standing'] if r else 'absent'} {r['A'] if r else ''}")
            continue
        if c["n_sign_changes"] > 1:
            flags.append(f"{c['n_sign_changes']} sign changes")
        d0, ca, q, th = c["inputs"]
        if abs(round(d0_jam, 4) - d0) > 1e-4:
            flags.append(f"run d0 {d0} vs jamming mean {d0_jam:.4f}")
        s = standing(c["D"])
        if r:
            de = abs(c["eta"] - float(r["eta"]))
            dd = abs(c["d_emit"] - float(r["d_emit"]))
            dA = abs(round(c["A"], 1) - float(r["A"]))
            dD = abs(c["D"] - float(r["D"]))
            maxdiff["eta"] = max(maxdiff["eta"], de)
            maxdiff["d"] = max(maxdiff["d"], dd)
            maxdiff["A"] = max(maxdiff["A"], dA)
            maxdiff["D"] = max(maxdiff["D"], dD)
            if s != r["standing"]:
                flags.append(f"standing {s} vs table {r['standing']}")
            if abs(float(r["ca"]) - ca) > 1e-4 or int(r["q"]) != q:
                flags.append("inputs differ from table")
            rtxt = f"{100 * float(r['eta']):.3f}  {float(r['d_emit']):.4f}  {float(r['A']):6.1f} {float(r['D']):.3f}"
        else:
            rtxt = "absent"
        say(f"{name:10s} {d0:.4f}   {ca:.4f}  {q:2d}  {100 * c['eta']:.3f}   {c['d_emit']:.4f}   {c['A']:6.1f}   "
            f"{c['D']:.3f}   {c['se_D']:.3f}  {s:9s}  | {rtxt} | {'; '.join(flags)}")
        n = len(cells[name]) if name in cells else ""
        rows_out.append("\t".join(str(x) for x in [
            name, n, q, d0, f"{d0_jam:.4f}", ca, f"{c['eta']:.6f}", f"{c['d_emit']:.5f}", f"{c['A']:.2f}",
            f"{c['A_lin']:.2f}", f"{c['D']:.4f}", f"{c['se_D']:.4f}", s] + [f"{c.get(k, float('nan')):.5g}" for k in
            ("d_bar", "wall", "wall_f", "wall_l", "xi_int", "s_max", "sc_mom", "n_r", "f_rw", "d_rec")] + [c["n_sign_changes"]]))
    say(f"largest differences from catalogue-results.tsv: eta* {maxdiff['eta']:.1e}, d_emit {maxdiff['d']:.1e}, "
        f"A {maxdiff['A']:.2f}, D {maxdiff['D']:.1e}")
    missing = [k for k in tab if k not in res]
    say(f"rows of catalogue-results.tsv with no run family here: {missing}")
    with open(os.path.join(HERE, "rebuild-crossings.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows_out) + "\n")

    # 3. counts over distinct shapes
    say("\n3. Counts over the distinct shapes")
    groups = {}
    for name in res:
        groups.setdefault(canon(cells[name]), []).append(name)
    dup = {k: v for k, v in groups.items() if len(v) > 1}
    say(f"entries {len(res)}; distinct shapes {len(groups)}; duplicate groups {[sorted(v) for v in dup.values()]}")
    with_c = {k: v for k, v in groups.items() if res[v[0]]["n_sign_changes"] > 0}
    say(f"distinct shapes with a crossing: {len(with_c)}")
    cnt = {}
    for k, v in with_c.items():
        sts = {standing(res[n]["D"]) for n in v}
        if len(sts) > 1:
            say(f"  duplicates disagree in standing: {v} {sts}")
        cnt.setdefault(sorted(sts)[0], []).append(sorted(v)[0])
    for s in ("target", "between", "empirical", "above", "below"):
        say(f"  {s:9s} {len(cnt.get(s, [])):2d}  {sorted(cnt.get(s, []))}")
    boxes = {k: v for k, v in with_c.items() if is_box(cells[v[0]])}
    say(f"boxes with a crossing: {len(boxes)}; target {sum(standing(res[v[0]]['D']) == 'target' for v in boxes.values())}; "
        f"empirical {sum(standing(res[v[0]]['D']) == 'empirical' for v in boxes.values())}")
    nonbox = sorted((res[v[0]]["D"], v[0]) for k, v in with_c.items() if not is_box(cells[v[0]]))
    say("non-box shapes in or near either window (D < 6.4): " + ", ".join(
        f"{n} {D:.3f} ({standing(D)}, se {res[n]['se_D']:.3f})" for D, n in nonbox if D < 6.4))
    # mirror-merged counts
    free = {}
    for k, v in with_c.items():
        cs = cells[v[0]]
        mk = min(k, canon([act(MIRROR, p) for p in cs]))
        free.setdefault(mk, []).extend(v)
    say(f"with mirror twins merged: {len(free)} shapes with a crossing; target "
        f"{sum(standing(res[v[0]]['D']) == 'target' for v in free.values())}; empirical "
        f"{sum(standing(res[v[0]]['D']) == 'empirical' for v in free.values())}")
    four = [n for n, c in cells.items() if n in res and res[n]["n_sign_changes"] and is_box(c)
            and 4 in [max(p[i] for p in c) - min(p[i] for p in c) + 1 for i in range(3)]]
    say("boxes with a side of four: " + ", ".join(f"{n} {res[n]['D']:.3f} {standing(res[n]['D'])}" for n in sorted(four)))

    # 4. scatter from chiral twins and duplicates (same inputs except the jamming noise in d0)
    say("\n4. Divisor scatter between chiral twins and duplicate entries (same shape statistics)")
    pairs = [("p5-04", "p5-06"), ("p5-07", "p5-08"), ("p5-17", "p5-19"), ("p5-18", "p5-21"), ("p5-22", "p5-23"),
             ("p5-24", "p5-25"), ("rod5", "p5-02"), ("rod6", "p6-01"), ("brick123", "plate23"), ("brick123", "p6-03")]
    diffs = []
    for a, b in pairs:
        ra, rb = res[a], res[b]
        dd0 = ra["inputs"][0] - rb["inputs"][0]
        dD = ra["D"] - rb["D"]
        diffs.append(dD)
        say(f"  {a:8s} vs {b:8s}: d0 {ra['inputs'][0]:.4f} / {rb['inputs'][0]:.4f} (diff {dd0:+.4f}); "
            f"D {ra['D']:.3f} / {rb['D']:.3f} (diff {dD:+.3f}); dD/dd0 {dD / dd0 if dd0 else float('nan'):.1f}")
    say(f"  rms divisor difference over the pairs: {math.sqrt(sum(x * x for x in diffs) / len(diffs)):.3f}")

    # 5. secondary properties of the shapes in the target window. The last column estimates the crossing's efficiency from the
    # flux balance of Paper II, Eq. (15): with <d_w> = (1 - f_rw) d0, the standing mean taken as d_rec and
    # A = 2/d_emit^2, eta* = d_emit ((1 - f_rw) d0 - d_rec)/2.
    say("\n5. The five shapes in the target window at the crossing (the same rule; every column interpolated at the same fraction)")
    say("shape     eta*%   d_emit  A     D      d_bar   wall%  wall_f% wall_l% xi_int s_max  sc_mom n_r    f_rw   d_rec   "
        "flux-balance eta")
    for n in ("rod4", "brick124", "box145", "box234", "p6-05"):
        c = res[n]
        d0 = c["inputs"][0]
        f_flux = c["d_emit"] * ((1 - c["f_rw"]) * d0 - c["d_rec"]) / 2
        say(f"{n:9s} {100 * c['eta']:.3f}  {c['d_emit']:.4f}  {c['A']:.1f}  {c['D']:.3f}  {c['d_bar']:.4f}  "
            f"{100 * c['wall']:.2f}   {100 * c['wall_f']:.2f}    {100 * c['wall_l']:.2f}    {c['xi_int']:.2f}   "
            f"{c['s_max']:.0f}    {c['sc_mom']:.0f}    {c['n_r']:.4f} {c['f_rw']:.4f} {c['d_rec']:.4f}  "
            f"{100 * f_flux:.2f} %")
        for r in fam[n]:
            if float(r["eta"]) in (c["eta_a"], c["eta_b"]):
                say(f"            bracket eta {float(r['eta']):.4f}: wall {100 * float(r['wall']):.2f} %, "
                    f"xi_int {float(r['xi_int']):.2f}, d_bar {float(r['d_bar']):.4f}")

    # 6. the rod's checks
    say("\n6. The checks of the rod (checks folder)")
    chk = load_families(os.path.join(UM, "checks"), "*-eta*.tsv")
    chk["L4-th0.01"] = fam["rod4"]
    cr = {}
    for k in sorted(chk):
        c = crossing(chk[k])
        cr[k] = c
        d0, ca, q, th = (float(chk[k][0]["d0"]), float(chk[k][0]["ca"]), int(float(chk[k][0]["q"])),
                         float(chk[k][0]["th"]))
        if c["n_sign_changes"]:
            say(f"  {k:22s} d0 {d0} ca {ca} q {q} th {th}: eta* {100 * c['eta']:.3f} %, d_emit {c['d_emit']:.4f}, "
                f"A {c['A']:.1f}, D {c['D']:.3f} (se {c['se_D']:.3f})")
        else:
            say(f"  {k:22s} d0 {d0} ca {ca} q {q} th {th}: no crossing; largest A {c['maxA']:.2f}; "
                f"identity needs {c['need']:.1f} at the smallest d_emit; etas {[float(r['eta']) for r in chk[k]]}")
    def th_at(D, target):
        pts = sorted(D.items())
        for (t1, D1), (t2, D2) in zip(pts, pts[1:]):
            if (D1 - target) * (D2 - target) <= 0:
                f = (target - D1) / (D2 - D1)
                return math.exp(math.log(t1) + f * (math.log(t2) - math.log(t1)))
        return None

    cr["brick124-th0.01"] = res["brick124"]
    for tag, fams in (("rod", ("L4-th0.003", "L4-th0.01", "L4-th0.03")),
                      ("brick", ("brick124-th0.003", "brick124-th0.01", "brick124-th0.03"))):
        Dt = {th: cr[k]["D"] for th, k in zip((0.003, 0.01, 0.03), fams)}
        lo_, hi_ = th_at(Dt, TARGET[1]), th_at(Dt, TARGET[0])
        say(f"  trigger-scale window of the {tag} (ln th linear between the three runs): {lo_:.5f} to {hi_:.5f}, "
            f"a factor of {hi_ / lo_:.2f}; divisors {Dt[0.003]:.3f} / {Dt[0.01]:.3f} / {Dt[0.03]:.3f}; "
            f"change per factor of three {100 * (Dt[0.003] / Dt[0.01] - 1):.1f} and {100 * (Dt[0.01] / Dt[0.03] - 1):.1f} percent")
    D = {th: cr[k]["D"] for th, k in ((0.003, "L4-th0.003"), (0.01, "L4-th0.01"), (0.03, "L4-th0.03"))}
    s_lo = (D[0.003] - D[0.01]) / math.log(0.01 / 0.003)
    s_hi = (D[0.01] - D[0.03]) / math.log(0.03 / 0.01)
    say(f"  slopes dD/dln(th): {s_lo:.4f} (0.003 to 0.01), {s_hi:.4f} (0.01 to 0.03)")
    Db = {th: cr[k]["D"] for th, k in ((0.003, "brick124-th0.003"), (0.01, "brick124-th0.01"), (0.03, "brick124-th0.03"))}
    b_lo = (Db[0.003] - Db[0.01]) / math.log(0.01 / 0.003)
    b_hi = (Db[0.01] - Db[0.03]) / math.log(0.03 / 0.01)
    say(f"  the brick's slopes: {b_lo:.4f} and {b_hi:.4f}")
    say("  estimated trigger-scale windows of the other shapes in the target window, from the rod's two slopes and from the brick's:")
    for n in ("brick124", "box145", "box234", "p6-05"):
        D0 = res[n]["D"]
        txt = []
        for sl, sh in ((s_lo, s_hi), (b_lo, b_hi)):
            t_hi = 0.01 * math.exp((D0 - TARGET[0]) / sh)
            t_lo = 0.01 * math.exp(-(TARGET[1] - D0) / sl)
            txt.append(f"{t_lo:.4f} to {t_hi:.4f} (factor {t_hi / t_lo:.2f})")
        say(f"    {n:9s} D(0.01) {D0:.3f}: rod slopes {txt[0]}; brick slopes {txt[1]}")

    # 6b. derived numbers. cH_dS is 5.42 and 5.87 in units of 1e-10 m s^-2 at H0 = 67.4 and 73 km/s/Mpc
    # (Paper II, Sec. 5.2); 5.60 is the upper edge of the empirical function's divisors.
    say("\n6b. Derived numbers")
    emp = sorted(res[v[0]]["D"] for v in with_c.values() if standing(res[v[0]]["D"]) == "empirical")
    tgt = sorted(res[v[0]]["D"] for v in with_c.values() if standing(res[v[0]]["D"]) == "target")
    say(f"  empirical-window shapes: D {emp[0]:.3f} to {emp[-1]:.3f}; below 5.70 by {100 * (5.70 / emp[-1] - 1):.1f} to "
        f"{100 * (5.70 / emp[0] - 1):.1f} percent of their own divisor ({100 * (1 - emp[-1] / 5.70):.1f} to "
        f"{100 * (1 - emp[0] / 5.70):.1f} percent of 5.70)")
    say(f"  target-window shapes: D {tgt[0]:.3f} to {tgt[-1]:.3f}; above 5.60 by {100 * (tgt[0] / 5.60 - 1):.1f} to "
        f"{100 * (tgt[-1] / 5.60 - 1):.1f} percent")
    bysize = {}
    for k, v in groups.items():
        bysize.setdefault(len(cells[v[0]]), []).append(sorted(v)[0])
    say("  distinct shapes run, by site count: " + "; ".join(f"{n}: {len(v)}" for n, v in sorted(bysize.items())))
    for n in ("rod4", "brick124", "box145", "box234", "p6-05"):
        Dn = res[n]["D"]
        say(f"  {n:9s} a0 = cH/D for cH 5.42 to 5.87: {5.42 / Dn:.3f} to {5.87 / Dn:.3f}; against 5.60: "
            f"{100 * (Dn / 5.60 - 1):.1f} percent above")

    with open(os.path.join(HERE, "rebuild_catalogue-results.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
