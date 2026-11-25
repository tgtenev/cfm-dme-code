"""The first-order steady trail of the stored axes behind a moving body, from the dipole equation, against the
extent of the arrangement.

Three rules for the direction of a new layer are compared (Paper II sections 3.1 and A.3):
    rule A  across the coarse-grained field, the potential gradient
    rule B  across the gradient inside a freshly smoothed, uncrimped domain
    rule C  across the transmitted flux, the identification that Paper II adopts
l1.L1.steady solves equation (2) of l1.py: the stored axis is the rule's axis 2 v T upstream, the stretch is the law
v T upstream, and the field obeys the dipole part of the field equation. The trail of the axes is reported over the
predicted trail 2 v T nu^p (p = 2, 1, 0 for rules A, B, C) at several radii, for an arrangement that ends at r_out,
beyond which the fabric holds the static arrangement about the body (phi1 = 0 at r_out). r_out = 5.4 R is the grid of
moving_body.py; the larger r_out extend the deep-regime arrangement with the stretch uncapped outside the body (it
reaches 45 to 450). At first order the trail is in proportion to v, so the prefactor does not depend on v.

In the deep regime outside the body (nu proportional to r, T proportional to nu) equation (2) has the solutions
phi1 = r and 1/r of linearized AQUAL, and the particular solutions give for an arrangement pinned at r_out >> r:
    rule A: trail / (2 v T nu^2) = (r_out/r - 1)/3,
    rule B: trail / (2 v T nu)   = ln(r_out/r)/2,
    rule C: trail / (2 v T)      = 1 - (corrections from the body and from r_out).
The script prints those forms beside the numbers, at the fiducial free volume of 0.19.

Supports Paper II section 3.1 and Appendix A.3: under rule C the trail is 0.85 to 0.92 times 2 v T inside 5 radii
for r_out = 50 and 100 R; under rule B it grows as (1/2) ln(R_out/r) times 2 v T nu; under rule A it grows linearly
with the extent of the arrangement.

Inputs: none (imports kinetics/axfem.py and kinetics/l1.py). Output: kinetics/steady_dipole-results.txt.
Run from the repository root (a few seconds):
    python kinetics/steady_dipole.py --out kinetics/steady_dipole-results.txt
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import axfem as A  # noqa: E402
import l1  # noqa: E402

OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
DELTA = 0.19                     # the fiducial free volume
RADII = (0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 40.0)
_lines = []


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def main():
    out("FIRST-ORDER STEADY TRAIL OF THE AXES OVER THE PREDICTED TRAIL 2 v T nu^p, AGAINST THE EXTENT OF THE ARRANGEMENT")
    out("=" * 118)
    out("body of moving_body.py (least stretch 4); the fabric beyond r_out holds the static arrangement about the body.")
    out("deep-regime forms: A (r_out/r - 1)/3, B ln(r_out/r)/2, C 1.")
    delta = DELTA
    law = A.Law(delta)
    body = A.Body(law)
    out("")
    out(f"delta = {delta}")
    for r_out in (5.4, 10.0, 20.0, 50.0, 100.0):
        P = l1.L1(body, r_out, cap_r=1.0)
        out(f"  r_out = {r_out:5.1f} R (stretch {P.nu[-1]:.1f} there)")
        for q in (0, 1, 2):
            ta, tf, pr, _ = P.steady(q, 1.0)
            cells = []
            for rr in RADII:
                if rr >= 0.95 * r_out:
                    continue
                val = float(P.at(ta, rr) / P.at(pr, rr))
                form = ((r_out / rr - 1.0) / 3.0, 0.5 * np.log(r_out / rr), 1.0)[q]
                cells.append(f"{rr:g}: {val:.3f} ({form:.2f})")
            out(f"    rule {'ABC'[q]}: " + "; ".join(cells))
    P = l1.L1(body, 5.4)
    out("  field trail over axis trail at r_out = 5.4 R (the axes trail the field by 2 v T / nu^q):")
    for q in (0, 1, 2):
        ta, tf, pr, _ = P.steady(q, 1.0)
        out(f"    rule {'ABC'[q]}: " + "; ".join(f"{rr:g}: {float(P.at(tf, rr) / P.at(ta, rr)):.4f}"
                                              for rr in RADII if rr < 5.2))
    if OUT:
        with open(OUT, "w", newline="\n", encoding="utf-8") as fh:
            fh.write("\n".join(_lines) + "\n")


if __name__ == "__main__":
    main()
