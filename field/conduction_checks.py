"""Two checks of the law's readout and guidance: conduction through an uneven store of excess, and guidance by
threshold laws other than the exponential.

1. The readout under an uneven store. Each cell holds the excess delta k, with k its run of unguided renewals,
   geometric with mean nbar and independent from cell to cell. On a grid of cells with conductivity 1/(1+e) along the
   field, the effective conductivity along the field is solved for four transverse laws (Paper I's crimp tensor 1+e,
   compliant interfaces 1, an isotropic softening 1/(1+e), fibres with no lateral exchange) and for resets that re-lay
   whole layers. Two dimensions and three, with a check at uniform excess. Conduction layer by layer, without the
   average over a catchment length, lets the flux pass around cells of high excess.
2. Guidance by a random threshold. A renewal event is guided when the field's contrast exceeds a threshold drawn at
   each event. Exponential thresholds give Paper I's Eq. (8); half-normal and uniform ones of the same mean give other
   laws.

Supports Paper I section 3.5 (part 2: departures of at most 0.017 and 0.035 dex; crossover drives of 0.174, 0.25 and
0.32) and Paper II Appendix A.2 (part 1, 3D: a nominal boost of 29.5, nbar = 150, cut to 0.57 of itself).
The random draws of part 1 come from one generator with a fixed seed, so the output repeats exactly.

Inputs: none. Output: field/conduction_checks-results.txt.
Run from the repository root (about 20 s; the script writes its output itself):
    python field/conduction_checks.py
"""
import math
import pathlib

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

OUT = []
DELTA = 0.19
rng = np.random.default_rng(20261119)


def say(s=""):
    print(s)
    OUT.append(s)


def conductivity_2d(s_par, s_perp):
    """Effective conductivity along axis 0 of a 2D grid, periodic along axis 1, electrodes at the ends of axis 0."""
    ny, nx = s_par.shape
    idx = np.arange(ny * nx).reshape(ny, nx)
    rows, cols, vals = [], [], []
    b = np.zeros(ny * nx)
    diag = np.zeros(ny * nx)
    g = (2.0 / (1.0 / s_perp + 1.0 / np.roll(s_perp, -1, axis=1))).ravel()
    a, c = idx.ravel(), np.roll(idx, -1, axis=1).ravel()
    rows += [a, c]; cols += [c, a]; vals += [-g, -g]
    np.add.at(diag, a, g); np.add.at(diag, c, g)
    g = (2.0 / (1.0 / s_par[:-1] + 1.0 / s_par[1:])).ravel()
    a, c = idx[:-1].ravel(), idx[1:].ravel()
    rows += [a, c]; cols += [c, a]; vals += [-g, -g]
    np.add.at(diag, a, g); np.add.at(diag, c, g)
    np.add.at(diag, idx[0], 2.0 * s_par[0])
    np.add.at(diag, idx[-1], 2.0 * s_par[-1]); np.add.at(b, idx[-1], 2.0 * s_par[-1])
    rows.append(np.arange(ny * nx)); cols.append(np.arange(ny * nx)); vals.append(diag)
    A = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ny * nx, ny * nx))
    phi = spla.spsolve(A.tocsc(), b).reshape(ny, nx)
    return np.sum(2.0 * s_par[0] * phi[0]) * ny / nx


def conductivity_3d(s_par, s_perp):
    nz, ny, nx = s_par.shape
    N = nz * ny * nx
    idx = np.arange(N).reshape(nz, ny, nx)
    rows, cols, vals = [], [], []
    diag = np.zeros(N)
    b = np.zeros(N)
    for axis in (1, 2):
        g = (2.0 / (1.0 / s_perp + 1.0 / np.roll(s_perp, -1, axis=axis))).ravel()
        a, c = idx.ravel(), np.roll(idx, -1, axis=axis).ravel()
        rows += [a, c]; cols += [c, a]; vals += [-g, -g]
        np.add.at(diag, a, g); np.add.at(diag, c, g)
    g = (2.0 / (1.0 / s_par[:-1] + 1.0 / s_par[1:])).ravel()
    a, c = idx[:-1].ravel(), idx[1:].ravel()
    rows += [a, c]; cols += [c, a]; vals += [-g, -g]
    np.add.at(diag, a, g); np.add.at(diag, c, g)
    np.add.at(diag, idx[0].ravel(), (2.0 * s_par[0]).ravel())
    np.add.at(diag, idx[-1].ravel(), (2.0 * s_par[-1]).ravel()); np.add.at(b, idx[-1].ravel(), (2.0 * s_par[-1]).ravel())
    rows.append(np.arange(N)); cols.append(np.arange(N)); vals.append(diag)
    A = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    phi, info = spla.cg(A, b, rtol=1e-10, maxiter=40000)
    assert info == 0, info
    phi = phi.reshape(nz, ny, nx)
    return np.sum(2.0 * s_par[0] * phi[0]) * nz / (nx * ny)


def runs(nbar, shape):
    return rng.geometric(1.0 / (1.0 + nbar), size=shape) - 1


# 1. the readout under an uneven store
say("1. The readout under an uneven store")
e = np.full((30, 30, 30), 1.9)
say(f"   check, uniform excess 1.9 in 3D: crimp boost {1 / conductivity_3d(1 / (1 + e), 1 + e):.4f} against 2.9")
say("   2D, 160 x 160, two realizations: nbar | 1+<e> | harmonic mean of 1+e | crimp | interfaces | isotropic | fibres | whole layers reset")
for nbar in (1.0, 5.0, 30.0, 150.0):
    res = {k: [] for k in ("kern", "harm", "crimp", "interface", "isotropic", "fibre", "layer")}
    for _ in range(2):
        e = DELTA * runs(nbar, (160, 160))
        res["kern"].append(1 + e.mean()); res["harm"].append(1 / np.mean(1 / (1 + e)))
        sp_ = 1 / (1 + e)
        res["crimp"].append(1 / conductivity_2d(sp_, 1 + e))
        res["interface"].append(1 / conductivity_2d(sp_, np.ones_like(e)))
        res["isotropic"].append(1 / conductivity_2d(sp_, sp_))
        res["fibre"].append(1 / conductivity_2d(sp_, 1e-9 * np.ones_like(e)))
        T = rng.exponential(nbar, size=(160, 1))
        el = DELTA * rng.poisson(np.repeat(T, 160, axis=1))
        res["layer"].append((1 / conductivity_2d(1 / (1 + el), 1 + el)) / (1 + el.mean()))
    m = {k: np.mean(v) for k, v in res.items()}
    say(f"   {nbar:5g} | {m['kern']:.2f} | {m['harm']:.2f} | {m['crimp']:.2f} ({m['crimp'] / m['kern']:.2f}) | "
        f"{m['interface']:.2f} ({m['interface'] / m['kern']:.2f}) | {m['isotropic']:.2f} ({m['isotropic'] / m['kern']:.2f}) | "
        f"{m['fibre']:.2f} ({m['fibre'] / m['kern']:.2f}) | {m['layer']:.3f} of its own 1+<e>")
say("   3D, 40^3: nbar | 1+<e> | harmonic mean | crimp | isotropic")
for nbar in (5.0, 30.0, 150.0):
    e = DELTA * runs(nbar, (40, 40, 40))
    kern, harm = 1 + e.mean(), 1 / np.mean(1 / (1 + e))
    bc = 1 / conductivity_3d(1 / (1 + e), 1 + e)
    bi = 1 / conductivity_3d(1 / (1 + e), 1 / (1 + e))
    say(f"   {nbar:5g} | {kern:.2f} | {harm:.2f} | {bc:.2f} ({bc / kern:.2f}) | {bi:.2f} ({bi / kern:.2f})")

# 2. guidance by a random threshold
say()
say("2. Guidance by a random threshold of mean 1 in units of x = g/(cH_dS)")
SIG = math.sqrt(math.pi / 2)
QS = {"exponential": lambda x: math.exp(-x), "half-normal": lambda x: math.erfc(x / (SIG * math.sqrt(2))),
      "uniform on [0, 2]": lambda x: max(0.0, 1 - x / 2)}


def make_law(Q):
    def nu(x):
        q = Q(x)
        return 1 + DELTA * q / (1 - q) if q < 1 else float("inf")
    lo, hi = 1e-9, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if nu(mid) > 2 else (lo, mid)
    return nu, 0.5 * (lo + hi)


def total_field(nu, x0, Y):
    lo, hi = 1e-12, 1e6
    for _ in range(300):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if mid / nu(mid * x0) < Y else (lo, mid)
    return math.sqrt(lo * hi)


Ys = [10 ** (k / 10) for k in range(-30, 11)]
laws = {k: make_law(q) for k, q in QS.items()}
ref = [total_field(*laws["exponential"], Y) for Y in Ys]
for k, (nu, x0) in laws.items():
    Xs = [total_field(nu, x0, Y) for Y in Ys]
    say(f"   {k}: crossover at x = {x0:.4f}; largest departure from Paper I's Eq. (8) for Y = 1e-3 to 10: "
        f"{max(abs(math.log10(a / b)) for a, b in zip(Xs, ref)):.3f} dex; X/sqrt(Y) at Y = 1e-3: {Xs[0] / math.sqrt(Ys[0]):.3f}")

out = pathlib.Path(__file__).with_name("conduction_checks-results.txt")
out.write_text("\n".join(OUT) + "\n", encoding="utf-8")
print("wrote field/conduction_checks-results.txt")
