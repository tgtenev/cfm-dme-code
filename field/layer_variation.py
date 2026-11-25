"""The variation of the field from the uneven excess of the layers.

Each layer holds the excess of the roughenings since its last smoothing, so the excess varies from layer to layer.
Part 1 gives that spread in the steady state at a free volume of 0.19.
Part 2 gives the anisotropy of the Sun's own field under the anisotropic form of the ambient-stretch coupling, in
which the layers' conductivity acts on the Sun's field directly.
Part 3 gives what the scale-selective coupling leaves: the stretch acts on the field only through its mean over a
volume one catchment length across, so the only variation is the sampling noise of that mean over the domains of
the volume, at most 2 A^2 of them by Paper II's Appendix C.

Supports Paper I section 3.4 (three layers in ten with no excess and a mean excess of 0.44 at the Sun; 6,050
domains; 0.006 and 0.016 percent per domain; 0.5 and 1.3 percent; under 1e-3 of a wide binary's internal
acceleration; under 0.006 dex) and section 3.7 (the ratio of 0.69), and Paper II section 3.1 and Table 1 (6,050
domains; 0.5 percent at the Sun and 1.3 percent at a boost of 30), section 6.3 (0.69), Appendix A.2 (0.47 and 1.25
percent, 1.3 percent and under 0.006 dex, N of about 50, under 2e-4) and Appendix C (2 A^2 of 6,050 to 6,500, at
most 0.055 of a catchment length, 3.2 to 11 pc).

Inputs: none. Output: field/layer_variation-results.txt.
Run from the repository root (under a second; the script writes its output itself):
    python field/layer_variation.py
"""
import math
import pathlib

OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


# constants, SI
G = 6.674e-11
MSUN = 1.989e30
AU = 1.496e11
PC = 3.0857e16
YR = 3.156e7
KM = 1.0e3

DELTA = 0.19
A0 = 0.97e-10                 # the fitted scale, m/s^2
LAM = math.log(1 + DELTA)
A_AMP = 55.0                  # amplification at a free volume of 0.19
GAMMA_INV = (190.0, 650.0)    # interval between renewal events of a domain, yr, tightened window
LC = (58.0, 200.0)            # catchment length, pc
V_SUN = 369.8                 # speed of the Solar System in the frame of the microwave background, km/s
R_SUN = 8.2e3                 # the Sun's distance from the Galaxy's centre, pc


def store(X):
    """Guided share, boost, mean and rms excess at the drive X = g / a0."""
    phi = 1 - math.exp(-LAM * X)
    nu = 1 + DELTA / (math.exp(LAM * X) - 1)
    m = nu - 1
    rms = math.sqrt(m * (m + DELTA))
    return phi, nu, m, rms


def X_of_boost(nu):
    return math.log(1 + DELTA / (nu - 1)) / LAM


say("THE VARIATION OF THE FIELD FROM THE UNEVEN EXCESS OF THE LAYERS")
say()
say("1. The spread of the excess from layer to layer, in the steady state at a free volume of 0.19")
say("   place | drive X | total field, m/s^2 | guided share | boost | mean excess | rms excess | share of layers with no "
    "excess | rms of a layer's field over the mean total field")
SUN, CROSS = "the Sun, total field 2e-10 m/s^2", "the crossover"
cases = [(SUN, 2.0e-10 / A0), (CROSS, 1.0), ("boost 10", X_of_boost(10.0)), ("boost 30", X_of_boost(30.0))]
tab = {}
for name, X in cases:
    phi, nu, m, rms = store(X)
    g = X * A0
    gN = g / nu
    tab[name] = dict(X=X, phi=phi, nu=nu, m=m, rms=rms, g=g, gN=gN, sig=rms * gN)
    say(f"   {name} | {X:.3f} | {g:.3e} | {phi:.3f} | {nu:.3f} | {m:.3f} | {rms:.3f} | {phi:.3f} | {rms / nu:.3f}")
sun = tab[SUN]
say(f"   at the Sun the baryons' field is {sun['gN']:.3e} m/s^2, and a layer's field differs from the mean by "
    f"{sun['sig']:.3e} m/s^2 rms")
say(f"   two neighbouring layers at the Sun differ by {math.sqrt(2) * sun['sig']:.3e} m/s^2 rms, and by "
    f"{DELTA * sun['gN']:.3e} m/s^2 for one roughening")

say()
say("2. The Sun's own field under the anisotropic form of the ambient-stretch coupling: a point mass in the uniaxial "
    "medium of stretch l has the potential -GM / sqrt(l^2 x_par^2 + x_perp^2)")
say(f"   at the Sun's ambient stretch of {sun['nu']:.3f} the Sun's field along the Galaxy's field would be "
    f"{1 / sun['nu']:.3f} of its field across it")

say()
say("3. The scale-selective coupling: the stretch acts on the field averaged over a catchment length")
N = 2 * A_AMP ** 2
L_orb = 2 * math.pi * R_SUN * PC
V = V_SUN * KM
s_bin, M_bin = 1.0e4 * AU, 1.5 * MSUN          # a wide binary of 1.5 solar masses 1e4 AU apart
a_int = G * M_bin / s_bin ** 2
A_MOON = 3.844e8
say("3a. the sampling noise of the mean, averaged over a volume one catchment length across")
say(f"   a catchment length cubed holds at least 2 A^2 = {N:.0f} domains, by the cap on a domain's volume")
say("   place | sampling noise of the mean excess, as a share of the total field, at that largest domain")
noise = {}
for name in tab:
    r = tab[name]
    noise[name] = r["rms"] / (r["nu"] * math.sqrt(N))
    say(f"   {name} | {noise[name] * 100:.2f} percent, {math.log10(1 + noise[name]):.4f} dex")
dg = noise[SUN] * sun["g"]
say(f"   at the Sun that is {dg:.2e} m/s^2 on the scale of a catchment length")
one = DELTA * sun["gN"] / N
say(f"   one roughening of one such domain at the Sun shifts the mean field over the whole catchment length by "
    f"{one:.2e} m/s^2, {one / sun['g']:.1e} of the field")
for lc in LC:
    n_orb = L_orb / (lc * PC)
    t_cross = lc * PC / V / YR
    redraw = [t_cross / (g_ / sun["phi"]) for g_ in GAMMA_INV]
    say(f"   catchment length {lc:.0f} pc: an orbit at the Sun crosses {n_orb:.0f} of them, so the orbit's mean field is "
        f"off by {noise[SUN] / math.sqrt(n_orb) * 100:.3f} percent; a body crosses one in "
        f"{t_cross:.2e} yr, in which its mean is redrawn {min(redraw):.0f} to {max(redraw):.0f} times")
    grad = dg / (lc * PC)
    tide = sun["g"] / (R_SUN * PC)
    say(f"      across a pair 1e4 AU apart the noise differs by {grad * s_bin:.1e} m/s^2, {grad * s_bin / a_int:.1e} of "
        f"the pair's internal acceleration; across the Earth-Moon distance by {grad * A_MOON:.1e} m/s^2; its gradient "
        f"is {grad / tide:.2f} of the Galaxy's own, the field over the Sun's distance from the centre")
    # a sharp-edged window moved by d exchanges the share 2 d / lc of its domains, for layers thinner than d
    sharp = math.sqrt(2 * s_bin / (lc * PC)) * dg
    say(f"      those differences hold for a window with soft edges; with a sharp-edged window the difference across "
        f"the pair is at most {sharp:.1e} m/s^2, {sharp / a_int:.1e} of its internal acceleration")

say("3b. one domain against the volume")
lim = 10 ** 0.057 - 1
say(f"   the field: place | one roughening of one domain, as a share of the total field | the whole deviation of one "
    f"domain, rms | the sum over the {N:.0f} domains of one volume, {math.sqrt(N):.1f} times that | the loss of boost "
    f"by flux passing around volumes of high mean excess, of the order of that noise squared | the number of domains "
    f"at which the noise would attain 0.057 dex, {lim * 100:.1f} percent")
for name in tab:
    r = tab[name]
    rel = r["rms"] / r["nu"]
    one_r = DELTA * r["gN"] / N
    n_min = (rel / lim) ** 2
    say(f"   {name} | {one_r / r['g']:.1e}, {one_r:.2e} m/s^2 | {rel / N:.1e}, {rel / N * r['g']:.2e} m/s^2 | "
        f"{noise[name] * 100:.2f} percent | {noise[name] ** 2:.1e} | {n_min:.0f}, a margin of {N / n_min:.0f}")

say("3c. the values that Papers I and II quote")
for lc in (30.0, 200.0):
    n_ring = L_orb / (lc * PC)
    sharp = math.sqrt(2 * s_bin / (lc * PC)) * dg
    say(f"   Paper I's window, catchment length {lc:.0f} pc: a ring at the Sun's radius holds {n_ring:.0f} catchment "
        f"lengths and its mean is off by {noise[SUN] / math.sqrt(n_ring) * 100:.3f} percent; across a pair 1e4 AU apart a "
        f"sharp-edged window gives at most {sharp:.1e} m/s^2, {sharp / a_int:.1e} of the internal acceleration at 1.5 "
        f"solar masses and {sharp / (G * MSUN / s_bin ** 2):.1e} at 1 solar mass")
lim_deep = 1 / math.sqrt(N)
say(f"   in the limit of a large boost the noise of one volume tends to 1 over the root of {N:.0f}, "
    f"{lim_deep * 100:.2f} percent, {math.log10(1 + lim_deep):.4f} dex")
say(f"   a domain as thick as it is wide is at most {N ** (-1 / 3):.4f} of a catchment length across, "
    f"{LC[0] * N ** (-1 / 3):.1f} to {LC[1] * N ** (-1 / 3):.1f} pc in the tightened window; at the rod's amplification "
    f"of 57 a volume one catchment length across holds at least {2 * 57 ** 2} domains")
say(f"   at the Sun the mean excess is {sun['m']:.2f}, the mean number of roughenings {sun['m'] / DELTA:.1f}, and the "
    f"share of layers with no excess {sun['phi']:.2f}")

out = pathlib.Path(__file__).with_name("layer_variation-results.txt")
out.write_text("\n".join(OUT) + "\n", encoding="utf-8")
print("wrote field/layer_variation-results.txt")
