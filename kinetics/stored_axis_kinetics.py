"""The kinetics of the stored axis: the interval between smoothings, the buildup of the stretch in unlayered fabric,
and the trail of the stored axes behind a moving galaxy.

A smoothing lays its domain's layer across the field and removes its excess, and a roughening adds the free volume
delta to the excess of a layered domain. By the identification of Paper II's section 4.1, a domain holds no lasting
excess until a smoothing has laid its layer. Averaged over the domains of a region, with p the share of domains that
hold a layer and m the mean excess over all domains,

    dp/dt = G (1 - p),        dm/dt = delta U p - G m,

with G = phi Gamma and U = (1 - phi) Gamma the rates of smoothing and roughening per domain, phi the guided share and
Gamma the total renewal rate. With p = 1 this is Paper II's Eq. (8). Time is counted in renewal events of a domain,
1/Gamma, which spans 190 to 650 yr in the tightened kinetic window of Paper II's section 4.1.

Part 1. The interval between two smoothings of a domain, at the crossover (boost 2) and at a boost of 28.
        Supports Paper II section 3.1: 1,200 to 4,100 yr and 2.7e4 to 9.3e4 yr.
Part 2. The time for the stretch to rise to 50 and 90 percent of its steady value in unlayered fabric about an isolated
        galaxy of 10^10.57 solar masses, with the share p growing from zero and with every domain layered from the
        start. Supports Paper II section 4.1: 1.0 to 3.6e4 yr at 20 kpc and 1.1 to 3.8e6 yr at 1 Mpc, and the boost
        of 28 at 200 kpc of section 3.1.
Part 3. The trail of the stored axes behind the same galaxy moving at 300 km/s when each new layer is laid across the
        transmitted flux: Delta = 2 v T, with T the interval between smoothings (Paper II section 3.1 and Appendix
        A.3, before the factor of 0.85 to 0.92 of kinetics/steady_dipole.py). Supports Paper II section 3.1: at most
        0.03 percent of the radius between 100 kpc and 1 Mpc.

The law is Paper II's Eq. (2) at the fiducial free volume of 0.19, with its scale a0 = c H_dS ln(1 + delta) =
0.9426e-10 m/s^2 at H0 = 67.4 km/s/Mpc.

Inputs: none. Output: kinetics/stored_axis_kinetics-results.txt.
Run from the repository root (a few seconds):
    python kinetics/stored_axis_kinetics.py > kinetics/stored_axis_kinetics-results.txt
"""
import math

DELTA = 0.19                       # the fiducial free volume
LAM = math.log(1.0 + DELTA)
RELAY_YR = (190.0, 650.0)          # years per renewal event of a domain, 1/Gamma, in the tightened window
G_N = 6.67430e-11
MSUN = 1.98847e30
MPC = 3.0856775814913673e22
KPC = MPC / 1.0e3
YR = 3.15576e7
PC_PER_YR_KMS = 977792.2           # 1 pc/yr in km/s
A0 = 0.9426e-10                    # m/s^2, the law's scale at the fiducial free volume
LOGM = 10.57                       # log10 of the galaxy's baryonic mass in solar masses
GM = G_N * 10.0 ** LOGM * MSUN
V_LENS = 300.0                     # km/s, the galaxy's speed through the fabric in Part 3


def phi_of(x):
    """The guided share at the total field x = g/a0."""
    return -math.expm1(-LAM * x)


def nu_law(x):
    """The boost at the total field x, Paper II Eq. (2)."""
    return 1.0 + DELTA / math.expm1(LAM * x)


def rule(title):
    print()
    print('=' * 100)
    print(title)
    print('=' * 100)


def steady_total(y):
    """The law's steady total field for a baryonic field y, both in units of a0: x = nu(x) y."""
    lo, hi = max(y, 1e-12), 1.0e4
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if nu_law(mid) * y > mid:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


# ---------------------------------------------------------------------------------------------------------------
# Part 1
# ---------------------------------------------------------------------------------------------------------------
rule('PART 1. THE INTERVAL BETWEEN SMOOTHINGS')
print('the law: mean excess delta (1 - phi) / phi, with (1 - phi) / phi roughenings between two smoothings of a domain')
for name, nu in [('the crossover, boost 2', 2.0), ('boost 28', 28.0)]:
    ph = DELTA / (nu - 1.0 + DELTA)
    nbar = (1.0 - ph) / ph
    print(f'  {name:32s} boost {nu:6.1f}, guided share {ph:.5f}, {nbar:6.1f} roughenings between smoothings, '
          f'{(nbar + 1) * RELAY_YR[0]:.2e} to {(nbar + 1) * RELAY_YR[1]:.2e} yr between smoothings')


# ---------------------------------------------------------------------------------------------------------------
# Part 2
# ---------------------------------------------------------------------------------------------------------------
def form_times(y, stored, dt=0.25, tmax=2.0e6):
    """Times, in renewal events of a domain, at which the mean excess reaches 50 and 90 percent of its steady value
    from zero. stored: the share p of layered domains grows from zero; otherwise every domain is layered."""
    xs = steady_total(y)
    m_ss = nu_law(xs) - 1.0
    p, m, t = (0.0 if stored else 1.0), 0.0, 0.0
    t50 = t90 = None
    while t < tmax:
        x = (1.0 + m) * y
        ph = phi_of(x)
        dp = ph * (1.0 - p) if stored else 0.0
        dm = DELTA * (1.0 - ph) * p - ph * m
        p += dt * dp
        m += dt * dm
        t += dt
        if t50 is None and m >= 0.5 * m_ss:
            t50 = t
        if m >= 0.9 * m_ss:
            t90 = t
            break
    return m_ss + 1.0, xs, t50, t90


rule('PART 2. THE TIME FOR THE STRETCH TO FORM IN UNLAYERED FABRIC')
print(f'an isolated body of 10^{LOGM} solar masses of baryons, with no ambient field; fields in units of a0 = {A0:.4e} m/s^2')
print(f'  radius | baryons\' field | steady boost | layered share from zero: t50, t90 (renewal events; yr at '
      f'{RELAY_YR[0]:.0f} and {RELAY_YR[1]:.0f}) | every domain layered: t50, t90')
for r_kpc in (20.0, 50.0, 100.0, 200.0, 400.0, 1000.0):
    y = GM / (r_kpc * KPC) ** 2 / A0
    nu_s, xs, a50, a90 = form_times(y, True)
    _, _, b50, b90 = form_times(y, False)
    fmt = lambda v: 'beyond 2e6' if v is None else f'{v:9.0f}'
    yrs = 'none' if a90 is None else f'{a90 * RELAY_YR[0]:.1e} to {a90 * RELAY_YR[1]:.1e} yr'
    print(f'  {r_kpc:6.0f} kpc | {y:.3e} | {nu_s:7.1f} | {fmt(a50)}, {fmt(a90)} ({yrs}) | {fmt(b50)}, {fmt(b90)}')

# ---------------------------------------------------------------------------------------------------------------
# Part 3
# ---------------------------------------------------------------------------------------------------------------
rule('PART 3. THE TRAIL OF THE STORED AXES BEHIND A MOVING GALAXY')
print('a domain keeps the axis of its last smoothing, so the arrangement of the axes trails a body that moves at v.')
print('With each new layer laid across the transmitted flux the trail is Delta = 2 v T, with T the interval between')
print(f'smoothings at the steady boost. The isolated body of Part 2, at {V_LENS:.0f} km/s; ranges run over the tightened window.')
print('  radius | steady boost | interval between smoothings (yr) | trail Delta (pc) | Delta over the radius')
for r_kpc in (20.0, 100.0, 400.0, 1000.0):
    y = GM / (r_kpc * KPC) ** 2 / A0
    xs = steady_total(y)
    nu_s = nu_law(xs)
    ph = phi_of(xs)
    T = [relay / ph for relay in RELAY_YR]
    d = [2.0 * V_LENS / PC_PER_YR_KMS * t for t in T]
    print(f'  {r_kpc:6.0f} kpc | {nu_s:6.1f} | {T[0]:.2e} to {T[1]:.2e} | {d[0]:.3g} to {d[1]:.3g} | '
          f'{d[0] / (r_kpc * 1.0e3):.4f} to {d[1] / (r_kpc * 1.0e3):.4f}')
