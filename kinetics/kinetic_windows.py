"""The kinetic window, the relaxation times, the residence screen, the renewal domains and the stability margin of
the renewal's noise, at the fiducial free volume of 0.19 and H0 = 67.4 km/s/Mpc.

Supports Paper II: the amplification A of section 2; the tilt tolerance and the boosts of 28 and 90 of sections 3.1
and 7.3; the kinetic window and the relaxation times of section 4.1; the residence screen of section 4.2 and the
radius of the residence wedge in the caption of Figure 2; the margins of section 5.3 and Table 1; the relaxation time
of section 6.5 and Table 4; and the size of a renewal domain in Appendix C. It also prints the fiducial divisor,
dilute slope and hardening onset of Paper I's section 3.5, which Paper II's sections 5.1 and 5.2 quote, and the memory
time and renewal count that Paper I's section 3.4 quotes.
Each printed line names the paper's location and printed value, and says whether the computed value agrees with it
at the paper's rounding. A line marked "bound" checks that the value lies within the paper's bound.

Formulas:
  lambda = ln(1 + delta), divisor D = 1/lambda, dilute slope s = lambda/delta, hardening onset xi*/lambda with
      exp(xi*) - 1 = sqrt(2) xi*                                                     [Paper I section 3.5]
  amplification A = 2/(Z1 delta^2) with Z1 = 1                                       [Paper II Eq. (4)]
  window: tau_c <= l_c/c at the ceilings l_c = 3 kpc and 200 pc; tau_c >= delta tau_fast (n >= 1);
      eps* = 3 H_dS tau_c; tau_eq <= tau_c/delta; n = tau_c/(delta tau_fast)        [Paper II Eqs. (8), (9)]
  relaxation: tau_rel/tau_eq = delta / [phi + X lambda exp(-lambda X) (nu - 1 + delta)/nu],
      the ratio form of Paper II Eq. (10), since G = phi Gamma and tau_eq = 1/(Gamma delta)
  residence: t_res = 2 R / v; the pattern builds at most 1 - exp(-t_res/tau_rel)    [Paper II section 4.2]
  nesting: a residence-screened system spans at most 6 v/(c delta) of the catchment length at X >= 0.1
  domains: a <= l_c / (2 A^2)^(1/3)                                                  [Paper II Appendix C]
  stability: margin n Z against the ordering point of the layers                    [Paper II section 5.3]
  tilt: a layer tilted by beta from the field trims a boost L by 10 percent where
      sin^2(beta) = (1/0.9 - 1)/(L^2 - 1)                                            [Paper II section 7.3]
The ordering point T* = 9.7494 misfit energies is the Monte Carlo value of field/stability_threshold_mc-fcc0.txt, and
15.50 is the top of the band of field/stability_threshold_calc-results.txt; both are typed in here.

Inputs: none. Output: kinetics/kinetic_windows-results.txt.
Run from the repository root (under a second):
    python kinetics/kinetic_windows.py > kinetics/kinetic_windows-results.txt
"""
import math

C = 2.99792458e8                    # m/s
PC = 3.0856775814913673e16          # m
AU = 1.495978707e11                 # m
YR = 3.15576e7                      # s, Julian year
MPC = 1.0e6 * PC
KPC = 1.0e3 * PC
OMEGA_L = 0.6847
GM_SUN = 1.32712440018e20           # m^3/s^2
C_PC_YR = C * YR / PC               # pc per yr
DELTA = 0.19                        # the fiducial free volume
H0 = 67.4                           # km/s/Mpc
TAU_FAST = 1.0e3                    # yr, the adopted fast edge of the equilibration time
L_CEIL, L_TIGHT = 3000.0, 200.0     # pc, the locality ceiling and the ceiling from the classical dwarf spheroidals
A0_LAW = 0.9426e-10                 # m/s^2, c H_dS ln(1 + delta) at the fiducial free volume
A0_FIT = 0.973e-10                  # m/s^2, the constrained best-fit scale at H0 = 67.4 (Paper I section 5.3)
LOG_M_GAL = 10.57                   # log10 of the galaxy's baryonic mass in solar masses (Paper II sections 3.1, 7.3)
V_SUN = 370.0                       # km/s, the Solar System's speed in the microwave background's frame
V_WB = 369.8                        # km/s, the same speed as the wide-binary line takes it
V_FAST = 600.0                      # km/s, about the Local Group's peculiar speed
T_STAR = 9.7494                     # the ordering point of staggered stacks, in misfit energies
T_TOP = 15.50                       # the top of the band of the ordering point


def h_ds(h0):
    return math.sqrt(OMEGA_L) * h0 * 1.0e3 / MPC          # s^-1


LAM = math.log1p(DELTA)
TH = 3.0 * h_ds(H0) * YR                                   # 3 H_dS in yr^-1


def xi_star():
    lo, hi = 0.1, 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.expm1(mid) - math.sqrt(2.0) * mid < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


XI = xi_star()


def nu_law(x):
    """The boost at the total field x = g/a0, Paper II Eq. (2)."""
    return 1.0 + DELTA / math.expm1(LAM * x)


def tauhat(x):
    """tau_rel / tau_eq at the drive x, Paper II Eq. (10)."""
    phi = -math.expm1(-LAM * x)
    nu = nu_law(x)
    return DELTA / (phi + x * LAM * math.exp(-LAM * x) * (nu - 1.0 + DELTA) / nu)


def drive_where_tauhat(target):
    """The drive below which tau_rel/tau_eq exceeds target (tauhat falls monotonically with X)."""
    lo, hi = 1e-6, 10.0
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if tauhat(mid) > target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def radius_of_boost(nu, log_m, a0):
    """The radius, in kpc, at which the law gives the boost nu about an isolated point mass: the total field
    x solves nu_law(x) = nu, and the baryonic field there is x/nu in units of a0."""
    x = math.log1p(DELTA / (nu - 1.0)) / LAM
    g_n = x / nu * a0
    return math.sqrt(GM_SUN * 10.0 ** log_m / g_n) / KPC


def tilt_deg(boost):
    """The tilt of a layer from the field that trims the boost by 10 percent."""
    return math.degrees(math.asin(math.sqrt((1.0 / 0.9 - 1.0) / (boost * boost - 1.0))))


def line(label, name, value, spec, where, paper=None, tol=None, bound=False, note=None):
    """One printed quantity. paper is the value as the paper prints it; tol is half its last printed digit; bound
    checks value <= paper."""
    text = "[%s] %s: %s" % (label, name, format(value, spec))
    if paper is not None:
        pv = float(paper.replace(",", ""))
        if bound:
            ok = "yes" if value <= pv else "NO"
            text += " || %s: at most %s; within the bound: %s" % (where, paper, ok)
        else:
            ok = "yes" if abs(value - pv) <= tol * (1.0 + 1e-12) else "no"
            text += " || %s: %s; agrees at the paper's rounding: %s" % (where, paper, ok)
    elif where:
        text += " || %s" % where
    print(text)
    if note:
        print("       " + note)




tc_ceil, tc_tight = L_CEIL / C_PC_YR, L_TIGHT / C_PC_YR
tc_floor = DELTA * TAU_FAST
teq_slow, teq_tight = tc_ceil / DELTA, tc_tight / DELTA
n_tight = tc_tight / (DELTA * TAU_FAST)
A = 2.0 / DELTA ** 2
P1, P2 = "Paper I, section ", "Paper II, section "

print("THE KINETIC WINDOW, RELAXATION, RESIDENCE SCREEN, RENEWAL DOMAINS AND STABILITY MARGIN")
print("=" * 112)
print("free volume delta = %.2f at H0 = %.1f km/s/Mpc, Omega_Lambda = %.4f" % (DELTA, H0, OMEGA_L))
print("c H_dS = %.4fe-10 m/s^2; 3 H_dS = %.4e /yr; c = %.6f pc/yr; tau_fast = %.0f yr; l_c ceilings %.0f and %.0f pc"
      % (C * h_ds(H0) * 1e10, TH, C_PC_YR, TAU_FAST, L_CEIL, L_TIGHT))
print("xi* = %.5f (exp(xi) - 1 = sqrt(2) xi)" % XI)

print()
print("S1. THE LAW'S CONSTANTS")
print("-" * 112)
line("S1.1", "lambda = ln(1 + delta)", LAM, ".4f", "")
line("S1.2", "divisor D = 1/lambda against c H_dS (kappa = 1)", 1.0 / LAM, ".3f", P2 + "5.2 and Table 2", "5.75", 0.005)
line("S1.3", "dilute slope s = lambda/delta", LAM / DELTA, ".4f", P1 + "3.5; " + P2 + "5.1", "0.92", 0.005)
line("S1.4", "onset of the hardening, xi*/lambda", XI / LAM, ".3f", P1 + "3.5; " + P2 + "5.1", "3.8", 0.05)
line("S1.5", "avalanche amplification A = 2/(Z1 delta^2), Z1 = 1", A, ".2f", P2 + "2", "55", 0.5)

print()
print("S2. THE KINETIC WINDOW")
print("-" * 112)
line("S2.1", "memory time, ceiling at l_c = 3 kpc (yr)", tc_ceil, ".1f", P2 + "4.1", "9,800", 50.0)
line("S2.2", "set-point strain, ceiling 3 H_dS tau_c(3 kpc)", TH * tc_ceil, ".3e", P2 + "4.1", "1.7e-6", 0.05e-6)
line("S2.3", "equilibration time, slow edge tau_c(3 kpc)/delta (yr)", teq_slow, ".4g", P2 + "4.1 and Table 2", "5.2e4",
     0.05e4, note="the paper's 5.2e4 yr is the rounded 9,800 yr over 0.19; at full precision the edge is 5.15e4 yr")
line("S2.4", "memory time, floor delta tau_fast (yr)", tc_floor, ".1f", P2 + "4.1", "190", 0.5)
line("S2.5", "catchment length, floor c delta tau_fast (pc)", C_PC_YR * tc_floor, ".2f", P2 + "4.1", "58", 0.5)
line("S2.6", "set-point strain, floor 3 H_dS delta tau_fast", TH * tc_floor, ".3e", P2 + "4.1", "3.3e-8", 0.05e-8)
line("S2.7", "memory time, tightened ceiling at 200 pc (yr)", tc_tight, ".1f", P1 + "3.4; " + P2 + "4.1", "650", 5.0)
line("S2.8", "set-point strain, tightened ceiling", TH * tc_tight, ".3e", P2 + "4.1", "1.1e-7", 0.05e-7)
line("S2.9", "equilibration time, tightened slow edge (yr)", teq_tight, ".4g", P2 + "4.1", "3.4e3", 0.05e3)
line("S2.10", "renewal events per domain in one memory time, top of the tightened window", n_tight, ".3f",
     P1 + "3.4; " + P2 + "4.1 (1 to 3.4)", "3.4", 0.05)
print("[S2.11] interval between renewal events of a domain, 1/Gamma = b/(3 H_dS), in the tightened window: %.0f to %.0f yr"
      " || %s4.1: every 10^2 to 10^3 yr" % (tc_floor, tc_tight, P2))

print()
print("S3. THE RELAXATION TIME AND THE RESIDENCE SCREEN")
print("-" * 112)
t1, t01, t2, t3 = tauhat(1.0), tauhat(0.1), tauhat(2.0), tauhat(3.0)
line("S3.1", "tau_rel/tau_eq at the crossover X = 1", t1, ".4f", P2 + "4.1", "0.77", 0.005)
line("S3.2", "tau_rel/tau_eq at X = 0.1", t01, ".4f", P2 + "4.2, for X >= 0.1", "6", bound=True)
line("S3.3", "tau_rel at X = 1 at the slow edge (yr)", t1 * teq_slow, ".4g", P2 + "4.1", "4.0e4", 0.05e4)
line("S3.4", "tau_rel at X = 0.1 at the slow edge (yr)", t01 * teq_slow, ".4g", P2 + "4.1, 6.5 and Table 4", "3.0e5",
     0.05e5, note="the paper's 3.0e5 yr follows from the rounded slow edge of S2.3; at full precision it is 2.95e5 yr")
line("S3.5", "tau_rel at X = 1 at the tightened slow edge (yr)", t1 * teq_tight, ".4g", P2 + "4.1", "2.6e3", 0.05e3)
line("S3.6", "tau_rel at X = 0.1 at the tightened slow edge (yr)", t01 * teq_tight, ".4g", P2 + "4.1", "2.0e4", 0.05e4)
line("S3.7", "tau_rel at X = 1 at the fast edge (yr)", t1 * TAU_FAST, ".1f", P2 + "4.1 (near)", "770", 5.0)
print("[S3.8] galactic dynamical times of 1e8 and 1e9 yr over tau_rel at X = 0.1 at the slow edge: %.0f and %.0f"
      " || %s6.5: three orders of magnitude" % (1.0e8 / (t01 * teq_slow), 1.0e9 / (t01 * teq_slow), P2))
line("S3.9", "tau_rel/tau_eq at X = 2 (the Galaxy's field near the Sun)", t2, ".4f", "")
line("S3.10", "tau_rel at X = 2 at the fast edge (yr)", t2 * TAU_FAST, ".1f", P2 + "4.2", "470", 5.0)
wb = 1.0e4 * AU / (V_WB * 1.0e3) / YR
line("S3.11", "residence of a wide binary 1e4 AU across at 369.8 km/s (yr)", wb, ".1f", P2 + "4.2", "130", 5.0)
line("S3.12", "the binary's margin, tau_rel(X = 2) over its residence", t2 * TAU_FAST / wb, ".3f", P2 + "4.2", "3.7",
     0.05)
line("S3.13", "the binary's pattern at most 1 - exp(-1/margin)", 1.0 - math.exp(-wb / (t2 * TAU_FAST)), ".4f",
     P2 + "4.2 (24 percent)", "0.24", 0.005)
r_shell = math.sqrt(GM_SUN / A0_FIT)
t_shell = 2.0 * r_shell / (V_SUN * 1.0e3) / YR
line("S3.14", "the Sun's crossover shell, radius sqrt(G M_sun / a0) at a0 = 0.973e-10 m/s^2 (AU)", r_shell / AU, ".0f",
     P2 + "4.2 (about)", "7,800", 50.0, note="that is %.4f pc (%s3.7: near 0.038 pc)" % (r_shell / PC, P1))
line("S3.15", "the shell's residence 2 r / (370 km/s) (yr)", t_shell, ".1f", P2 + "4.2 (roughly)", "200", 5.0)
lo_share, hi_share = 1.0 - math.exp(-t_shell / (t1 * TAU_FAST)), 1.0 - math.exp(-t_shell / (t3 * TAU_FAST))
print("[S3.16] the shell's pattern at most 1 - exp(-t_res/tau_rel) at the fast edge, from X = 1 (the Sun's own field at the"
      " shell) to X = 3 (with the Galaxy's X of about 2): %.4f to %.4f || %s4.2: 23 to 42 percent"
      % (lo_share, hi_share, P2))
r_lo = math.sqrt(GM_SUN / A0_LAW)
t_lo = 2.0 * r_lo / (V_SUN * 1.0e3) / YR
print("       at the scale 0.9426e-10 m/s^2 of the fiducial free volume: radius %.0f AU, residence %.1f yr, %.4f to %.4f"
      % (r_lo / AU, t_lo, 1.0 - math.exp(-t_lo / (t1 * TAU_FAST)), 1.0 - math.exp(-t_lo / (t3 * TAU_FAST))))

print()
print("S4. THE NESTING OF THE RESIDENCE SCREEN INSIDE THE SCREEN BELOW THE CATCHMENT LENGTH")
print("-" * 112)
cd = C / 1.0e3 * DELTA                                     # km/s
line("S4.1", "nesting bound 6 v/(c delta) at 370 km/s", 6.0 * V_SUN / cd, ".4f", P2 + "4.2 (3.9 percent)", "0.039",
     0.0005)
line("S4.2", "nesting bound 6 v/(c delta) at 600 km/s", 6.0 * V_FAST / cd, ".4f", P2 + "4.2 (6.3 percent)", "0.063",
     0.0005)
line("S4.3", "the bound at 370 km/s in the tightened window, at 200 pc (pc)", 6.0 * V_SUN / cd * L_TIGHT, ".2f",
     P2 + "4.2", "7.8", 0.05)
line("S4.4", "the bound at 600 km/s in the tightened window, at 200 pc (pc)", 6.0 * V_FAST / cd * L_TIGHT, ".2f",
     P2 + "4.2", "12.6", 0.05)
line("S4.5", "speed below which the nesting holds at drives above 0.1, c delta/6 (km/s)", cd / 6.0, ".0f",
     P2 + "4.2", "9,500", 50.0)
line("S4.6", "drive down to which the nesting holds at 600 km/s", drive_where_tauhat(cd / V_FAST), ".5f",
     P2 + "4.2", "0.006", 0.0005)
line("S4.7", "radius below which the residence 2R/v at 600 km/s is under the fast edge, v tau_fast / 2 (pc)",
     V_FAST * 1.0e3 * TAU_FAST * YR / PC / 2.0, ".3f", "Paper II, caption of Figure 2 (about)", "0.3", 0.05)

print()
print("S5. THE RENEWAL DOMAINS")
print("-" * 112)
k = (2.0 * A * A) ** (1.0 / 3.0)
line("S5.1", "avalanche cutoff 2 A^2 at the amplification of S1.5", 2.0 * A * A, ".1f", "",
     note="Appendix C's 6,050 to 6,500 is 2 A^2 at A = 55 and at the rod's A = 57: %.0f and %.0f"
     % (2 * 55.0 ** 2, 2 * 57.0 ** 2))
line("S5.2", "l_c / a = (2 A^2)^(1/3)", k, ".3f", "")
line("S5.3", "largest domain width over the catchment length, (2 A^2)^(-1/3)", 1.0 / k, ".4f", "Paper II, Appendix C",
     "0.055", 0.0005)
line("S5.4", "domain width at the catchment floor (pc)", C_PC_YR * tc_floor / k, ".2f", "Paper II, Appendix C", "3.2",
     0.05)
line("S5.5", "domain width at 200 pc (pc)", L_TIGHT / k, ".2f", "Paper II, Appendix C", "11", 0.5)

print()
print("S6. THE STABILITY MARGIN OF THE RENEWAL'S NOISE")
print("-" * 112)
line("S6.1", "margin n Z at n = 1 (Z = 2/delta^2), misfit energies per domain", A, ".2f", P2 + "5.3", "55", 0.5)
line("S6.2", "margin n Z at the tightened window's top", A * n_tight, ".2f", P2 + "5.3", "190", 5.0)
line("S6.3", "factor by which it passes the ordering point of 9.7494, n = 1", A / T_STAR, ".3f",
     P2 + "5.3 and Table 1", "5.7", 0.05)
line("S6.4", "the same at the tightened window's top", A * n_tight / T_STAR, ".3f", P2 + "5.3", "19.5", 0.05)
line("S6.5", "factor at n = 1 against the top of the band of the ordering point, 15.50", A / T_TOP, ".3f",
     P2 + "5.3 and Table 1", "3.6", 0.05)

print()
print("S7. THE TILT TOLERANCE AND THE BOOSTS OF 28 AND 90")
print("-" * 112)
line("S7.1", "tilt that trims a boost of 28 by 10 percent (degrees)", tilt_deg(28.0), ".3f",
     P2 + "7.3 (0.2 to 0.7)", "0.7", 0.05)
line("S7.2", "tilt that trims a boost of 90 by 10 percent (degrees)", tilt_deg(90.0), ".3f",
     P2 + "7.3 (0.2 to 0.7)", "0.2", 0.05)
line("S7.3", "radius of a boost of 28 about an isolated galaxy of 10^10.57 solar masses, a0 = 0.9426e-10 (kpc)",
     radius_of_boost(28.0, LOG_M_GAL, A0_LAW), ".1f", P2 + "3.1 and 7.3 (about)", "200", 50.0)
line("S7.4", "radius of a boost of 90 about the same galaxy (kpc)", radius_of_boost(90.0, LOG_M_GAL, A0_LAW), ".1f",
     P2 + "7.3 (about)", "600", 50.0,
     note="the paper's 600 kpc holds at one significant figure; at two the radius is 640 kpc")
