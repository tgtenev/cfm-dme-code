"""The cosmological background of the computations in this folder: the Planck 2018 TT,TE,EE+lowE+lensing parameters
(Planck Collaboration 2020, A&A 641, A6, Table 2), flat, with one massive neutrino of 0.06 eV.

The matter power of web_lensing.py is a CAMB run at these parameters, and the distances, the lensing kernels, the
groups' measurement geometry and the cosmic baryon fraction use the same background:
    h = 0.6736, Omega_b = omega_b / h^2 = 0.0493, Omega_m = (omega_b + omega_c + omega_nu) / h^2 = 0.3152,
with omega_nu = m_nu / 93.14 eV.
"""
P18 = dict(H0=67.36, ombh2=0.02237, omch2=0.1200, mnu=0.06, tau=0.0544, ln10As=3.044, ns=0.9649)

H = P18['H0'] / 100.0
OMEGA_B = P18['ombh2'] / H ** 2
OMEGA_M = (P18['ombh2'] + P18['omch2'] + P18['mnu'] / 93.14) / H ** 2
