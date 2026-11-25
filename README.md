# Code and data for the dark-matter effect in the Cosmic Fabric Model, Papers I and II

This repository holds the code and the outputs behind two papers by T. G. Tenev:

- Paper I: *The Dark-Matter Effect in the Cosmic Fabric Model. I. A Derived Law for Galaxies with One Fitted Constant*;
- Paper II: *The Dark-Matter Effect in the Cosmic Fabric Model. II. The Field Theory and the Kinetics of the Law for Galaxies*.

Each release is archived on Zenodo with its own DOI.
Release 1.0.0 holds the code and outputs that Paper I cites.
Release 1.0.1 adds the test of the stellar normalization of Paper II, Sect. 6.1, the option `--ml-centres` of three scripts with its outputs, and takes the empirical function's upper divisor as 5.60 in the catalogue of unit shapes; every other output is that of 1.0.0.

## Layout

| Folder | What it computes |
|---|---|
| `rotation-curves/` | The fits of the law to the SPARC rotation curves, the exact AQUAL solutions in the disks and their refits, and the tests on disks |
| `mond-classes/` | A MOND-type law against the globular clusters, the groups, the cluster cores and the cosmic web, with the stacked lensing profiles of the groups, on the Planck 2018 background of `background.py` |
| `kinetics/` | The kinetic windows, the residence screen, the stability margins and the stored axis with its trail behind a moving body |
| `field/` | Checks of the field theory: the layers' uneven excess, conduction on the scale of the layers, the two potentials, the bending, the ordering threshold and the Eshelby partition |
| `globular-clusters/` | The globular-cluster tests at the adopted floor of the catchment length, 58 pc |
| `avalanche/` | The mean-field model of the avalanche kinetics, the lattice simulation and the unit shapes |
| `data/` | The SPARC data, and the sources and checksums of the datasets that are not redistributed (`data/README.md`) |

Every script is run from the root of this repository.
Its opening lines state what it computes, which results of the papers it supports, its inputs and outputs, its command and its run time.
Its outputs are committed beside it, and a run writes them again.
The `.py` scripts run under Python 3 with the packages of `requirements.txt`, and the `.pl` scripts under Perl 5.20 or later with its core modules.
The commands are written with `python`; on Windows, `py -3` serves.

## Paper I

| Result | Paper I | Code | Outputs |
|---|---|---|---|
| The constrained fit of the 153 SPARC galaxies with one constant, in sixteen treatments, beside MOND's simple and standard functions and the empirical function at equal freedom; the full field solution; the scales of the comparison functions | §§4.1, 4.2, Table 2, Eq. (16) | `rotation-curves/constrained_fit.py`, with `constrained_core.py` and `constrained_solve.py` | `constrained_fit-results.txt` |
| The fits at each value of the counting factor | §4.2, Table 2 | `rotation-curves/counting_factor_menu.py` | `counting_factor_menu-results.txt` |
| The two-constant fit, a test of the relation between the free volume and the scale; the median shifts of distance and inclination; the bootstrap of the product | §§4.1, 4.2, Figures 5 and 6 | `rotation-curves/two_constant_fit.pl` | `two_constant_fit-results.txt`, `two_constant_fit-frozen-results.txt` |
| The full solution with catalog distances | §4.2 | `rotation-curves/catalog_distances_nodes.py`, `catalog_distances.py` | `catalog_distances-results.txt` |
| A second implementation of the fit machinery, which reproduces the Perl fit | §4.1 | `rotation-curves/aqual-refit-fit.py --part machinery` | `aqual-refit-machinery-results.txt` |
| The data points inside 1 kpc and 0.5 kpc | §4.1 | `rotation-curves/inner_points.py` | `inner_points-results.txt` |
| The data of Figure 4 | Figure 4 | `rotation-curves/rar_figure_data.py` | `rar-law.tsv`, `rar-simple.tsv`, `rar_figure_data-results.txt` |
| The data of Figures 5 and 6 | Figures 5 and 6 | `rotation-curves/constrained_fit.py` and `two_constant_fit.pl` | `constrained_fit-valley.csv`; the node table of `two_constant_fit-results.txt` |
| The layers' mean excess and the share of layers without excess | §3.4 | `field/layer_variation.py` | `layer_variation-results.txt` |
| The cost of conduction on the scale of the layers | §3.5 | `field/conduction_checks.py` | `conduction_checks-results.txt` |
| Lensing against dynamics | §5.2 | `field/two_potentials_checks.py` | `two_potentials_checks-results.txt` |
| The kinetic values that Paper I quotes, the fiducial divisor, slope and onset among them | §§3.4, 3.5 | `kinetics/kinetic_windows.py` | `kinetic_windows-results.txt` |
| Table 1's MOND column for the globular clusters | Table 1, §3.7 | `globular-clusters/gc_dynamical.py`, then `mond-classes/mond_gc.py` | `mond_gc-results.txt` |
| Table 1's MOND column for the groups, the cluster cores and the cosmic web | Table 1, §2.4 | `mond-classes/groups_stack.py`, then `mond-classes/mond_classes.py` | `groups-stack-profiles.json`, `groups_stack-results.txt`, `mond_classes-results.txt` |

## Paper II

| Result | Paper II | Code | Outputs |
|---|---|---|---|
| The constrained fit and its full field solution | §§2, 5.2, 6.1 | `rotation-curves/constrained_fit.py` | `constrained_fit-results.txt` |
| The exact AQUAL solutions in the disks, their check on a Kuzmin disk, and QUMOND's agreement | §6.1 | `rotation-curves/aqual-disk-densities.py`, `aqual-disk-solver.py` | `aqual-disk-solver-results.txt`, `aqual-disk-solver-validate-results.txt` |
| The full solution against the algebraic relation, with catalog distances and wider priors, the slope tests, and the free volume of the two-constant fit | §6.1 | `rotation-curves/aqual-refit-solve.py`, `aqual-refit-fit.py` | `aqual-refit-main-*-results.txt`, `aqual-refit-kuzmin-results.txt` |
| The same solutions coarse-grained at the catchment length | §6.1 | `rotation-curves/coarse_grained_solve.py`, `coarse_grained_merge.py`, `coarse_grained_fit.py` | `coarse_grained_fit-*-results.txt` |
| The full solution against the relation averaged over the gas layer: the slope test and the distance from noiseless rotation curves of the full solution; the share of AQUAL's own departure that the data show | §§6.1, 7.3 | `rotation-curves/gas_layer_solve.py`, `gas_layer_baseline.py` | `gas_layer_baseline-model-results.txt`, `gas_layer_baseline-sparc-results.txt` |
| The disk results with the stellar mass-to-light priors centred 10 and 20 percent higher: the constrained fit against MOND's functions, the full solution against the algebraic relation, the slope tests and the test against the gas layer | §6.1 | `rotation-curves/constrained_fit.py`, `aqual-refit-fit.py` and `gas_layer_baseline.py`, each with `--ml-centres` | `constrained_fit-ml0.55-results.txt` and `-ml0.60-`, `aqual-refit-main-T11-ml0.55-results.txt` and `-ml0.60-`, `gas_layer_baseline-model-ml0.55-results.txt` and the three others |
| The direction of a new layer, tested on the inner rotation curves of the 32 galaxies that the law boosts by more than 5 | §3.1, Table 1 | `rotation-curves/fetch_*.py`, `layer_direction_test.py` | `layer_direction_test-results.txt` |
| The quality split of disturbed galaxies | §6.5 | `rotation-curves/quality_split.pl` | `quality_split-results.txt` |
| The kinetic windows, the relaxation time, the residence screen, the stability margins, the domains, the tilt of the layers and where the boosts of 28 and 90 fall | §§2, 3.1, 4.1, 4.2, 5.3, 7.3, Appendix C, Tables 1 and 4 | `kinetics/kinetic_windows.py` | `kinetic_windows-results.txt` |
| The stored axis: its lifetime, its buildup and its lag behind a moving body | §§3.1, 4.1 | `kinetics/stored_axis_kinetics.py` | `stored_axis_kinetics-results.txt` |
| The trail behind a moving body under the flux rule (rule C in the scripts) and the two other rules, its linear response, its full relaxation, and the tilt by a uniform axis | §3.1, Appendix A.3 | `kinetics/steady_dipole.py`, `linear_response.py`, `moving_body.py`, `uniform_axis_point_mass.py` | the four `-results.txt` files |
| The variation that the layers' uneven excess gives the field | §3.1, Appendix A.2, Appendix C, Table 1 | `field/layer_variation.py` | `layer_variation-results.txt` |
| The loss of boost under conduction on the scale of the layers, and the effective medium | Appendix A.2 | `field/conduction_checks.py` | `conduction_checks-results.txt` |
| The equality of the two potentials, and the bending | §§3.3, 7.3, Appendix A.4 | `field/two_potentials_checks.py`, `bending_checks.py` | the two `-results.txt` files |
| The ordering threshold of the stability check | §5.3, Table 1 | `field/stability_ordering.py`, `stability_threshold_mc.py`, `stability_threshold_calc.py` | `stability_ordering-results.txt`, `stability_threshold_mc-sc0.txt`, `-fcc0.txt`, `stability_threshold_calc-results.txt` |
| The partition of the Eshelby energy | §5.1, Eq. (11) | `field/eshelby_partition.py` | `eshelby_partition-results.txt` |
| The globular-cluster tests at the adopted floor of 58 pc | §6.3, Figure 3, Tables 2 and 4 | `globular-clusters/gc_dynamical.py`, `gc_radial_trends.py` | `gc_dynamical-results.txt`, `gc_radial_trends-results.txt` |
| A MOND-type law on the globular clusters | §6.6 | `mond-classes/mond_gc.py` | `mond_gc-results.txt` |
| The mean-field model of the avalanche kinetics | §5.1, Appendix C | `avalanche/mean_field_toy.pl` | `mean-field-scan.tsv`, `mean-field-scan-signed-modulus.tsv` |
| The lattice simulation of the avalanche kinetics | Appendix C, Table 1 | `avalanche/lattice/lattice_sim.pl`, `lattice_analyze.py`, `rebuild_crossings.py` | the runs `run-*.tsv`, `lattice_analyze-results.txt`, `crossings.tsv`, `rebuild_crossings-results.txt` |
| The rod of four sites and the catalog of unit shapes, with the checks of the trigger scale, the activation and the laying rule | §§5.1, 5.2, Appendix C, Table 1 | `avalanche/unit-shapes/latticebricks.py`, `avalanche/lattice/lattice_sim.pl`, `catalogue/` (`polycubes.py`, `build_results.py`, `rebuild/rebuild_catalogue.py`), `pair-holes/` (`gen_holes.py`, `lattice_sim_pair.pl`, `summarize.py`) | the runs in `rod/`, `checks/`, `catalogue/` and `pair-holes/`, `catalogue-results.tsv`, `rebuild_catalogue-results.txt`, `summary.txt`, `pair-holes-results.tsv` |

Paper II's Figure 3 is drawn from `globular-clusters/gc_dynamical-figdata.tsv`, which `gc_dynamical.py` writes.
That file copies values from the globular-cluster database, which is not redistributed, so it is not in the repository.
The other figures of the two papers are diagrams or curves of their equations.

## Run order

Some scripts read what others write.
The committed outputs let each script run on its own, except where an input is not redistributed or is rebuilt in seconds.

- `rotation-curves/`:
  - `aqual-disk-densities.py`, then `aqual-disk-solver.py` (its parts validate, galaxies and summary);
  - `aqual-refit-solve.py`, then `aqual-refit-fit.py`;
  - `constrained_solve.py` (gamma, upsvar), then `constrained_fit.py --parts main`, then `constrained_solve.py --part atopt`, then `constrained_fit.py` and `counting_factor_menu.py`;
  - `coarse_grained_solve.py`, then `coarse_grained_merge.py`, then `coarse_grained_fit.py`;
  - `gas_layer_solve.py` (its parts gamma, upsvar and relvar), then `gas_layer_baseline.py`;
  - for the test of the stellar normalization, `constrained_fit.py --parts main`, `aqual-refit-fit.py --part main --law T11` and `gas_layer_baseline.py` (both inputs), each with `--ml-centres 0.55,0.77` and with `--ml-centres 0.6,0.84`;
  - `catalog_distances_nodes.py`, then `catalog_distances.py`;
  - the four `fetch_*.py`, then `layer_direction_test.py`.
- `globular-clusters/gc_dynamical.py`, then `globular-clusters/gc_radial_trends.py` and `mond-classes/mond_gc.py`.
- `mond-classes/groups_stack.py`, then `mond-classes/mond_classes.py`.
- `field/stability_threshold_mc.py` (jobs sc0 and fcc0), then `field/stability_threshold_calc.py`.
- `avalanche/`: the runs, then the analyses (`lattice_analyze.py`, then `rebuild_crossings.py`; `build_results.py`, then `rebuild_catalogue.py`; `gen_holes.py` and the pair runs, then `summarize.py`).

The long runs are the solutions of the AQUAL equation in the disks (hours), the constrained and two-constant fits (under 1.5 hours each), the moving body (about 1.5 hours) and the avalanche runs (about 35 hours of run time in all).
The rest take seconds to minutes.

## Numbers derived from the outputs by hand

- Paper I §4.2, the product of the two-constant fit, 1.52 ± 0.60 and 1.64 ± 0.65 at 67.4 and 73 km/s/Mpc: `two_constant_fit-results.txt` gives 1.832 and 1.985, with bootstrap standard deviations of 0.7217 and 0.7817, for the rate cH₀; with the de Sitter rate, every value is multiplied by √Ω_Λ = 0.8275.
- Paper I §4.2, the product's span of 0.3 to 2.9: (cH_dS/a₀) ln(1 + δ) over the two-constant columns of the sixteen treatments in `constrained_fit-results.txt`, at both Hubble constants, runs from 0.318 to 2.851.
- Paper I §4.1, the free volume of 0.25 at a scale of 1.20 × 10⁻¹⁰ m s⁻²: exp(1.20/5.4185) − 1 = 0.248, with cH_dS = 5.4185 × 10⁻¹⁰ m s⁻² from `constrained_fit-results.txt`.
- Paper II §3.1, three to four standard deviations: the slope of 0.143 ± 0.197 in `layer_direction_test-results.txt` lies 4.35 formal errors and 3.7 bootstrap standard deviations from 1.
- Paper II §5.2, the scales 0.951 and 0.948 × 10⁻¹⁰ m s⁻² at the edges of the divisor's window: cH_dS divided by the divisor, 5.42/5.70 and 5.87/6.19, with cH_dS = 5.42 and 5.87 × 10⁻¹⁰ m s⁻² at 67.4 and 73 km/s/Mpc.
- Paper II §7.3, the 55 to 90 percent of AQUAL's own departure that the data do not show, and the scale's increase of 17 to 29 percent: `gas_layer_baseline-*-results.txt` give the share that the data show as 0.395 and 0.447 by the slope test and 0.106 and 0.180 by the comparison of χ², (E_R − observed)/(E_R − E_F), so that 0.553 to 0.894 is not shown; the increase is 2 × 0.0601 dex times that fraction, 0.0665 to 0.1075 dex, or 16.5 to 28.1 percent.
- Paper II §6.1, the median departure of about 3 percent: the per-bin medians of `aqual-disk-solver-results.txt` §1 are −0.011 dex, and the median over the 153 galaxies of the column `G_T11_-10.0` of `aqual-refit-gamma-0.csv` and `-1.csv` is −0.0131 dex.

## Data

`data/README.md` lists the eleven datasets the scripts read, with their sources, the dates they were fetched and their checksums.

## Licences

The code is licensed under the MIT License, in `LICENSE`.
The outputs and the documents are licensed under the Creative Commons Attribution 4.0 International License, in `LICENSE-CC-BY-4.0`.
The SPARC data in `data/sparc/` are redistributed under their authors' own CC BY 4.0 license.
