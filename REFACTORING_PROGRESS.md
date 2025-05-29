# Matlab Code Refactoring Progress

This document tracks the refactoring progress of Matlab (.m, .mlx) files into Python/PyTorch.

## EPG-X Functionality (`epgx/` directory)

The functionality of the following Matlab files in the `epgx/` directory is being refactored into the new `epgx_simulator.py` module, specifically within the `EPGXSimulator` class.

| Matlab File             | Status                                     | Notes                                                                 |
|-------------------------|--------------------------------------------|-----------------------------------------------------------------------|
| `epgx/epg_X_CMPG.m`     | Addressed by `EPGXSimulator.epgx_cpmg`     | Python implementation for two-compartment CPMG with exchange.         |
| `epgx/epg_X_grad.m`     | Addressed by `EPGXSimulator.epgx_grad`     | Python implementation for gradient operator in two-compartment model. |
| `epgx/epg_X_m0.m`       | Addressed by `EPGXSimulator.epgx_m0`       | Python implementation for initial state in two-compartment model.   |
| `epgx/epg_X_relax.m`    | Addressed by `EPGXSimulator.epgx_relax`    | Python implementation for relaxation and exchange in two-compartment model. |
| `epgx/epg_X_rf.m`       | Addressed by `EPGXSimulator.epgx_rf`       | Python implementation for RF pulse operator in two-compartment model.   |
| `epgx/epg_X_rfspoil.m`  | Addressed by `EPGXSimulator.epgx_rfspoil`  | Python implementation for RF spoiled sequence with two compartments.  |
| `epgx/Test_MET.m`       | Addressed by `test_epgx_simulator.test_met_simulation` | Python test created for MET simulation.                             |
| `epgx/Test_SPGR.m`      | Pending Analysis                           | To be reviewed for porting to a Python test.                          |
| `epgx/Test_fitting_MET.m`| Pending Analysis                           | Likely an example/application script; assess for testable components. |
| `epgx/test_saturation_MT.m` | Addressed by `test_epgx_simulator.test_mt_saturation_simulation` | Python test created for MT saturation simulation.                  |

## Root Directory EPG Functions (`epg_*.m`)

Status of `epg_*.m` files found in the repository root. Many of these are expected to be covered by the existing `epg_simulator.py` (single compartment) or the new `epgx_simulator.py` (two compartments).

| Matlab File             | Status                                     | Notes                                                                 |
|-------------------------|--------------------------------------------|-----------------------------------------------------------------------|
| `epg_FZ2spins.m`        | Covered by epg_simulator.py                | Functionality matches EPGSimulator.epg_FZ2spins(). Mark for deletion. |
| `epg_animexample1.m`    | Pending Analysis                           | Likely an example script. Assess for porting to Python example.       |
| `epg_animexample2.m`    | Pending Analysis                           | Likely an example script. Assess for porting to Python example.       |
| `epg_animexample3.m`    | Pending Analysis                           | Likely an example script. Assess for porting to Python example.       |
| `epg_animgrad.m`        | Pending Analysis                           | Animation function. Assess for porting.                               |
| `epg_animrf.m`          | Pending Analysis                           | Animation function. Assess for porting.                               |
| `epg_cpmg.m`            | Largely Covered by epg_simulator.py        | Core CPMG simulation matches EPGSimulator.epg_cpmg(). Matlab version includes a Hennig flip angle trick not in Python version. Plotting is not part of core function. Consider for deletion. |
| `epg_cpmg_all.m`        | Pending Analysis                           | Review functionality.                                                 |
| `epg_echotrain.m`       | Pending Analysis                           | Review functionality.                                                 |
| `epg_grad.m`            | Covered by epg_simulator.py                | Functionality matches EPGSimulator.epg_grad(). Mark for deletion.     |
| `epg_gradecho.m`        | Largely Covered by epg_simulator.py        | Core gradient echo simulation with spoiling options matches EPGSimulator.epg_gradecho(). Plotting is not part of core function. Consider for deletion. |
| `epg_gradspoil.m`       | Pending Analysis                           | Review functionality, likely related to `epg_grad`.                   |
| `epg_grelax.m`          | Partially Covered by epg_simulator.py      | Basic relaxation and gradient application matches EPGSimulator.epg_grelax(). However, the diffusion modeling component (with D and kg parameters) in epg_grelax.m is NOT implemented in the Python version. This specific diffusion logic would need separate porting if required. |
| `epg_m0.m`              | Covered by epg_simulator.py                | Functionality matches EPGSimulator.epg_m0(). Mark for deletion.       |
| `epg_mgrad.m`           | Covered by epg_simulator.py                | Functionality matches EPGSimulator.epg_mgrad() (which uses epg_grad with positive_lobe=False). Mark for deletion. |
| `epg_relax.m`           | Covered by epg_simulator.py                | Functionality matches EPGSimulator.epg_relax(). Mark for deletion.    |
| `epg_rf.m`              | Covered by epg_simulator.py                | Functionality matches EPGSimulator.epg_rf(). Mark for deletion.       |
| `epg_rfspoil.m`         | Pending Analysis                           | Review functionality.                                                 |
| `epg_saveframe.m`       | Pending Analysis                           | Helper for animations.                                                |
| `epg_show.m`            | Pending Analysis                           | Visualization. Assess for porting to Python plotting.                 |
| `epg_show_grad.m`       | Pending Analysis                           | Visualization. Assess for porting.                                    |
| `epg_show_grelax.m`     | Pending Analysis                           | Visualization. Assess for porting.                                    |
| `epg_show_relax.m`      | Pending Analysis                           | Visualization. Assess for porting.                                    |
| `epg_show_rf.m`         | Pending Analysis                           | Visualization. Assess for porting.                                    |
| `epg_showorder.m`       | Pending Analysis                           | Visualization. Assess for porting.                                    |
| `epg_showstate.m`       | Pending Analysis                           | Visualization. Assess for porting.                                    |
| `epg_spins2FZ.m`        | Covered by epg_simulator.py                | Functionality matches EPGSimulator.epg_spins2FZ(). Mark for deletion. |
| `epg_stim.m`            | Covered by epg_simulator.py                | Functionality (as epg_stim_calc) matches EPGSimulator.epg_stim_calc(). Mark for deletion. |
| `epg_trim.m`            | Covered by epg_simulator.py                | Functionality effectively covered and improved by EPGSimulator.epg_trim(). Mark for deletion. |
| `epg_zrot.m`            | Covered by epg_simulator.py                | Functionality matches EPGSimulator.epg_zrot(). Mark for deletion.     |

## Other Matlab Files

This section will list other `.m` and `.mlx` files found in the repository and their refactoring status.

| Matlab File             | Status                                     | Notes                                                                 |
|-------------------------|--------------------------------------------|-----------------------------------------------------------------------|
| ...                     | ...                                        | ...                                                                   |
