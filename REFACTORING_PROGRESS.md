# Matlab Code Refactoring Progress

This document tracks the refactoring progress of Matlab (.m, .mlx) files into Python/PyTorch.

## EPG-X Functionality (`epgx/` directory)

The functionality of the following Matlab files in the `epgx/` directory is being refactored into the new `epgx_simulator.py` module, specifically within the `EPGXSimulator` class.

| Matlab File             | Status                                     | Notes                                                                 |
|-------------------------|--------------------------------------------|-----------------------------------------------------------------------|
| `epgx/epg_X_CMPG.m`     | Deleted                                    | Python implementation for two-compartment CPMG with exchange. File deleted. |
| `epgx/epg_X_grad.m`     | Deleted                                    | Python implementation for gradient operator in two-compartment model. File deleted. |
| `epgx/epg_X_m0.m`       | Deleted                                    | Python implementation for initial state in two-compartment model. File deleted. |
| `epgx/epg_X_relax.m`    | Deleted                                    | Python implementation for relaxation and exchange in two-compartment model. File deleted. |
| `epgx/epg_X_rf.m`       | Deleted                                    | Python implementation for RF pulse operator in two-compartment model. File deleted. |
| `epgx/epg_X_rfspoil.m`  | Deleted                                    | Python implementation for RF spoiled sequence with two compartments. File deleted. |
| `epgx/Test_MET.m`       | Deleted                                    | Python test created for MET simulation. File deleted.                 |
| `epgx/test_saturation_MT.m` | Deleted                                | Python test created for MT saturation simulation. File deleted.      |

## Root Directory EPG Functions (`epg_*.m`)

Status of `epg_*.m` files found in the repository root. Many of these are expected to be covered by the existing `epg_simulator.py` (single compartment) or the new `epgx_simulator.py` (two compartments).

| Matlab File             | Status                                     | Notes                                                                 |
|-------------------------|--------------------------------------------|-----------------------------------------------------------------------|
| `epg_FZ2spins.m`        | Deleted                                    | Functionality matches EPGSimulator.epg_FZ2spins(). Mark for deletion. File deleted. |
| `epg_animexample1.m`    | Visualization/Animation Script             | Matlab example script for animating EPG states during a CPMG-like sequence. Not for direct porting to a class. Python examples using a library like Matplotlib would be a separate effort. |
| `epg_animexample2.m`    | Visualization/Animation Script             | Matlab example script for animating EPG states. Not for direct porting. |
| `epg_animexample3.m`    | Visualization/Animation Script             | Matlab example script for animating EPG states. Not for direct porting. |
| `epg_animgrad.m`        | Visualization/Animation Script             | Likely calls epg_grad and then updates a visualization. Not for direct porting. |
| `epg_animrf.m`          | Visualization/Animation Script             | Likely calls epg_rf and then updates a visualization. Not for direct porting. |
| `epg_cpmg.m`            | Deleted                                    | Core CPMG simulation matches EPGSimulator.epg_cpmg(). Matlab version includes a Hennig flip angle trick not in Python version. Plotting is not part of core function. Consider for deletion. File deleted. |
| `epg_cpmg_all.m`        | Example Script                             | Matlab script that calls epg_cpmg in a loop for demo/plotting. Not a core function for direct porting into a class. Could be adapted as a Python example if needed. |
| `epg_echotrain.m`       | Example Script                             | Matlab script that calls epg_cpmg with various parameters for demo/plotting. Not a core function. Could be adapted as a Python example. |
| `epg_grad.m`            | Deleted                                    | Functionality matches EPGSimulator.epg_grad(). Mark for deletion. File deleted. |
| `epg_gradecho.m`        | Deleted                                    | Core gradient echo simulation with spoiling options matches EPGSimulator.epg_gradecho(). Plotting is not part of core function. Consider for deletion. File deleted. |
| `epg_gradspoil.m`       | Deleted                                    | Simulates a basic gradient-spoiled sequence. Functionality achievable with EPGSimulator.epg_gradecho() using appropriate parameters (e.g., TE near zero, gspoil_flag=1 or 100). Consider for deletion. File deleted. |
| `epg_grelax.m`          | Deleted                                    | Functionality, including diffusion modeling, is now incorporated into EPGSimulator.epg_grelax(). Mark for deletion. File deleted. |
| `epg_m0.m`              | Deleted                                    | Functionality matches EPGSimulator.epg_m0(). Mark for deletion. File deleted. |
| `epg_mgrad.m`           | Deleted                                    | Functionality matches EPGSimulator.epg_mgrad() (which uses epg_grad with positive_lobe=False). Mark for deletion. File deleted. |
| `epg_relax.m`           | Deleted                                    | Functionality matches EPGSimulator.epg_relax(). Mark for deletion. File deleted. |
| `epg_rf.m`              | Deleted                                    | Functionality matches EPGSimulator.epg_rf(). Mark for deletion. File deleted. |
| `epg_rfspoil.m`         | Deleted                                    | Quadratic RF phase cycling functionality is now implemented in EPGSimulator.epg_rfspoil_quadratic_phase(). Mark for deletion. File deleted. |
| `epg_saveframe.m`       | Visualization/Animation Script             | Helper function for saving frames during animations. Matlab-specific visualization utility. |
| `epg_show.m`            | Visualization/Animation Script             | Core Matlab script for displaying EPG states. Python equivalents would use libraries like Matplotlib. |
| `epg_show_grad.m`       | Visualization/Animation Script             | Specialized EPG state visualization (likely after gradient). Python equivalents would use Matplotlib. |
| `epg_show_grelax.m`     | Visualization/Animation Script             | Specialized EPG state visualization. Python equivalents would use Matplotlib. |
| `epg_show_relax.m`      | Visualization/Animation Script             | Specialized EPG state visualization. Python equivalents would use Matplotlib. |
| `epg_show_rf.m`         | Visualization/Animation Script             | Specialized EPG state visualization. Python equivalents would use Matplotlib. |
| `epg_showorder.m`       | Visualization/Animation Script             | Specialized EPG state visualization. Python equivalents would use Matplotlib. |
| `epg_showstate.m`       | Visualization/Animation Script             | Core Matlab function for plotting individual EPG spin states. Python equivalents would use Matplotlib. |
| `epg_spins2FZ.m`        | Deleted                                    | Functionality matches EPGSimulator.epg_spins2FZ(). Mark for deletion. File deleted. |
| `epg_stim.m`            | Deleted                                    | Functionality (as epg_stim_calc) matches EPGSimulator.epg_stim_calc(). Mark for deletion. File deleted. |
| `epg_trim.m`            | Deleted                                    | Functionality effectively covered and improved by EPGSimulator.epg_trim(). Mark for deletion. File deleted. |
| `epg_zrot.m`            | Deleted                                    | Functionality matches EPGSimulator.epg_zrot(). Mark for deletion. File deleted. |

## Other Matlab Files

This section will list other `.m` and `.mlx` files found in the repository and their refactoring status.

| Matlab File             | Status                                     | Notes                                                                 |
|-------------------------|--------------------------------------------|-----------------------------------------------------------------------|
| ...                     | ...                                        | ...                                                                   |

## General Matlab Utilities, Scripts, and Lecture Materials

This section covers miscellaneous Matlab files that are not part of the EPG groups.

| Matlab File                     | Status                                     | Notes                                                                 |
|---------------------------------|--------------------------------------------|-----------------------------------------------------------------------|
| `ft.m`                          | Utility function                           | Centered 2D FFT. Replaceable with `torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(data)))`. Consider for deletion or inclusion in a Python util module if widely used. |
| `ift.m`                         | Utility function                           | Centered 2D IFFT. Replaceable with `torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(data)))`. Consider for deletion or inclusion in a Python util module. |
| `sinc.m`                        | Covered by PyTorch                         | Normalized sinc function. Use `torch.sinc()`. Mark for deletion.      |
| `gaussian.m`                    | Utility function                           | Standard 1D Gaussian PDF. Can be custom Python function if needed.    |
| `Rad229_MRI_Phantom.m`          | Likely covered by Python/Rad229_MRI_Phantom.py | Matlab version seems incomplete. Python version likely supersedes it. Needs verification, then mark for deletion. |
| `epgx/Test_SPGR.m`              | Pending Analysis                           | (Moved from EPGX section for clarity) Test script for SPGR with EPGX. Needs review for porting to a Python test for `EPGXSimulator`. |
| `epgx/Test_fitting_MET.m`       | Pending Analysis                           | (Moved from EPGX section for clarity) MET fitting script. Assess if it's a test case for `EPGXSimulator` or an application example. |
