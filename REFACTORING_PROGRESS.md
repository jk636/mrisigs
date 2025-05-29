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
| `Diffusion_Coeff_H20.m`         | Data provider function                     | Returns hardcoded diffusion coefficients for H2O at various temperatures. Easy to port to a Python dictionary or data file if needed. |
| `Rad229_Conventional_FlowComp.m`| Specialized gradient design function       | Designs flow-compensated gradient waveforms (Pelc et al.). Valuable to port to a Python gradient design module if functionality is desired. |
| `Rad229_Conventional_FlowEncode.m`| Specialized gradient design function       | Designs flow-encoding gradient waveforms (Pelc et al.). Valuable to port to a Python gradient design module if functionality is desired. |
| `Rad229_Fourier_Encoding.m`     | Educational script/demo                    | Demonstrates Fourier encoding principles. Suitable for conversion to a Python example/Jupyter notebook. |
| `Rad229_MRI_Signal_Eqn.m`       | Incomplete script/demo                     | Marked as incomplete in source. Demonstrates MRI signal equation using other demo functions (possibly missing). Likely not for direct porting. |
| `bssfp.m`                       | Analytical bSSFP signal calculation        | Calculates bSSFP signal using analytical formula, distinct from EPG simulation. Could be ported as a Python utility function. |
| `relax.m`                       | Utility function for Bloch relaxation matrices | Returns A and B components for m' = A*m+B. Functionality implicitly in `EPGSimulator.epg_relax`. Low priority for direct porting unless returning A,B explicitly is needed. |
| `adiabatic.m`                   | RF pulse design function (Adiabatic Silver-Hoult) | Designs Silver-Hoult adiabatic RF pulse. Specialized; could be part of a Python RF pulse design module. Simulation part depends on external `bloch.m`. |
| `throt.m`                       | Utility function for 3D rotation matrix    | Rotation about an arbitrary axis in xy-plane. Replaceable with SciPy or custom Python utility. `mri_rotation.py` might have equivalents. |
| `xrot.m`                        | Utility function for 3D rotation matrix    | Rotation about x-axis. Replaceable with SciPy or custom Python utility. `mri_rotation.py` might have equivalents. |
| `yrot.m`                        | Utility function for 3D rotation matrix    | Rotation about y-axis. Replaceable with SciPy or custom Python utility. `mri_rotation.py` might have equivalents. |
| `zrot.m`                        | Utility function for 3D rotation matrix    | Rotation about z-axis. Replaceable with SciPy or custom Python utility. `mri_rotation.py` might have equivalents. (Note: `EPGSimulator.epg_zrot` applies phase, not this Cartesian matrix). |
| `abprop.m`                      | Utility for combining Bloch steps          | Propagates m' = A*m+B for sequential operations. Could be a Python utility. |
| `absplit.m`                     | Utility for matrix transform interpolation | Splits A,B into n sub-operations. Matrix fractional power (e.g., SciPy) is an alternative for Ai=A^(1/n). |
| `abanim.m`                      | Visualization/Animation Script             | Animates m'=A*M+B propagation using `absplit`. Matlab-specific animation. |
| `ahncholinphase.m`              | Signal processing utility (linear phase estimation) | Calculates Ahn-Cho linear phase. Port if this specific algorithm is needed. |
| `bvalue.m`                      | Utility function (b-value calculation)     | Calculates b-value from a gradient waveform. Useful for diffusion MRI. Port to Python gradient utils if needed. |
| `calc_harmonics.m`              | Specialized utility (spherical harmonics for gradient non-linearity) | Calculates field values from SH coefficients (Siemens convention). For advanced MRI processing. |
| `calc_spherical_harmonics.m`    | Specialized utility (spherical harmonics for gradient non-linearity) | Similar to calc_harmonics.m. For advanced MRI processing.             |
| `calcgradinfo.m`                | Gradient analysis utility                  | Calculates k-space, moments, slew rate, coil voltage from gradient waveform. Valuable to port to Python gradient/sequence utils. |
| `circ.m`                        | Utility function (2D circle generator)     | Creates a 2D circular mask. Easily done with NumPy. Low priority for direct porting. |
| `conventional_flowcomp.m`       | Specialized gradient design function (flow comp) | Similar to `Rad229_Conventional_FlowComp.m`. Needs comparison to determine if redundant or unique. |
| `conventional_flowencode.m`     | Specialized gradient design function (flow encode) | Similar to `Rad229_Conventional_FlowEncode.m`. Needs comparison to determine if redundant or unique. |
| `corrnoise.m`                   | Utility function (correlated noise generator) | Generates correlated Gaussian noise. Covered by `numpy.random.multivariate_normal`. Consider for deletion. |
| `cropim.m`                      | Utility function (image cropping)          | Crops image around center. Easily done with NumPy slicing. Low priority for direct porting. |
| `csens2d.m`                     | MRI utility (coil sensitivity map estimation) | Estimates coil sensitivity maps from k-space calibration data. Important for SENSE. Check Python MRI libs or port. |
| `demo.m`                        | Demonstration Script (`gropt` examples)    | Shows usage of `gropt`, `get_min_TE_diff`, `plot_waveform`. Convert to Python example/notebook if `gropt` is ported. |
| `demo_moments.m`                | Demonstration Script (gradient moment design) | Shows gradient design with moment constraints, likely using `gropt`. Convert to Python example if relevant tools are ported. |
| `demo_pns.m`                    | Demonstration Script (PNS constraints in gradient design) | Shows b-value optimization with PNS constraints. Convert to Python example if relevant tools are ported. |
| `design_symmetric_gradients.m`  | Specialized gradient design function (DWI waveforms) | Designs monopolar, bipolar, modified bipolar DWI gradients. Includes `trapTransform` helper. Valuable to port if DWI sequence design is needed. |
| `diamond.m`                     | Utility function (2D diamond shape generator) | Creates a 2D diamond mask. Easily done with NumPy. Low priority. |
| `dispangle.m`                   | Visualization utility (displays phase of complex data) | Wrapper for `dispim` to show phase. Python equivalent uses Matplotlib. |
| `dispim.m`                      | Visualization utility (displays image magnitude) | Core image display. Python equivalent uses Matplotlib. |
| `dispkspim.m`                   | Visualization utility (k-space and image display) | Displays k-space/image mag/phase in 2x2 subplot. Python equivalent uses Matplotlib & FFT functions. |
| `displogim.m`                   | Visualization utility (displays log-magnitude of image) | Displays log-magnitude using `dispim`. Python equivalent uses Matplotlib. |
| `exrecsignal.m`                 | Analytical SPGR signal calculation         | Calculates steady-state SPGR signal using Ernst angle formula. Python utility if analytical SPGR needed. |
| `gaussian2d.m`                  | Utility function (2D Gaussian generator)   | Creates a 2D Gaussian image using 1D `gaussian.m`. Python utility if needed. |
| `get_bval.m`                    | Utility function (b-value calculation with inversion) | Calculates b-value from gradient waveform, handling inversion. Important for DWI. Port to Python gradient utils. |
| `get_coords.m`                  | Utility function (coordinate transformation) | Calculates a coordinate transformation matrix. Utility depends on specific coordinate system needs. |
| `get_min_TE_diff.m`             | Optimization script (min TE for diffusion using `gropt`) | Finds minimum TE for diffusion sequence from `gropt`. Porting depends on `gropt`. |
| `get_min_TE_free.m`             | Optimization script (min TE for `gropt` free mode) | Finds minimum TE for `gropt` free mode. Porting depends on `gropt`. |
| `get_moments.m`                 | Utility function (gradient moment calculation with inversion) | Calculates first 5 gradient moments, handling inversion. Valuable for sequence design. |
| `get_stim.m`                    | PNS calculation utility                    | Calculates PNS metric from gradient waveform based on a model. Check vs `pns_constraint_op.py`. Port if not covered. |
| `ghist.m`                       | Visualization utility (histogram with Gaussian fit) | Plots histogram and Gaussian fit. Python equivalent uses Matplotlib/SciPy. |
| `gradanimation.m`               | Visualization/Animation Script             | Example script for animating magnetization vectors. Not for direct porting. |
| `gresignal.m`                   | Analytical Gradient-Spoiled GRE signal calculation | Calculates steady-state gradient-spoiled GRE signal (Buxton 1989). Python utility if analytical signal needed. |
| `grid_phantom.m`                | Phantom generator (grid pattern)           | Creates a circular phantom with a grid. Port to Python if this phantom is needed. |
| `gridmat.m`                     | MRI utility (2D Gridding for non-Cartesian recon) | Grids non-Cartesian k-space data using Kaiser-Bessel kernel. Assess vs Python NUFFT libs or port. Includes `kb` helper. |
| `gropt.m`                       | Gradient Optimization tool (wrapper for MEX code) | Matlab wrapper for `mex_gropt_diff_fixN` / `mex_gropt_diff_fixdt`. Core logic is in C/MEX. Porting requires re-implementing C code or finding Python equivalent. |
| `homodyneshow.m`                | MRI utility (Homodyne Reconstruction for partial Fourier) | Implements homodyne reconstruction. Port core logic to Python recon module. Visualization is Matlab-specific. |
| `imagescn.m`                    | Advanced Visualization utility (multi-image display) | Sophisticated multi-image display. Python equivalent uses Matplotlib, possibly with helpers. |
| `ksquare.m`                     | K-space data simulator (square object)     | Generates k-space data for a square object. Port to Python if needed for testing. |
| `lfphase.m`                     | Utility function (low-frequency random phase generator) | Generates smooth random phase maps. Port to Python if needed for simulations. |
| `lplot.m`                       | Plotting utility wrapper                   | Labels plot, sets grid, calls `setprops.m`. Python uses Matplotlib directly. |
| `lsfatwater.m`                  | MRI utility (Least-Squares Fat-Water Separation - Dixon type) | Implements Dixon-like fat-water separation. Valuable algorithm to port to Python image processing module. |
| `mag.m`                         | Utility function (vector/matrix magnitude) | Calculates L2 norm. Replaceable with `numpy.linalg.norm` or equivalent. |
| `magphase.m`                    | Visualization utility (magnitude and phase plots) | Plots magnitude and phase of complex data. Python equivalent uses Matplotlib. |
| `make.m`                        | Build script (for `gropt` MEX files)       | Compiles C source files for `gropt` into MEX. Not for direct porting. Informs `gropt.m` status. |
| `makenoisykspace.m`             | Script (generates noisy k-space data)      | Creates phantom, adds noise in k-space. Python example/test script. |
| `mc2mr.m`                       | Utility function (EPG state to Cartesian M) | Converts [F+; F-; Z] to [Mx; My; Mz]. Could be EPG util in Python. |
| `mr2mc.m`                       | Utility function (Cartesian M to EPG state) | Converts [Mx; My; Mz] to [F+; F-; Z]. Could be EPG util in Python. |
| `mingrad.m`                     | Gradient design utility (time-optimal gradient) | Calculates fastest gradient for a given area. Valuable to port to Python gradient utils. |
| `msinc.m`                       | Utility function (windowed sinc)           | Generates a Hamming-windowed sinc function. Replaceable with SciPy components. |
| `nft.m`                         | Utility function (Normalized centered 2D FFT) | Normalized version of `ft.m`. Replaceable with PyTorch FFT and manual normalization. |
| `nift.m`                        | Utility function (Normalized centered 2D IFFT) | Normalized version of `ift.m`. Replaceable with PyTorch IFFT and manual normalization. (Filename in source `nft.m` but likely `nift.m`). |
| `nlegend.m`                     | Plotting utility wrapper (numerical legend) | Creates plot legend from numerical array. Minor helper, Matplotlib handles legends. |
| `plot_waveform.m`               | Visualization script (gradient waveform analysis) | Plots gradient, moments, slew, PNS. Python equivalent uses Matplotlib and ported utils. |
| `plotc.m`                       | Plotting utility (complex vector)          | Plots real, imag, mag of complex data. Python equivalent uses Matplotlib. |
| `plotgradinfo.m`                | Visualization script (gradient analysis results) | Uses `calcgradinfo` and plots results. Python equivalent uses Matplotlib and ported `calcgradinfo`. |
| `plotm.m`                       | Visualization utility (3D magnetization vector display) | Shows 3D spin vectors in multiple views using `showspins`. Python equivalent uses Matplotlib 3D. |
| `psf2d.m`                       | MRI utility (2D PSF calculation and display) | Calculates and plots 2D PSF from k-space. Port core logic; visualize with Matplotlib. |
| `senseweights.m`                | MRI utility (SENSE weights, g-factor calculation) | Calculates SENSE parameters. Valuable. Check Python MRI libs or port. |
| `setprops.m`                    | Obsolete/Non-functional script             | File content indicates it's not working and commented out. Mark for deletion. |
| `showspins.m`                   | Visualization utility (3D spin vector display) | Displays 3D spin vectors, with color options. Python equivalent uses Matplotlib 3D. |
| `smart_subplot.m`               | Plotting utility (subplot layout enhancement) | Creates subplots with custom spacing. Matplotlib `GridSpec` or `subplots_adjust` offer this. |
| `sweptfreqrf.m`                 | Demonstration Script (swept frequency RF pulse design & sim) | Designs and simulates a swept frequency RF pulse. Could be Python example. |
| `time2freq.m`                   | Utility function (FFT frequency axis generation) | Converts time array to FFT frequency array. Covered by `numpy.fft.fftfreq` and `numpy.fft.fftshift`. |
| `vds.m`                         | Specialized k-space trajectory design (Variable Density Spiral) | Complex function to design VDS trajectories. Includes `findq2r2`, `qdf` helpers. Major porting effort if needed. |
| `vecdcf.m`                      | MRI utility (Density Correction Factor calculation - vector method) | Calculates DCFs for non-Cartesian trajectories. Valuable. Port to Python if needed. |
