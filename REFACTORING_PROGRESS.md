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
| `ft.m`                          | Deleted                                    | Centered 2D FFT. Replaceable with `torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(data)))`. Consider for deletion or inclusion in a Python util module if widely used. File deleted. |
| `ift.m`                         | Deleted                                    | Centered 2D IFFT. Replaceable with `torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(data)))`. Consider for deletion or inclusion in a Python util module. File deleted. |
| `sinc.m`                        | Deleted                                    | Normalized sinc function. Use `torch.sinc()`. Mark for deletion. File deleted. |
| `gaussian.m`                    | Deleted                                    | Standard 1D Gaussian PDF. Can be custom Python function if needed. File deleted. |
| `Rad229_MRI_Phantom.m`          | Likely covered by Python/Rad229_MRI_Phantom.py | Matlab version seems incomplete. Python version likely supersedes it. Needs verification, then mark for deletion. |
| `epgx/Test_SPGR.m`              | Deleted                                    | (Moved from EPGX section for clarity) Tests epg_X_rfspoil (MT-SPGR). Core simulation covered by EPGXSimulator.epgx_rfspoil. Comparison to external Malik et al. code and specific RF_phase_cycle helper (likely quadratic, covered by epg_rfspoil_quadratic_phase or can be generated) are test setup details. Python test can replicate forward sim. Consider for deletion. File deleted. |
| `epgx/Test_fitting_MET.m`       | Deleted                                    | (Moved from EPGX section for clarity) Simulates MET data using epg_X_CMPG (covered by EPGXSimulator.epgx_cpmg) and then uses qMRLab for model fitting. Fitting part is external. Not for direct porting beyond data generation. Consider for deletion or as Python example for data generation. File deleted. |
| `Diffusion_Coeff_H20.m`         | Ported to Python                           | Data provider function. Ported to `physical_properties.py` as `DIFFUSION_COEFF_H2O_MM2_PER_S` dictionary and `get_diffusion_coeff_h2o()` function. Mark for deletion. |
| `Rad229_Conventional_FlowComp.m`| Specialized gradient design function       | Designs flow-compensated gradient waveforms (Pelc et al.). Valuable to port to a Python gradient design module if functionality is desired. |
| `Rad229_Conventional_FlowEncode.m`| Specialized gradient design function       | Designs flow-encoding gradient waveforms (Pelc et al.). Valuable to port to a Python gradient design module if functionality is desired. |
| `Rad229_Fourier_Encoding.m`     | Educational script/demo                    | Demonstrates Fourier encoding principles. Suitable for conversion to a Python example/Jupyter notebook. |
| `Rad229_MRI_Signal_Eqn.m`       | Incomplete script/demo                     | Marked as incomplete in source. Demonstrates MRI signal equation using other demo functions (possibly missing). Likely not for direct porting. |
| `bssfp.m`                       | Ported to Python                           | Analytical bSSFP signal calculation. Ported to `analytical_signals.py` as `calculate_bssfp_signal()`. Mark for deletion. |
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
| `bvalue.m`                      | Ported to Python                           | Utility function (b-value calculation). Ported to `mri_utils.py` as `calculate_b_value()`. Mark for deletion. |
| `calc_harmonics.m`              | Specialized utility (spherical harmonics for gradient non-linearity) | Calculates field values from SH coefficients (Siemens convention). For advanced MRI processing. |
| `calc_spherical_harmonics.m`    | Specialized utility (spherical harmonics for gradient non-linearity) | Similar to calc_harmonics.m. For advanced MRI processing.             |
| `calcgradinfo.m`                | Gradient analysis utility                  | Calculates k-space, moments, slew rate, coil voltage from gradient waveform. Valuable to port to Python gradient/sequence utils. |
| `circ.m`                        | Utility function (2D circle generator)     | Creates a 2D circular mask. Easily done with NumPy. Low priority for direct porting. |
| `conventional_flowcomp.m`       | Specialized gradient design function (flow comp) | Similar to `Rad229_Conventional_FlowComp.m`. Needs comparison to determine if redundant or unique. |
| `conventional_flowencode.m`     | Specialized gradient design function (flow encode) | Similar to `Rad229_Conventional_FlowEncode.m`. Needs comparison to determine if redundant or unique. |
| `corrnoise.m`                   | Deleted                                    | Generates correlated Gaussian noise. Covered by `numpy.random.multivariate_normal`. Consider for deletion. File deleted. |
| `cropim.m`                      | Deleted                                    | Crops image around center. Easily done with NumPy slicing. Low priority for direct porting. File deleted. |
| `csens2d.m`                     | MRI utility (coil sensitivity map estimation) | Estimates coil sensitivity maps from k-space calibration data. Important for SENSE. Check Python MRI libs or port. |
| `demo.m`                        | Demonstration Script (`gropt` examples)    | Shows usage of `gropt`, `get_min_TE_diff`, `plot_waveform`. Convert to Python example/notebook if `gropt` is ported. |
| `demo_moments.m`                | Demonstration Script (gradient moment design) | Shows gradient design with moment constraints, likely using `gropt`. Convert to Python example if relevant tools are ported. |
| `demo_pns.m`                    | Demonstration Script (PNS constraints in gradient design) | Shows b-value optimization with PNS constraints. Convert to Python example if relevant tools are ported. |
| `design_symmetric_gradients.m`  | Specialized gradient design function (DWI waveforms) | Designs monopolar, bipolar, modified bipolar DWI gradients. Includes `trapTransform` helper. Valuable to port if DWI sequence design is needed. |
| `diamond.m`                     | Deleted                                    | Creates a 2D diamond mask. Easily done with NumPy. Low priority. File deleted. |
| `dispangle.m`                   | Visualization utility (displays phase of complex data) | Wrapper for `dispim` to show phase. Python equivalent uses Matplotlib. |
| `dispim.m`                      | Visualization utility (displays image magnitude) | Core image display. Python equivalent uses Matplotlib. |
| `dispkspim.m`                   | Visualization utility (k-space and image display) | Displays k-space/image mag/phase in 2x2 subplot. Python equivalent uses Matplotlib & FFT functions. |
| `displogim.m`                   | Visualization utility (displays log-magnitude of image) | Displays log-magnitude using `dispim`. Python equivalent uses Matplotlib. |
| `exrecsignal.m`                 | Ported to Python                           | Analytical SPGR signal calculation. Ported to `analytical_signals.py` as `calculate_spgr_signal()`. Mark for deletion. |
| `gaussian2d.m`                  | Utility function (2D Gaussian generator)   | Creates a 2D Gaussian image using 1D `gaussian.m`. Python utility if needed. |
| `get_bval.m`                    | Ported to Python                           | Utility function (b-value calculation with inversion). Ported to `mri_utils.py` as `calculate_b_value_refocused()`. Mark for deletion. |
| `get_coords.m`                  | Utility function (coordinate transformation) | Calculates a coordinate transformation matrix. Utility depends on specific coordinate system needs. |
| `get_min_TE_diff.m`             | Optimization script (min TE for diffusion using `gropt`) | Finds minimum TE for diffusion sequence from `gropt`. Porting depends on `gropt`. |
| `get_min_TE_free.m`             | Optimization script (min TE for `gropt` free mode) | Finds minimum TE for `gropt` free mode. Porting depends on `gropt`. |
| `get_moments.m`                 | Ported to Python                           | Utility function (gradient moment calculation with inversion). Ported to `mri_utils.py` as `calculate_gradient_moments()`. Mark for deletion. |
| `get_stim.m`                    | PNS calculation utility                    | Calculates PNS metric from gradient waveform based on a model. Check vs `pns_constraint_op.py`. Port if not covered. |
| `ghist.m`                       | Visualization utility (histogram with Gaussian fit) | Plots histogram and Gaussian fit. Python equivalent uses Matplotlib/SciPy. |
| `gradanimation.m`               | Visualization/Animation Script             | Example script for animating magnetization vectors. Not for direct porting. |
| `gresignal.m`                   | Ported to Python                           | Analytical Gradient-Spoiled GRE signal calculation. Ported to `analytical_signals.py` as `calculate_gre_spoiled_signal()`. Mark for deletion. |
| `grid_phantom.m`                | Phantom generator (grid pattern)           | Creates a circular phantom with a grid. Port to Python if this phantom is needed. |
| `gridmat.m`                     | MRI utility (2D Gridding for non-Cartesian recon) | Grids non-Cartesian k-space data using Kaiser-Bessel kernel. Assess vs Python NUFFT libs or port. Includes `kb` helper. |
| `gropt.m`                       | Gradient Optimization tool (wrapper for MEX code) | Matlab wrapper for `mex_gropt_diff_fixN` / `mex_gropt_diff_fixdt`. Core logic is in C/MEX. Porting requires re-implementing C code or finding Python equivalent. |
| `homodyneshow.m`                | MRI utility (Homodyne Reconstruction for partial Fourier) | Implements homodyne reconstruction. Port core logic to Python recon module. Visualization is Matlab-specific. |
| `imagescn.m`                    | Advanced Visualization utility (multi-image display) | Sophisticated multi-image display. Python equivalent uses Matplotlib, possibly with helpers. |
| `ksquare.m`                     | K-space data simulator (square object)     | Generates k-space data for a square object. Port to Python if needed for testing. |
| `lfphase.m`                     | Utility function (low-frequency random phase generator) | Generates smooth random phase maps. Port to Python if needed for simulations. |
| `lplot.m`                       | Plotting utility wrapper                   | Labels plot, sets grid, calls `setprops.m`. Python uses Matplotlib directly. |
| `lsfatwater.m`                  | MRI utility (Least-Squares Fat-Water Separation - Dixon type) | Implements Dixon-like fat-water separation. Valuable algorithm to port to Python image processing module. |
| `mag.m`                         | Deleted                                    | Calculates L2 norm. Replaceable with `numpy.linalg.norm` or equivalent. File deleted. |
| `magphase.m`                    | Visualization utility (magnitude and phase plots) | Plots magnitude and phase of complex data. Python equivalent uses Matplotlib. |
| `make.m`                        | Build script (for `gropt` MEX files)       | Compiles C source files for `gropt` into MEX. Not for direct porting. Informs `gropt.m` status. |
| `makenoisykspace.m`             | Script (generates noisy k-space data)      | Creates phantom, adds noise in k-space. Python example/test script. |
| `mc2mr.m`                       | Utility function (EPG state to Cartesian M) | Converts [F+; F-; Z] to [Mx; My; Mz]. Could be EPG util in Python. |
| `mr2mc.m`                       | Utility function (Cartesian M to EPG state) | Converts [Mx; My; Mz] to [F+; F-; Z]. Could be EPG util in Python. |
| `mingrad.m`                     | Ported to Python                           | Gradient design utility (time-optimal gradient). Ported to `mri_utils.py` as `design_time_optimal_gradient()`. Mark for deletion. |
| `msinc.m`                       | Utility function (windowed sinc)           | Generates a Hamming-windowed sinc function. Replaceable with SciPy components. |
| `nft.m`                         | Deleted                                    | Normalized version of `ft.m`. Replaceable with PyTorch FFT and manual normalization. File deleted. |
| `nift.m`                        | Deleted                                    | Normalized version of `ift.m`. Replaceable with PyTorch IFFT and manual normalization. (Filename in source `nft.m` but likely `nift.m`). File deleted. |
| `nlegend.m`                     | Plotting utility wrapper (numerical legend) | Creates plot legend from numerical array. Minor helper, Matplotlib handles legends. |
| `plot_waveform.m`               | Visualization script (gradient waveform analysis) | Plots gradient, moments, slew, PNS. Python equivalent uses Matplotlib and ported utils. |
| `plotc.m`                       | Plotting utility (complex vector)          | Plots real, imag, mag of complex data. Python equivalent uses Matplotlib. |
| `plotgradinfo.m`                | Visualization script (gradient analysis results) | Uses `calcgradinfo` and plots results. Python equivalent uses Matplotlib and ported `calcgradinfo`. |
| `plotm.m`                       | Visualization utility (3D magnetization vector display) | Shows 3D spin vectors in multiple views using `showspins`. Python equivalent uses Matplotlib 3D. |
| `psf2d.m`                       | MRI utility (2D PSF calculation and display) | Calculates and plots 2D PSF from k-space. Port core logic; visualize with Matplotlib. |
| `senseweights.m`                | MRI utility (SENSE weights, g-factor calculation) | Calculates SENSE parameters. Valuable. Check Python MRI libs or port. |
| `setprops.m`                    | Deleted                                    | File content indicates it's not working and commented out. Mark for deletion. File deleted. |
| `showspins.m`                   | Visualization utility (3D spin vector display) | Displays 3D spin vectors, with color options. Python equivalent uses Matplotlib 3D. |
| `smart_subplot.m`               | Plotting utility (subplot layout enhancement) | Creates subplots with custom spacing. Matplotlib `GridSpec` or `subplots_adjust` offer this. |
| `sweptfreqrf.m`                 | Demonstration Script (swept frequency RF pulse design & sim) | Designs and simulates a swept frequency RF pulse. Could be Python example. |
| `time2freq.m`                   | Deleted                                    | Converts time array to FFT frequency array. Covered by `numpy.fft.fftfreq` and `numpy.fft.fftshift`. File deleted. |
| `vds.m`                         | Specialized k-space trajectory design (Variable Density Spiral) | Complex function to design VDS trajectories. Includes `findq2r2`, `qdf` helpers. Major porting effort if needed. |
| `vecdcf.m`                      | MRI utility (Density Correction Factor calculation - vector method) | Calculates DCFs for non-Cartesian trajectories. Valuable. Port to Python if needed. |
| `Rad229_Eddy_Currents_Demo.m`   | Educational script/demo (Eddy Currents, outdated) | Demonstrates eddy current effects and pre-emphasis. Marked as outdated. Low priority for porting. |
| `Rad229_Fourier_Encoding_Demo.m`| Incomplete educational script/demo (Fourier Encoding) | Intended to demo Fourier encoding but marked as incomplete. |
| `Rad229_Freq_Encode_Demo.m`     | Gradient design function (Frequency Encoding Demo) | Designs frequency encoding gradients. Could be part of a Python sequence building example/module. |
| `Rad229_MRI_Resolution_Phantom.m`| Phantom generator (resolution phantom)     | Creates a bar pattern phantom with T2-star values. Port to Python if this phantom is needed. |
| `Rad229_MRI_sys_config.m`       | Ported to Python                           | Configuration data (MRI system parameters). Ported to `physical_properties.py` as `RAD229_MRI_SYSTEM_CONFIG` dictionary. Mark for deletion. |
| `Rad229_Motion_Artifacts_Demo.m`| Educational script/demo (Motion Artifacts) | Demonstrates bulk and pulsatile motion artifacts using a phantom. Could be Python example/notebook. |
| `Rad229_PSD_fig.m`              | Visualization utility (Pulse Sequence Diagram plotting) | Generates PSD plots. Python equivalent uses Matplotlib. (Function name `PAM_PSD_fig` in file). |
| `Rad229_Phase_Encode_Demo.m`    | Educational script/demo (Phase Encode Gradient Design) | Designs and demonstrates phase encoding gradients. |
| `Rad229_RandomWalk_Diffusion.m` | Educational script/demo (Random Walk Diffusion Simulation) | Simulates and visualizes random walk diffusion with gradients. Includes local helper functions. |
| `Rad229_Random_Walk.m`          | Educational script/demo (Basic Random Walk) | Simulates 1D and 3D random walks. |
| `Rad229_Slice_Select_Demo.m`    | Educational script/demo (Slice Select Design) | Designs RF and gradients for slice selection. |
| `Rad229_Structure_Definitions.m`| Documentation (Structure Definitions)      | Describes Matlab struct conventions for MRI parameters. Informs Python data structures. |
| `Rad229_fig_style.m`            | Plotting utility (Figure styling)          | Applies custom styles to Matlab figures. Python uses Matplotlib styling. |
| `Rad229_plot_style.m`           | Plotting utility (Plot color and line styling) | Defines and applies custom plot colors/styles. Python uses Matplotlib styling. |
| `whirl.m`                       | Specialized k-space trajectory design (WHIRL) | Designs WHIRL k-space trajectories (J. Pipe). Major porting effort if needed. |
| `zpadcrop.m`                    | Utility function (Image zero-padding/cropping) | Zero-pads or crops an image, centered. Replaceable with NumPy/SciPy. |
| `Lecture_05A_Nonlinear_Gradients.mlx` | Educational MLX (Matlab Live Script)   | Interactive lecture/demo. Consider for manual conversion to Jupyter Notebook if content is desired in Python format. |
| `Lecture_05B_Eddy_Currents.mlx`       | Educational MLX (Matlab Live Script)   | Interactive lecture/demo. Consider for manual conversion to Jupyter Notebook. |
| `Lecture_05C_Concomitant_Fields.mlx`  | Educational MLX (Matlab Live Script)   | Interactive lecture/demo. Consider for manual conversion to Jupyter Notebook. |
| `Rad229_CODE.mlx`                     | Educational MLX (Matlab Live Script)   | Appears to be a collection of code for Rad229 course. Assess content for useful examples; consider conversion to Jupyter Notebooks. |
| `Rad229_EPI_Chemical_Shift_Demo.mlx`  | Educational MLX (Matlab Live Script)   | EPI demo. Consider for manual conversion to Jupyter Notebook. |
| `Rad229_EPI_Ghosting_Demo.mlx`        | Educational MLX (Matlab Live Script)   | EPI demo. Consider for manual conversion to Jupyter Notebook. |
| `Rad229_EPI_OffRes_Dist_Demo.mlx`     | Educational MLX (Matlab Live Script)   | EPI demo. Consider for manual conversion to Jupyter Notebook. |
| `Rad229_EPI_T2star_Blurring_Demo.mlx` | Educational MLX (Matlab Live Script)   | EPI demo. Consider for manual conversion to Jupyter Notebook. |
| `Rad229_Eddy_Currents_Demo.mlx`     | Educational MLX (Matlab Live Script)   | Eddy currents demo (MLX version). Consider for manual conversion to Jupyter Notebook. |
| `Rad229_Flow_Encoding_Gradients.mlx`| Educational MLX (Matlab Live Script)   | Flow encoding demo. Consider for manual conversion to Jupyter Notebook. |
| `Rad229_Motion_Artifacts_Demo.mlx`  | Educational MLX (Matlab Live Script)   | Motion artifacts demo (MLX version). Consider for manual conversion to Jupyter Notebook. |
| `Rad229_Nonlinear_Gradients.mlx`    | Educational MLX (Matlab Live Script)   | Nonlinear gradients demo (MLX version, see also Lecture_05A). Consider for manual conversion to Jupyter Notebook. |
