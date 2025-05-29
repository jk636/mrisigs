# Summary of MATLAB (.m file) Porting to Python/PyTorch

This document tracks the status of refactoring original MATLAB `.m` script files from this repository into the new Python/PyTorch-based MRI simulation library.

The general approach is to replace MATLAB script functionalities with Python functions or methods within the new class-based structure, primarily using PyTorch for numerical computations. Once a `.m` file's core functionality is confirmed to be covered by a Python equivalent and unit tested, the original `.m` file will be removed from the active codebase.

## Porting Status:

| MATLAB File             | Python Module / Class / Function                                     | Status / Notes                                                                 |
|-------------------------|----------------------------------------------------------------------|--------------------------------------------------------------------------------|
| `Diffusion_Coeff_H20.m` | `physical_properties.py::get_diffusion_coeff_h2o()`                | Ported. `.m` file ready for removal.                                           |
| `Rad229_Structure_Definitions.m` | `mri_definitions.py` (various dataclasses)                     | Ported (conceptual definitions and examples). `.m` file ready for removal.     |
| `Rad229_MRI_sys_config.m` | `mri_definitions.py::SystemParameters` (default values)            | Ported (default values integrated). `.m` file ready for removal.               |
| `ghist.m`               | `plotting_utils.py::plot_histogram_with_gaussian_fit()`            | Ported. `.m` file ready for removal.                                           |
| `mag.m`                 | `torch.abs()` or `torch.linalg.norm()`                               | Functionality covered by PyTorch. `.m` file ready for removal.                 |
| `msinc.m`               | `mri_signal_processor.py::MRISignalProcessor.generate_windowed_sinc()` | Ported. `.m` file ready for removal.                                           |
| `gaussian2d.m`          | `mri_signal_processor.py::MRISignalProcessor.gaussian()` (2D case)   | Functionality covered. `.m` file ready for removal.                          |
| `xrot.m`                | `mri_rotation.py::MRIRotation.xrot()`                                | Ported. `.m` file ready for removal.                                           |
| `yrot.m`                | `mri_rotation.py::MRIRotation.yrot()`                                | Ported. `.m` file ready for removal.                                           |
| `zrot.m`                | `mri_rotation.py::MRIRotation.zrot()`                                | Ported. `.m` file ready for removal.                                           |
| `throt.m`               | `mri_rotation.py::MRIRotation.throt()`                               | Ported. `.m` file ready for removal.                                           |
| `relax.m`               | `mri_relaxation.py::MRIRelaxation.relax()`                           | Ported. `.m` file ready for removal.                                           |
| `ft.m`                  | `mri_signal_processor.py::MRISignalProcessor.ft()`                   | Ported. `.m` file ready for removal.                                           |
| `ift.m`                 | `mri_signal_processor.py::MRISignalProcessor.ift()`                  | Ported. `.m` file ready for removal.                                           |
| `sinc.m`                | `mri_signal_processor.py::MRISignalProcessor.sinc()`                 | Ported. `.m` file ready for removal.                                           |
| `lplot.m`               | `plotting_utils.py::lplot()`                                         | Ported. `.m` file ready for removal.                                           |
| `epg_rf.m`              | `epg_simulator.py::EPGSimulator.epg_rf()`                            | Ported. `.m` file ready for removal.                                           |
| `epg_grad.m`            | `epg_simulator.py::EPGSimulator.epg_grad()`                          | Ported. `.m` file ready for removal.                                           |
| `epg_relax.m`           | `epg_simulator.py::EPGSimulator.epg_relax()`                         | Ported. `.m` file ready for removal.                                           |
| `epg_grelax.m`          | `epg_simulator.py::EPGSimulator.epg_grelax()`                        | Ported. `.m` file ready for removal.                                           |
| `epg_trim.m`            | `epg_simulator.py::EPGSimulator.epg_trim()`                          | Ported. `.m` file ready for removal.                                           |
| `cropim.m`              | `image_manipulation.py::ImageManipulation.cropim()`                  | Ported. `.m` file ready for removal.                                           |
| `zpadcrop.m`            | `image_manipulation.py::ImageManipulation.zpadcrop()`                | Ported. `.m` file ready for removal.                                           |
| `dispim.m`              | `image_manipulation.py::ImageManipulation.dispim()`                  | Ported. `.m` file ready for removal.                                           |
| `epg_spins2FZ.m`        | `epg_simulator.py::EPGSimulator.epg_spins2FZ()`                      | Ported. `.m` file ready for removal.                                           |
| `epg_FZ2spins.m`        | `epg_simulator.py::EPGSimulator.epg_FZ2spins()`                      | Ported. `.m` file ready for removal.                                           |
| `epg_zrot.m`            | `epg_simulator.py::EPGSimulator.epg_zrot()`                          | Ported. `.m` file ready for removal.                                           |
| `epg_m0.m`              | `epg_simulator.py::EPGSimulator.epg_m0()`                            | Ported. `.m` file ready for removal.                                           |
| `epg_mgrad.m`           | `epg_simulator.py::EPGSimulator.epg_mgrad()`                         | Ported. `.m` file ready for removal.                                           |
| `epg_cpmg.m`            | `epg_simulator.py::EPGSimulator.epg_cpmg()`                          | Ported. `.m` file ready for removal.                                           |
| `epg_gradecho.m`        | `epg_simulator.py::EPGSimulator.epg_gradecho()`                      | Ported. `.m` file ready for removal.                                           |
| `epg_stim.m`            | `epg_simulator.py::EPGSimulator.epg_stim_calc()`                     | Ported (as `epg_stim_calc`). `.m` file ready for removal.                      |
| `epg_show.m`            | `epg_simulator.py::EPGSimulator.show_matrix()`                       | Core data display ported. `.m` file ready for removal.                         |
| `epg_showstate.m`       | `epg_simulator.py::EPGSimulator.show_matrix()`                       | Core data display ported. `.m` file ready for removal.                         |
| `bvalue.m`              | `gradient_kspace_tools.py::GradientKSpaceTools.bvalue()`             | Ported. `.m` file ready for removal.                                           |
| `calcgradinfo.m`        | `gradient_kspace_tools.py::GradientKSpaceTools.calcgradinfo()`       | Ported. `.m` file ready for removal.                                           |
| `mingrad.m`             | `gradient_kspace_tools.py::GradientKSpaceTools.mingrad()`            | Ported. `.m` file ready for removal.                                           |
| `ksquare.m`             | `gradient_kspace_tools.py::GradientKSpaceTools.ksquare()`            | Ported. `.m` file ready for removal.                                           |
| `time2freq.m`           | `gradient_kspace_tools.py::GradientKSpaceTools.time2freq()`          | Ported. `.m` file ready for removal.                                           |
| `gridmat.m`             | `gradient_kspace_tools.py::GradientKSpaceTools.gridmat()`            | Ported. `.m` file ready for removal.                                           |
| `vecdcf.m`              | `gradient_kspace_tools.py::GradientKSpaceTools.vecdcf()`             | Ported. `.m` file ready for removal.                                           |
| `vds.m`                 | `gradient_kspace_tools.py` (components available)                    | Components for VDS available. Direct script not ported. `.m` file ready for removal. |
| `kb.m`                  | `gradient_kspace_tools.py::GradientKSpaceTools._kb()`                | Ported (as helper). `.m` file ready for removal.                               |
| `cumint.m`              | `gradient_kspace_tools.py::GradientKSpaceTools._cumint()`            | Ported (as helper). `.m` file ready for removal.                               |
| `findq2r2.m`            | N/A                                                                  | Not directly ported; functionality may be internal or superseded. Ok to remove. |
| `qdf.m`                 | N/A                                                                  | Not directly ported; functionality may be internal or superseded. Ok to remove. |
| `corrnoise.m`           | `noise_simulator.py::NoiseSimulator.corrnoise()`                     | Ported. `.m` file ready for removal.                                           |
| `circ.m`                | User-generated with PyTorch / `mri_signal_processor`                 | Basic shape, easily replicable. `.m` file ready for removal.                   |
| `diamond.m`             | User-generated with PyTorch                                          | Basic shape, easily replicable. `.m` file ready for removal.                   |
| `sq.m`                  | User-generated with PyTorch                                          | Basic shape, easily replicable. `.m` file ready for removal.                   |
| `angleim.m`             | `image_manipulation.py::ImageManipulation.dispangle()`               | Ported (display part). Generation via `torch.atan2`. `.m` file ready for removal.|
| `csens2d.m`             | `mri_reconstructor.py::MRIReconstructor.csens2d()`                   | Ported. `.m` file ready for removal.                                           |
| `senseweights.m`        | `mri_reconstructor.py::MRIReconstructor.senseweights()`              | Ported. `.m` file ready for removal.                                           |
| `senserecon.m`          | `mri_reconstructor.py::MRIReconstructor.senserecon()`                | Ported. `.m` file ready for removal.                                           |
| `rmscombine.m`          | `mri_reconstructor.py::MRIReconstructor.rmscombine()`                | Ported. `.m` file ready for removal.                                           |
| `dispkspim.m`           | `image_manipulation.py::ImageManipulation.dispkspim()`               | Ported. `.m` file ready for removal.                                           |
| `dispangle.m`           | `image_manipulation.py::ImageManipulation.dispangle()`               | Ported. `.m` file ready for removal.                                           |
| `displogim.m`           | `image_manipulation.py::ImageManipulation.displogim()`               | Ported. `.m` file ready for removal.                                           |
| `abprop.m`              | `Python/examples.py::exampleB1_15` (logic ported)                    | Logic ported to example. `.m` file ready for removal.                        |
|                         |                                                                      |                                                                                |

**Further files will be added to this list as they are processed.**
