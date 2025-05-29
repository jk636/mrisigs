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
|                         |                                                                      |                                                                                |

**Further files will be added to this list as they are processed.**
