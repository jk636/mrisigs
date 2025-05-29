# THIS FILE IS OBSOLETE.
# Its functionality has been refactored into separate class-based modules:
# - mri_signal_processor.py
# - mri_relaxation.py
# - mri_rotation.py
# - epg_simulator.py
# - mri_reconstructor.py
# - gradient_kspace_tools.py
# - image_manipulation.py
# - noise_simulator.py
# - phantom_generator.py
#
# Please update your imports and usage accordingly.
# For example, instead of `import mrsigpy as mrs` and `mrs.xrot()`,
# you might use `from mri_rotation import MRIRotation` and then `rot_module = MRIRotation(); rot_module.xrot()`.
#
# The original content of this file might be found in version control history
# or potentially in a file named `mrsigpy_legacy.py` if such a backup was made.
print("WARNING: The 'mrsigpy.py' module is obsolete. Please refactor your code to use the new class-based modules.")

# To prevent accidental usage of any old functions if some parts of this file were not cleared:
# raise ImportError("mrsigpy.py is obsolete. Please use the new refactored modules.")
# For now, let's just have the print statement.
