import torch # Not strictly needed here, but often included in the project
import math # For math.floor

# Data from Diffusion_Coeff_H20.m (diffusion coefficients in mm^2/s)
# Index corresponds to temperature in Celsius.
DIFFUSION_COEFF_H2O_DATA = {
    1: 1.1750e-3, 2: 1.2110e-3, 3: 1.2479e-3, 4: 1.2856e-3, 5: 1.3241e-3,
    6: 1.3635e-3, 7: 1.4038e-3, 8: 1.4450e-3, 9: 1.4870e-3, 10: 1.5300e-3,
    11: 1.5740e-3, 12: 1.6188e-3, 13: 1.6646e-3, 14: 1.7114e-3, 15: 1.7592e-3,
    16: 1.8079e-3, 17: 1.8576e-3, 18: 1.9084e-3, 19: 1.9602e-3, 20: 2.0130e-3,
    21: 2.0668e-3, 22: 2.1218e-3, 23: 2.1778e-3, 24: 2.2349e-3, 25: 2.2930e-3,
    26: 2.3523e-3, 27: 2.4127e-3, 28: 2.4743e-3, 29: 2.5370e-3, 30: 2.6009e-3,
    31: 2.6659e-3, 32: 2.7321e-3, 33: 2.7995e-3, 34: 2.8681e-3, 35: 2.9380e-3,
    36: 3.0090e-3, 37: 3.0813e-3, 38: 3.1549e-3, 39: 3.2298e-3, 40: 3.3059e-3
}

def get_diffusion_coeff_h2o(temperature_celsius: int) -> float | None:
    """
    Returns the diffusion coefficient of water (in mm^2/s) at a given temperature.

    Based on data from http://dtrx.de/od/diff/ for temperatures 1-40°C.

    Args:
        temperature_celsius: Integer temperature in Celsius (1-40).
                             Floats will be floored with a warning.

    Returns:
        The diffusion coefficient as a float, or None if the temperature
        is outside the valid range [1, 40] or input is invalid.
    """
    if not isinstance(temperature_celsius, (int, float)):
        try:
            # Attempt to convert if it's a string representation of a number
            temperature_celsius = float(temperature_celsius)
            print(f"Warning: Temperature was string, converted to float {temperature_celsius}°C.")
        except (ValueError, TypeError):
            print(f"Warning: Invalid temperature value '{temperature_celsius}'. Must be a number.")
            return None

    if isinstance(temperature_celsius, float):
        if temperature_celsius != math.floor(temperature_celsius): # Check if it has a fractional part
            temp_int = int(math.floor(temperature_celsius))
            print(f"Warning: Temperature {temperature_celsius}°C floored to {temp_int}°C for lookup.")
            temperature_celsius = temp_int
        else: # It's a float like 20.0
            temperature_celsius = int(temperature_celsius)
    
    # At this point, temperature_celsius should be an int, or the function returned None

    if temperature_celsius in DIFFUSION_COEFF_H2O_DATA:
        return DIFFUSION_COEFF_H2O_DATA[temperature_celsius]
    else:
        # Check if it was originally a valid number before potential flooring made it out of range
        # Or if it was an int from the start but out of range
        print(f"Warning: Temperature {temperature_celsius}°C is outside the calibrated range [1, 40]. Returning None.")
        return None

# Example usage for basic check (optional, will be in unit tests)
if __name__ == '__main__':
    print(f"D at 20°C: {get_diffusion_coeff_h2o(20)}")
    print(f"D at 25°C: {get_diffusion_coeff_h2o(25)}")
    print(f"D at 0°C: {get_diffusion_coeff_h2o(0)}") # Expected: None and warning
    print(f"D at 41°C: {get_diffusion_coeff_h2o(41)}") # Expected: None and warning
    print(f"D at 20.5°C: {get_diffusion_coeff_h2o(20.5)}") # Expected: Value for 20°C and warning
    print(f"D at '22': {get_diffusion_coeff_h2o('22')}") # Expected: Value for 22°C and warning
    print(f"D at 'invalid': {get_diffusion_coeff_h2o('invalid')}") # Expected: None and warning
    print(f"D at [20]: {get_diffusion_coeff_h2o([20])}") # Expected: None and warning
