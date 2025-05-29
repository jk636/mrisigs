import unittest
# Assumes physical_properties.py is in the same directory or in PYTHONPATH
from physical_properties import get_diffusion_coeff_h2o, DIFFUSION_COEFF_H2O_DATA 
import warnings 
import io # For capturing print statements (warnings)
import sys # For redirecting stdout

class TestPhysicalProperties(unittest.TestCase):

    def test_get_diffusion_coeff_h2o_valid_temperatures(self):
        self.assertAlmostEqual(get_diffusion_coeff_h2o(1), DIFFUSION_COEFF_H2O_DATA[1])
        self.assertAlmostEqual(get_diffusion_coeff_h2o(10), DIFFUSION_COEFF_H2O_DATA[10])
        self.assertAlmostEqual(get_diffusion_coeff_h2o(25), DIFFUSION_COEFF_H2O_DATA[25])
        self.assertAlmostEqual(get_diffusion_coeff_h2o(40), DIFFUSION_COEFF_H2O_DATA[40])

    def test_get_diffusion_coeff_h2o_float_input_temperature(self):
        # Test if float input is handled (e.g. floored with warning)
        # Capture stdout to check for print warnings
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        result = get_diffusion_coeff_h2o(20.5)
        
        sys.stdout = sys.__stdout__  # Reset stdout
        
        self.assertAlmostEqual(result, DIFFUSION_COEFF_H2O_DATA[20])
        self.assertIn("Warning: Temperature 20.5°C floored to 20°C for lookup.", captured_output.getvalue())

        # Test for float that is effectively an integer (e.g., 20.0)
        captured_output_int_float = io.StringIO()
        sys.stdout = captured_output_int_float
        
        result_int_float = get_diffusion_coeff_h2o(20.0)
        
        sys.stdout = sys.__stdout__
        self.assertAlmostEqual(result_int_float, DIFFUSION_COEFF_H2O_DATA[20])
        # No flooring warning should be printed for 20.0 as it's treated as int(20)
        self.assertNotIn("floored to", captured_output_int_float.getvalue())


    def test_get_diffusion_coeff_h2o_out_of_range_low(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        self.assertIsNone(get_diffusion_coeff_h2o(0))
        sys.stdout = sys.__stdout__
        self.assertIn("Warning: Temperature 0°C is outside the calibrated range [1, 40].", captured_output.getvalue())


    def test_get_diffusion_coeff_h2o_out_of_range_high(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        self.assertIsNone(get_diffusion_coeff_h2o(41))
        sys.stdout = sys.__stdout__
        self.assertIn("Warning: Temperature 41°C is outside the calibrated range [1, 40].", captured_output.getvalue())
        
    def test_get_diffusion_coeff_h2o_non_numeric_input(self):
        captured_output_str = io.StringIO()
        sys.stdout = captured_output_str
        self.assertIsNone(get_diffusion_coeff_h2o("test"))
        sys.stdout = sys.__stdout__
        self.assertIn("Warning: Invalid temperature value 'test'. Must be a number.", captured_output_str.getvalue())

        captured_output_list = io.StringIO()
        sys.stdout = captured_output_list
        self.assertIsNone(get_diffusion_coeff_h2o([20]))
        sys.stdout = sys.__stdout__
        self.assertIn("Warning: Invalid temperature value '[20]'. Must be a number.", captured_output_list.getvalue())

    def test_get_diffusion_coeff_h2o_string_numeric_input(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        result = get_diffusion_coeff_h2o("22")
        sys.stdout = sys.__stdout__
        self.assertAlmostEqual(result, DIFFUSION_COEFF_H2O_DATA[22])
        self.assertIn("Warning: Temperature was string, converted to float 22.0°C.", captured_output.getvalue())


if __name__ == '__main__':
    unittest.main()
