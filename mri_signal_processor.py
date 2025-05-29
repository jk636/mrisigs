import torch

class MRISignalProcessor:
    def ft(self, data):
        """
        Performs a 2D Fast Fourier Transform.
        Equivalent to np.fft.fftshift(np.fft.fft2(np.fft.fftshift(data))).
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.complex64)
        # Apply fftshift before fft2
        data_shifted = torch.fft.fftshift(data)
        # Apply fft2
        transformed_data = torch.fft.fft2(data_shifted)
        # Apply fftshift after fft2
        result = torch.fft.fftshift(transformed_data)
        return result

    def ift(self, data):
        """
        Performs a 2D Inverse Fast Fourier Transform.
        Equivalent to np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(data))).
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.complex64)
        # Apply fftshift before ifft2
        data_shifted = torch.fft.fftshift(data)
        # Apply ifft2
        transformed_data = torch.fft.ifft2(data_shifted)
        # Apply ifftshift after ifft2
        result = torch.fft.ifftshift(transformed_data)
        return result

    def sinc(self, x):
        """
        Computes the normalized sinc function: sin(pi*x) / (pi*x).
        Handles the case x=0 where sinc(0) = 1.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Use torch.where to handle x=0 separately
        # pi_x = torch.pi * x
        # result = torch.where(x == 0, torch.tensor(1.0), torch.sin(pi_x) / pi_x)
        
        # A more direct way to implement sinc using torch.sinc
        # Note: torch.sinc is defined as sin(pi*x)/(pi*x)
        result = torch.sinc(x)
        return result

    def gaussian(self, x, mnx, sigx, y=None, mny=None, sigy=None):
        """
        Calculates a 1D or 2D Gaussian distribution.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # 1D Gaussian for x
        x_gauss = torch.exp(-(x - mnx)**2 / (2 * sigx**2)) / (torch.sqrt(2 * torch.pi) * sigx)

        if y is None:
            return x_gauss
        else:
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float32)
            
            if mny is None or sigy is None:
                raise ValueError("mny and sigy must be provided for 2D Gaussian")

            # 1D Gaussian for y
            y_gauss = torch.exp(-(y - mny)**2 / (2 * sigy**2)) / (torch.sqrt(2 * torch.pi) * sigy)
            
            # Outer product to form 2D Gaussian
            # x_gauss.unsqueeze(1) creates a column vector
            # y_gauss.unsqueeze(0) creates a row vector
            return x_gauss.unsqueeze(1) @ y_gauss.unsqueeze(0)

if __name__ == '__main__':
    # Example Usage (optional - for testing during development)
    processor = MRISignalProcessor()

    # Test ft and ift
    print("Testing ft and ift...")
    data_np = [[1, 2], [3, 4]]
    data_torch = torch.tensor(data_np, dtype=torch.complex64)
    
    print("Original data:\n", data_torch)
    
    ft_data = processor.ft(data_torch)
    print("FT data:\n", ft_data)
    
    ift_data = processor.ift(ft_data)
    print("IFT data (should be close to original):\n", ift_data)

    # Test sinc
    print("\nTesting sinc...")
    x_vals_list = [-2., -1., 0., 1., 2.]
    x_vals_tensor = torch.tensor(x_vals_list)
    sinc_vals = processor.sinc(x_vals_tensor)
    print(f"sinc({x_vals_list}) = {sinc_vals.tolist()}")
    sinc_val_at_0 = processor.sinc(torch.tensor(0.0))
    print(f"sinc(0) = {sinc_val_at_0.item()}")
    sinc_val_at_0_list = processor.sinc([0.0]) # Test with list input
    print(f"sinc([0.0]) = {sinc_val_at_0_list.tolist()}")


    # Test gaussian
    print("\nTesting gaussian...")
    x_coords = torch.linspace(-3, 3, 10)
    
    # 1D Gaussian
    gauss_1d = processor.gaussian(x_coords, mnx=0, sigx=1)
    print("1D Gaussian (x_coords, mnx=0, sigx=1):\n", gauss_1d)

    # 2D Gaussian
    y_coords = torch.linspace(-2, 2, 8)
    gauss_2d = processor.gaussian(x_coords, mnx=0, sigx=1, y=y_coords, mny=0.5, sigy=0.5)
    print("2D Gaussian (x_coords, mnx=0, sigx=1, y_coords, mny=0.5, sigy=0.5) shape:\n", gauss_2d.shape)
    # print("2D Gaussian values:\n", gauss_2d)

    print("\nChecking if torch.sinc handles 0 correctly (it should by definition)")
    test_sinc_zero = torch.sinc(torch.tensor(0.0))
    print(f"torch.sinc(0.0) = {test_sinc_zero}")
    test_sinc_non_zero = torch.sinc(torch.tensor(1.0)) # sin(pi)/pi = 0
    print(f"torch.sinc(1.0) = {test_sinc_non_zero}")
    test_sinc_non_zero_half = torch.sinc(torch.tensor(0.5)) # sin(pi/2)/(pi/2) = 1/(pi/2) = 2/pi
    print(f"torch.sinc(0.5) = {test_sinc_non_zero_half}")

    print("Script finished.")
