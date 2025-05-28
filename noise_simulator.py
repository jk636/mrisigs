import torch
import numpy as np # For np.allclose in tests if needed

class NoiseSimulator:
    def __init__(self, device=None):
        """
        Initializes NoiseSimulator.
        Manages device consistency for tensors created within the class.
        """
        self.device = device if device is not None else torch.device('cpu')

    def _to_tensor(self, data, dtype=torch.float32, allow_none=False):
        """Helper to convert data to a PyTorch tensor on self.device."""
        if data is None:
            if allow_none: return None
            else: raise ValueError("Input data cannot be None unless allow_none is True.")
        
        if not isinstance(data, torch.Tensor):
            return torch.tensor(data, dtype=dtype, device=self.device)
        return data.to(device=self.device, dtype=dtype)

    def corrnoise(self, mean_vec=None, R_covariance=None, n_samples=100, num_channels=3):
        """
        Generates correlated noise.
        Outputs real-valued noise by default.
        """
        # Determine num_channels from inputs if possible, else use default
        if R_covariance is not None:
            R_covariance = self._to_tensor(R_covariance, dtype=torch.float32) # Assuming real covariance
            actual_num_channels = R_covariance.shape[0]
            if R_covariance.shape[0] != R_covariance.shape[1]:
                raise ValueError("R_covariance must be a square matrix.")
        elif mean_vec is not None:
            mean_vec = self._to_tensor(mean_vec, dtype=torch.float32)
            actual_num_channels = mean_vec.shape[0]
        else:
            actual_num_channels = num_channels

        # Default mean_vec
        if mean_vec is None:
            mean_vec_processed = torch.zeros(actual_num_channels, device=self.device, dtype=torch.float32)
        else:
            mean_vec_processed = self._to_tensor(mean_vec, dtype=torch.float32)
            if mean_vec_processed.shape[0] != actual_num_channels:
                raise ValueError(f"mean_vec length {mean_vec_processed.shape[0]} does not match derived num_channels {actual_num_channels}")

        # Default R_covariance
        if R_covariance is None:
            R_cov_processed = torch.eye(actual_num_channels, device=self.device, dtype=torch.float32)
        else:
            R_cov_processed = self._to_tensor(R_covariance, dtype=torch.float32) # Already converted if used for actual_num_channels
            if R_cov_processed.shape[0] != actual_num_channels:
                 raise ValueError(f"R_covariance dimensions {R_cov_processed.shape} do not match derived num_channels {actual_num_channels}")


        # Symmetrize R_covariance (assuming R is real)
        R_cov_processed = 0.5 * (R_cov_processed + R_cov_processed.T)

        # Eigenvalue decomposition
        # For real symmetric matrices, eigvals are real, eigvecs can be chosen real.
        # torch.linalg.eig returns complex tensors by default.
        eigenvalues_complex, eigenvectors_complex = torch.linalg.eig(R_cov_processed)
        eigenvalues = eigenvalues_complex.real
        eigenvectors = eigenvectors_complex.real # Eigenvectors of real symmetric matrix are real

        # Check for non-positive eigenvalues
        if torch.any(eigenvalues <= 0):
            # Using a very small tolerance for "non-positive"
            non_positive_mask = eigenvalues <= 1e-7 # Tolerance for floating point
            if torch.any(non_positive_mask):
                print(f"Warning: R_covariance has non-positive eigenvalues. Clamping them for noise generation.")
                eigenvalues = torch.clamp(eigenvalues, min=1e-7) # Clamp to small positive

        # Transformation matrix W such that W @ W.T approximates R_covariance
        # W = V @ sqrt(diag(D))
        sqrt_diag_D = torch.diag(torch.sqrt(eigenvalues)) # eigenvalues already clamped positive
        W_transform = eigenvectors @ sqrt_diag_D
        W_transform = W_transform.float() # Ensure W is float if inputs were float

        # Generate uncorrelated random samples (standard normal)
        # Output should be real noise, so randn is appropriate.
        random_samples_uncorrelated = torch.randn(actual_num_channels, n_samples, device=self.device, dtype=torch.float32)

        # Generate correlated noise (centered at zero)
        correlated_noise_centered = W_transform @ random_samples_uncorrelated

        # Add mean
        correlated_noise = correlated_noise_centered + mean_vec_processed.unsqueeze(1)

        # Empirical covariance for checking
        R_output_check = (correlated_noise_centered @ correlated_noise_centered.T) / n_samples
        
        return correlated_noise, R_output_check


if __name__ == '__main__':
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {test_device} for tests ---")
    simulator = NoiseSimulator(device=test_device)

    # Test 1: Default parameters
    print("\n--- Test 1: Default parameters ---")
    n_s = 10000
    def_noise, def_R_check = simulator.corrnoise(n_samples=n_s, num_channels=3)
    print(f"Generated noise shape: {def_noise.shape}")
    print(f"Empirical R shape: {def_R_check.shape}")
    assert def_noise.shape == (3, n_s), "Default noise shape mismatch"
    assert def_R_check.shape == (3, 3), "Default R_check shape mismatch"
    # For default (identity R), empirical R should be close to identity
    expected_R_def = torch.eye(3, device=test_device)
    print("Expected R (Identity):\n", expected_R_def.cpu().numpy())
    print("Empirical R:\n", def_R_check.cpu().numpy())
    assert torch.allclose(def_R_check, expected_R_def, atol=0.1), "Default R_check not close to identity"
    # Mean should be close to zero
    assert torch.allclose(torch.mean(def_noise, dim=1), torch.zeros(3, device=test_device), atol=0.1), "Default noise mean not close to zero"

    # Test 2: Custom mean and R
    print("\n--- Test 2: Custom mean and R ---")
    custom_mean = torch.tensor([1.0, 2.0, 3.0], device=test_device)
    custom_R = torch.tensor([[1.0, 0.5, 0.2],
                             [0.5, 1.2, 0.3],
                             [0.2, 0.3, 1.5]], device=test_device)
    
    cust_noise, cust_R_check = simulator.corrnoise(mean_vec=custom_mean, R_covariance=custom_R, n_samples=n_s)
    print(f"Generated noise shape: {cust_noise.shape}")
    print(f"Empirical R shape: {cust_R_check.shape}")
    assert cust_noise.shape[0] == custom_mean.shape[0], "Custom noise channel mismatch"
    assert cust_R_check.shape == custom_R.shape, "Custom R_check shape mismatch"
    print("Expected R (Custom):\n", custom_R.cpu().numpy())
    print("Empirical R:\n", cust_R_check.cpu().numpy())
    assert torch.allclose(cust_R_check, custom_R, atol=0.1), "Custom R_check not close to target R"
    # Mean should be close to custom_mean
    print("Expected Mean (Custom):\n", custom_mean.cpu().numpy())
    print("Empirical Mean:\n", torch.mean(cust_noise, dim=1).cpu().numpy())
    assert torch.allclose(torch.mean(cust_noise, dim=1), custom_mean, atol=0.1), "Custom noise mean not close to target mean"

    # Test 3: R_covariance with non-positive eigenvalues (e.g., not strictly PD, or indefinite)
    print("\n--- Test 3: R_covariance not strictly positive definite ---")
    # This matrix has eigenvalues approx 2.0, 0.5, -0.5 (indefinite)
    # R_non_psd = torch.tensor([[1.0, 0.0, 1.0],
    #                           [0.0, 0.5, 0.0],
    #                           [1.0, 0.0, 1.0]], device=test_device)
    # Let's make one that's PSD but has a zero eigenvalue (singular)
    R_singular = torch.tensor([[1.0, 1.0, 0.0],
                               [1.0, 1.0, 0.0],
                               [0.0, 0.0, 2.0]], device=test_device) # Eigenvalues will be 2, 2, 0 for [[1,1],[1,1]] and [[2]] blocks
    # Eigenvalues of [[1,1],[1,1]] are 2 and 0. So for R_singular, they are 2, 2, 0.
    
    print("Testing with singular R (one zero eigenvalue):")
    # Expect a warning about non-positive eigenvalues and clamping
    singular_noise, singular_R_check = simulator.corrnoise(R_covariance=R_singular, n_samples=n_s)
    print(f"Generated noise shape (singular R): {singular_noise.shape}")
    # The check here is more about whether it runs and produces output of correct shape.
    # The empirical R_check will be different from original R_singular due to clamping.
    # It should approximate R_singular_clamped where 0 eigenvalue is replaced by 1e-7.
    # For example, R_singular_clamped = V @ diag([2,2,1e-7]) @ V.T
    assert singular_noise.shape[0] == R_singular.shape[0], "Singular R noise channel mismatch"
    print("Empirical R (from singular R_input):\n", singular_R_check.cpu().numpy())
    # Check that the variance of the component corresponding to the original zero eigenvalue is small.
    # (This requires knowing the eigenvector for the zero eigenvalue)
    # For R_singular = [[1,1,0],[1,1,0],[0,0,2]], eigenvectors are approx [1/sqrt(2), -1/sqrt(2), 0] (for eig 0),
    # [1/sqrt(2), 1/sqrt(2), 0] (for eig 2), [0,0,1] (for eig 2).
    # So, (noise_x - noise_y) should have small variance.
    variance_diff_xy = torch.var(singular_noise[0,:] - singular_noise[1,:])
    print(f"Variance of (noise_x - noise_y) for singular R case: {variance_diff_xy.item()} (should be small due to clamped eigenvalue)")
    assert variance_diff_xy < 0.1, "Variance for direction of zero eigenvalue is not small after clamping."


    print("\nAll tests for NoiseSimulator seem to run.")
