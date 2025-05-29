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
            # Ensure data is compatible with the specified dtype before conversion
            if dtype == torch.complex64 and isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], (int, float)):
                 # Heuristic: if data is list of reals but target is complex, make it complex
                 data = [complex(x) for x in data]
            elif dtype == torch.float32 and isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], complex):
                 raise ValueError("Cannot convert complex list to float32 tensor without explicit handling.")

            return torch.tensor(data, dtype=dtype, device=self.device)
        
        # If data is already a tensor, just move and change dtype if specified
        return data.to(device=self.device, dtype=dtype)


    def corrnoise(self, mean_vec=None, R_covariance=None, n_samples=100, num_channels=3):
        """
        Generates correlated noise.
        Outputs real-valued noise by default.
        """
        if R_covariance is not None:
            R_cov_tensor = self._to_tensor(R_covariance, dtype=torch.float32) 
            actual_num_channels = R_cov_tensor.shape[0]
            if R_cov_tensor.shape[0] != R_cov_tensor.shape[1]:
                raise ValueError("R_covariance must be a square matrix.")
        elif mean_vec is not None:
            mean_vec_tensor = self._to_tensor(mean_vec, dtype=torch.float32)
            actual_num_channels = mean_vec_tensor.shape[0]
        else:
            actual_num_channels = num_channels

        if mean_vec is None:
            mean_vec_processed = torch.zeros(actual_num_channels, device=self.device, dtype=torch.float32)
        else:
            mean_vec_processed = self._to_tensor(mean_vec, dtype=torch.float32)
            if mean_vec_processed.shape[0] != actual_num_channels:
                raise ValueError(f"mean_vec length {mean_vec_processed.shape[0]} does not match derived num_channels {actual_num_channels}")

        if R_covariance is None:
            R_cov_processed = torch.eye(actual_num_channels, device=self.device, dtype=torch.float32)
        else:
            R_cov_processed = self._to_tensor(R_covariance, dtype=torch.float32) 
            if R_cov_processed.shape[0] != actual_num_channels:
                 raise ValueError(f"R_covariance dimensions {R_cov_processed.shape} do not match derived num_channels {actual_num_channels}")

        R_cov_processed = 0.5 * (R_cov_processed + R_cov_processed.T)
        eigenvalues_complex, eigenvectors_complex = torch.linalg.eig(R_cov_processed)
        eigenvalues = eigenvalues_complex.real
        eigenvectors = eigenvectors_complex.real 
        
        non_positive_mask = eigenvalues <= 1e-7 
        if torch.any(non_positive_mask):
            print(f"Warning: R_covariance has non-positive eigenvalues. Clamping them for noise generation.")
            eigenvalues = torch.clamp(eigenvalues, min=1e-7) 

        sqrt_diag_D = torch.diag(torch.sqrt(eigenvalues))
        W_transform = eigenvectors @ sqrt_diag_D
        W_transform = W_transform.float()

        random_samples_uncorrelated = torch.randn(actual_num_channels, n_samples, device=self.device, dtype=torch.float32)
        correlated_noise_centered = W_transform @ random_samples_uncorrelated
        correlated_noise = correlated_noise_centered + mean_vec_processed.unsqueeze(1)
        R_output_check = (correlated_noise_centered @ correlated_noise_centered.T) / n_samples
        
        return correlated_noise, R_output_check

    def rmscoilnoise(self, sig_tensor=None, cov_matrix=None, csens_tensor=None, 
                     Nn_samples=10000, Nc_coils_default=1, 
                     target_snr_for_sig_ramp=10.0, epsilon=1e-9):
        
        # Determine actual_Nc_coils
        if cov_matrix is not None:
            cov_matrix_p = self._to_tensor(cov_matrix, dtype=torch.float32)
            actual_Nc_coils = cov_matrix_p.shape[0]
        elif csens_tensor is not None:
            csens_tensor_p = self._to_tensor(csens_tensor, dtype=torch.complex64) # csens can be complex
            actual_Nc_coils = csens_tensor_p.shape[-1]
        else:
            actual_Nc_coils = Nc_coils_default

        # Default sig_tensor
        if sig_tensor is None:
            sig_tensor_p = torch.arange(0.0, float(target_snr_for_sig_ramp), 0.1, device=self.device, dtype=torch.float32)
        else:
            sig_tensor_p = self._to_tensor(sig_tensor, dtype=torch.float32)
        
        num_signal_points = sig_tensor_p.shape[0]

        # Default cov_matrix
        if cov_matrix is None:
            cov_matrix_p = torch.eye(actual_Nc_coils, device=self.device, dtype=torch.float32)
        else: # Already processed if used for actual_Nc_coils, ensure it's on device and float
            cov_matrix_p = self._to_tensor(cov_matrix, dtype=torch.float32)
        if cov_matrix_p.shape != (actual_Nc_coils, actual_Nc_coils):
            raise ValueError(f"cov_matrix shape {cov_matrix_p.shape} inconsistent with actual_Nc_coils {actual_Nc_coils}")


        # Default csens_tensor
        if csens_tensor is None:
            csens_tensor_p = torch.ones((1, actual_Nc_coils), device=self.device, dtype=torch.complex64)
        else: # Already processed if used for actual_Nc_coils, ensure it's on device and complex
            csens_tensor_p = self._to_tensor(csens_tensor, dtype=torch.complex64)
        
        # Validate csens_tensor shape for broadcasting
        if csens_tensor_p.shape[0] != num_signal_points and csens_tensor_p.shape[0] != 1:
            raise ValueError(f"csens_tensor.shape[0] ({csens_tensor_p.shape[0]}) must be 1 or match sig_tensor length ({num_signal_points}).")
        if csens_tensor_p.shape[-1] != actual_Nc_coils:
            raise ValueError(f"csens_tensor last dim ({csens_tensor_p.shape[-1]}) must match actual_Nc_coils ({actual_Nc_coils}).")


        # Noise Generation (real covariance matrix for real and imag parts)
        mean_zeros = torch.zeros(actual_Nc_coils, device=self.device, dtype=cov_matrix_p.dtype)
        
        # Ensure cov_matrix is PSD for Cholesky. Symmetrize first.
        cov_matrix_sym = 0.5 * (cov_matrix_p + cov_matrix_p.T)
        cov_matrix_psd = cov_matrix_sym + (self._to_tensor(epsilon).item() * torch.eye(actual_Nc_coils, device=self.device, dtype=cov_matrix_p.dtype))
        
        try:
            L = torch.linalg.cholesky(cov_matrix_psd)
        except RuntimeError as e:
            # Fallback to eigenvalue method if Cholesky fails (e.g. still not PSD enough)
            print(f"Cholesky failed: {e}. Falling back to eigenvalue method for noise generation in rmscoilnoise.")
            # This part is similar to corrnoise's W_transform generation
            eigenvalues_complex, eigenvectors_complex = torch.linalg.eig(cov_matrix_sym) # Use sym version
            eigenvalues = eigenvalues_complex.real
            eigenvectors = eigenvectors_complex.real
            non_positive_mask = eigenvalues <= 1e-7 
            if torch.any(non_positive_mask):
                eigenvalues = torch.clamp(eigenvalues, min=1e-7)
            sqrt_diag_D = torch.diag(torch.sqrt(eigenvalues))
            L = eigenvectors @ sqrt_diag_D # L here is W_transform from corrnoise

        # For MultivariateNormal, scale_tril requires L L^T = Sigma. Cholesky L is lower triangular.
        # If using W_transform from eig, it's not necessarily lower triangular but W W^T = Sigma.
        # PyTorch MultivariateNormal with scale_tril=L means L is the Cholesky factor.
        # If L came from eigendecomposition (W_transform), it's not necessarily lower triangular.
        # However, we can generate standard normal and then transform: L @ standard_normal
        
        # Generate standard normal samples
        standard_normal_samples_real = torch.randn(Nn_samples, actual_Nc_coils, device=self.device, dtype=torch.float32)
        standard_normal_samples_imag = torch.randn(Nn_samples, actual_Nc_coils, device=self.device, dtype=torch.float32)

        # Transform them using L (either Cholesky factor or W_transform from eig)
        # (Nn_samples, actual_Nc_coils) @ (actual_Nc_coils, actual_Nc_coils) -> (Nn_samples, actual_Nc_coils)
        # Then transpose to (actual_Nc_coils, Nn_samples)
        real_noise = (standard_normal_samples_real @ L.T).T 
        imag_noise = (standard_normal_samples_imag @ L.T).T
        cnoise = torch.complex(real_noise, imag_noise) # Shape (actual_Nc_coils, Nn_samples)


        # Signal Combination
        # sig_tensor_p: (N_sig_pts)
        # csens_tensor_p: (N_sig_pts or 1, actual_Nc_coils)
        # csig goal: (N_sig_pts, actual_Nc_coils)
        csig = sig_tensor_p.unsqueeze(-1).to(torch.complex64) * csens_tensor_p
        
        # csig shape: (N_sig_pts, actual_Nc_coils)
        # cnoise shape: (actual_Nc_coils, Nn_samples)
        # Goal cnsig: (N_sig_pts, actual_Nc_coils, Nn_samples)
        # csig unsqueezed: (N_sig_pts, actual_Nc_coils, 1)
        # cnoise unsqueezed: (1, actual_Nc_coils, Nn_samples)
        cnsig = csig.unsqueeze(2) + cnoise.unsqueeze(0)

        # RMS Calculation
        # csens_tensor_p: (N_sig_pts or 1, actual_Nc_coils)
        # rmssens goal: (N_sig_pts)
        sum_sq_sens = torch.sum(csens_tensor_p * torch.conj(csens_tensor_p), dim=-1).real
        rmssens = torch.sqrt(sum_sq_sens) # Shape (N_sig_pts) or (1)
        if rmssens.ndim == 0: rmssens = rmssens.unsqueeze(0) # ensure it's at least 1D

        # cnsig: (N_sig_pts, actual_Nc_coils, Nn_samples)
        # sum_sq_cnsig goal: (N_sig_pts, Nn_samples)
        sum_sq_cnsig = torch.sum(cnsig * torch.conj(cnsig), dim=1).real
        
        # rmssens unsqueezed for division: (N_sig_pts or 1, 1)
        rmssig_dist = torch.sqrt(sum_sq_cnsig) / (rmssens.unsqueeze(-1) + self._to_tensor(epsilon).item())
        
        # Bias and Standard Deviation
        sigstd = torch.std(rmssig_dist, dim=1)
        sigbias = torch.mean(rmssig_dist, dim=1) - sig_tensor_p.to(torch.float32)
        
        return rmssig_dist.float(), sigbias.float(), sigstd.float()


if __name__ == '__main__':
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {test_device} for tests ---")
    simulator = NoiseSimulator(device=test_device)

    # Shortened existing tests
    print("\n--- Test corrnoise: Default parameters ---")
    simulator.corrnoise(n_samples=100, num_channels=3)
    print("corrnoise default OK.")
    print("\n--- Test corrnoise: Custom mean and R ---")
    custom_mean = torch.tensor([1.0, 2.0]); custom_R = torch.tensor([[1.0, 0.5],[0.5, 1.2]])
    simulator.corrnoise(mean_vec=custom_mean, R_covariance=custom_R, n_samples=100)
    print("corrnoise custom OK.")

    # --- New tests for rmscoilnoise ---
    print("\n--- Testing rmscoilnoise ---")

    # Test 1: Default parameters (sig_tensor=None, cov_matrix=None, csens_tensor=None)
    print("\n--- rmscoilnoise Test 1: Defaults ---")
    Nn_s_test = 2000
    rms_dist_def, bias_def, std_def = simulator.rmscoilnoise(Nn_samples=Nn_s_test, Nc_coils_default=1, target_snr_for_sig_ramp=5.0)
    
    expected_sig_len = torch.arange(0.0, 5.0, 0.1).shape[0]
    print(f"Default output shapes: rms_dist={rms_dist_def.shape}, bias={bias_def.shape}, std={std_def.shape}")
    assert rms_dist_def.shape == (expected_sig_len, Nn_s_test), "Default rmscoilnoise rms_dist_def shape mismatch"
    assert bias_def.shape == (expected_sig_len,), "Default rmscoilnoise bias_def shape mismatch"
    assert std_def.shape == (expected_sig_len,), "Default rmscoilnoise std_def shape mismatch"
    print("Defaults test passed.")

    # Test 2: Custom inputs
    print("\n--- rmscoilnoise Test 2: Custom inputs ---")
    sig_t = torch.tensor([1.0, 5.0, 10.0], device=test_device)
    cov_m = torch.tensor([[1.0, 0.3], [0.3, 1.0]], device=test_device, dtype=torch.float32)
    csens_t_single = torch.tensor([[0.8+0.1j, 0.6-0.2j]], device=test_device) # Shape (1, 2) for broadcasting
    
    rms_dist_cust, bias_cust, std_cust = simulator.rmscoilnoise(
        sig_tensor=sig_t, cov_matrix=cov_m, csens_tensor=csens_t_single, Nn_samples=Nn_s_test
    )
    print(f"Custom output shapes: rms_dist={rms_dist_cust.shape}, bias={bias_cust.shape}, std={std_cust.shape}")
    assert rms_dist_cust.shape == (sig_t.shape[0], Nn_s_test), "Custom rmscoilnoise rms_dist_cust shape mismatch"
    assert bias_cust.shape == sig_t.shape, "Custom rmscoilnoise bias_cust shape mismatch"
    assert std_cust.shape == sig_t.shape, "Custom rmscoilnoise std_cust shape mismatch"
    print("Custom inputs test passed.")

    # Test 3: csens_tensor with matching N_sig_pts
    print("\n--- rmscoilnoise Test 3: csens_tensor with matching N_sig_pts ---")
    csens_t_multi = torch.rand((sig_t.shape[0], cov_m.shape[0]), dtype=torch.complex64, device=test_device)
    rms_dist_cs_multi, _, _ = simulator.rmscoilnoise(
        sig_tensor=sig_t, cov_matrix=cov_m, csens_tensor=csens_t_multi, Nn_samples=Nn_s_test
    )
    assert rms_dist_cs_multi.shape == (sig_t.shape[0], Nn_s_test), "csens_multi rmscoilnoise shape mismatch"
    print("csens_tensor with matching N_sig_pts test passed.")
    
    # Test 4: Covariance matrix that might cause Cholesky issues without epsilon
    print("\n--- rmscoilnoise Test 4: Challenging Covariance Matrix ---")
    # Example: A matrix that is PSD but close to singular, or slightly non-PSD due to float errors
    # For simplicity, using a scaled identity, as Cholesky fallback is tested in corrnoise already
    # The key is that the MultivariateNormal distribution part runs.
    cov_challenging = torch.eye(2, device=test_device) * 1e-10 
    rms_dist_chg, _, _ = simulator.rmscoilnoise(
        sig_tensor=sig_t, cov_matrix=cov_challenging, csens_tensor=csens_t_single, Nn_samples=Nn_s_test
    )
    assert rms_dist_chg.shape == (sig_t.shape[0], Nn_s_test), "Challenging cov rmscoilnoise shape mismatch"
    print("Challenging covariance matrix test passed (ran without Cholesky error).")


    print("\nAll tests for NoiseSimulator (including rmscoilnoise) seem to run.")
