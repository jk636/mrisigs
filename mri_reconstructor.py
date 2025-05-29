import torch

class MRIReconstructor:
    def __init__(self, device=None):
        """
        Initializes the MRI Reconstructor.
        Device can be 'cpu' or 'cuda' etc.
        """
        self.device = device if device is not None else torch.device('cpu')

    def csens2d(self, klow):
        """
        Calculates coil sensitivity maps from low-resolution k-space data.
        Input klow: A PyTorch tensor representing k-space data for multiple coils (Ny, Nx, Nc).
        """
        if not isinstance(klow, torch.Tensor):
            klow = torch.tensor(klow, device=self.device)
        if klow.dtype != torch.complex64:
            klow = klow.to(dtype=torch.complex64)
        klow = klow.to(self.device)

        sz = klow.shape
        if len(sz) != 3:
            raise ValueError(f"klow must be a 3D tensor (Ny, Nx, Nc). Got shape {sz}")
        ny, nx, nc = sz

        cs_ffty = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(klow, dim=0), dim=0), dim=0)
        cs = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(cs_ffty, dim=1), dim=1), dim=1)
        cs = cs.to(torch.complex64)

        cmrms = torch.sqrt(torch.sum(torch.conj(cs) * cs, dim=-1).real) # Ensure real before sqrt
        cmrms = cmrms.to(torch.float32) 

        meancm = torch.mean(cmrms) 
        zero_rms_mask = (cmrms == 0)
        
        cmrms_div = cmrms.clone()
        cmrms_div[zero_rms_mask] = meancm / 100.0 if meancm > 1e-9 else 1e-9

        cs_flat = cs.reshape(ny * nx, nc)
        
        final_cs_content = cs.clone() 
        final_cs_flat = final_cs_content.reshape(ny * nx, nc)
        cs_flat_zero_mask = zero_rms_mask.reshape(ny * nx)
        final_cs_flat[cs_flat_zero_mask, :] = (meancm / 10000.0 if meancm > 1e-9 else 1e-12) + 0j


        ncs_flat = cs_flat / cmrms_div.reshape(ny * nx, 1)
        
        ncs = ncs_flat.reshape(sz)
        final_cs = final_cs_flat.reshape(sz) 
        
        return ncs, final_cs, cmrms.reshape(ny, nx)


    def senseweights(self, coilsens, noisecov=None, gfactorcalc=False, noisecalc=False):
        if not isinstance(coilsens, torch.Tensor):
            coilsens = torch.tensor(coilsens, device=self.device)
        if coilsens.dtype != torch.complex64:
            coilsens = coilsens.to(dtype=torch.complex64)
        coilsens = coilsens.to(self.device)

        cs_shape = coilsens.shape
        Nc = cs_shape[-2]
        R_factor = cs_shape[-1] 
        
        imshape_dims = cs_shape[:-2] 
        Npts = coilsens.numel() // (R_factor * Nc)

        coilsens_reshaped = coilsens.reshape((Npts, Nc, R_factor))
        
        sense_weights_out = torch.zeros((Npts, R_factor, Nc), dtype=torch.complex64, device=self.device)

        if noisecov is None:
            noisecov_eff = torch.eye(Nc, dtype=torch.complex64, device=self.device)
        else:
            if not isinstance(noisecov, torch.Tensor):
                noisecov = torch.tensor(noisecov, device=self.device)
            if noisecov.dtype != torch.complex64:
                 noisecov_eff = noisecov.to(dtype=torch.complex64)
            else:
                noisecov_eff = noisecov
            noisecov_eff = noisecov_eff.to(self.device)

        try:
            psi_inv = torch.linalg.inv(noisecov_eff)
        except torch.linalg.LinAlgError as e:
            # print(f"Warning: Could not invert noise covariance matrix: {e}. Using pseudo-inverse.")
            psi_inv = torch.linalg.pinv(noisecov_eff)

        for i in range(Npts):
            C = coilsens_reshaped[i, :, :] 
            Ch = torch.conj(C.T)           
            Ch_psi_inv = Ch @ psi_inv            
            Ch_psi_inv_C = Ch_psi_inv @ C        
            try:
                Ch_psi_inv_C_inv = torch.linalg.inv(Ch_psi_inv_C)
            except torch.linalg.LinAlgError:
                Ch_psi_inv_C_inv = torch.linalg.pinv(Ch_psi_inv_C)
            sensemat = Ch_psi_inv_C_inv @ Ch_psi_inv
            sense_weights_out[i, :, :] = sensemat

        output_shape_list = list(imshape_dims) + [R_factor, Nc]
        sense_weights_out = sense_weights_out.reshape(tuple(output_shape_list))

        if gfactorcalc:
            print("g-factor calculation is not fully implemented in this version.")
        if noisecalc:
            print("Noise calculation is not fully implemented in this version.")
        return sense_weights_out

    # --- New methods start here ---

    def senserecon(self, signal, sweights):
        """ Applies SENSE combination to reconstruct the image. """
        if not isinstance(signal, torch.Tensor):
            signal = torch.tensor(signal, device=self.device)
        if signal.dtype != torch.complex64:
            signal = signal.to(dtype=torch.complex64)
        signal = signal.to(self.device)

        if not isinstance(sweights, torch.Tensor):
            sweights = torch.tensor(sweights, device=self.device)
        if sweights.dtype != torch.complex64:
            sweights = sweights.to(dtype=torch.complex64)
        sweights = sweights.to(self.device)

        ws_shape = sweights.shape
        imshape_dims = ws_shape[:-2] 
        Nc = ws_shape[-1] 
        R = ws_shape[-2]  
        Npts = sweights.numel() // (Nc * R)

        ss_shape = signal.shape
        if signal.shape[-1] != Nc:
            raise ValueError(f"Signal last dimension ({signal.shape[-1]}) must match SENSE weights Nc ({Nc}).")

        signal_reshaped = signal.reshape((Npts, Nc))
        sweights_reshaped = sweights.reshape((Npts, R, Nc))

        sigout = torch.zeros((Npts, R), dtype=torch.complex64, device=self.device)
        for i in range(Npts):
            s_point = signal_reshaped[i, :]          # Shape (Nc)
            w_point = sweights_reshaped[i, :, :]     # Shape (R, Nc)
            sigout[i, :] = torch.matmul(w_point, s_point) # (R,Nc) @ (Nc) -> (R)
        
        output_shape_tuple = tuple(list(imshape_dims) + [R])
        sigout_reshaped = sigout.reshape(output_shape_tuple)
        
        return sigout_reshaped

    def rmscombine(self, csignals, csens=None, epsilon=1e-12):
        """ Performs Root Mean Square (Sum of Squares - SOS) coil combination. """
        if not isinstance(csignals, torch.Tensor):
            csignals = torch.tensor(csignals, device=self.device)
        if csignals.dtype != torch.complex64: # Typically complex, but mag squared will be real
            csignals = csignals.to(dtype=torch.complex64)
        csignals = csignals.to(self.device)

        if csens is None:
            # Sum over the coil dimension (assumed to be the last one)
            rmssig = torch.sqrt(torch.sum(csignals * torch.conj(csignals), dim=-1).real)
        else:
            if not isinstance(csens, torch.Tensor):
                csens = torch.tensor(csens, device=self.device)
            if csens.dtype != torch.complex64:
                csens = csens.to(dtype=torch.complex64)
            csens = csens.to(self.device)

            if csignals.shape != csens.shape:
                raise ValueError(f"csignals shape {csignals.shape} and csens shape {csens.shape} must be the same.")
            
            csens_mag_sq = (csens * torch.conj(csens)).real
            rmssens = torch.sqrt(torch.sum(csens_mag_sq, dim=-1))
            
            # Numerator is sum of (weighted signal * conj(weighted signal))
            # Or, simpler: weighted_signal = csignals * conj(csens) -> then sum this over coils and divide by sum(csens_mag_sq)
            # The prompt implies: sqrt(sum(|csignals|^2)) / (sqrt(sum(|csens|^2)) + epsilon)
            # This is not standard sensitivity weighted combination.
            # Standard is: sum(csignals * conj(csens)) / (sum(csens * conj(csens)) + epsilon)
            # Let's implement what is in the prompt:
            csignals_mag_sq = (csignals * torch.conj(csignals)).real
            sum_csignals_mag_sq = torch.sum(csignals_mag_sq, dim=-1)
            rmssig = torch.sqrt(sum_csignals_mag_sq) / (rmssens + epsilon)
            
        return rmssig.to(torch.float32) # Result should be real


if __name__ == '__main__':
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {test_device} for tests ---")
    
    reconstructor = MRIReconstructor(device=test_device)

    # --- Test csens2d ---
    print("\n--- Testing csens2d ---")
    Ny_c, Nx_c, Nc_c = 16, 16, 4
    klow_test_data = torch.zeros((Ny_c, Nx_c, Nc_c), dtype=torch.complex64, device=test_device)
    klow_test_data[Ny_c // 2, Nx_c // 2, :] = 1.0 + 0.5j
    ncs, final_cs, cmrms_map = reconstructor.csens2d(klow_test_data)
    print(f"csens2d output shapes: ncs {ncs.shape}, final_cs {final_cs.shape}, cmrms_map {cmrms_map.shape}")
    assert ncs.shape == (Ny_c, Nx_c, Nc_c)
    assert final_cs.shape == (Ny_c, Nx_c, Nc_c)
    assert cmrms_map.shape == (Ny_c, Nx_c)

    # --- Test senseweights ---
    print("\n--- Testing senseweights ---")
    Npix_y_sw, Npix_x_sw, Nc_sw, R_sw = 8, 8, 4, 2
    coilsens_test_data = torch.rand((Npix_y_sw, Npix_x_sw, Nc_sw, R_sw), dtype=torch.complex64, device=test_device)
    sense_w = reconstructor.senseweights(coilsens_test_data)
    print(f"senseweights output shape: {sense_w.shape}")
    assert sense_w.shape == (Npix_y_sw, Npix_x_sw, R_sw, Nc_sw)

    # --- Test senserecon ---
    print("\n--- Testing senserecon ---")
    Npix_y_sr, Npix_x_sr, Nc_sr, R_sr = 8, 8, 4, 2
    # Dummy signal (Ny, Nx, Nc)
    signal_test_sr = torch.rand((Npix_y_sr, Npix_x_sr, Nc_sr), dtype=torch.complex64, device=test_device)
    # Dummy weights (Ny, Nx, R, Nc) - ensure R and Nc match from senseweights output
    sweights_test_sr = torch.rand((Npix_y_sr, Npix_x_sr, R_sr, Nc_sr), dtype=torch.complex64, device=test_device)
    
    recon_img_sr = reconstructor.senserecon(signal_test_sr, sweights_test_sr)
    print(f"senserecon output shape: {recon_img_sr.shape}")
    expected_sr_shape = (Npix_y_sr, Npix_x_sr, R_sr)
    assert recon_img_sr.shape == expected_sr_shape, f"senserecon: shape mismatch. Expected {expected_sr_shape}, got {recon_img_sr.shape}"

    # Test with flattened inputs for senserecon
    Npts_sr_flat = Npix_y_sr * Npix_x_sr
    signal_flat_sr = signal_test_sr.reshape(Npts_sr_flat, Nc_sr)
    sweights_flat_sr = sweights_test_sr.reshape(Npts_sr_flat, R_sr, Nc_sr)
    recon_img_flat_sr = reconstructor.senserecon(signal_flat_sr, sweights_flat_sr)
    print(f"senserecon output shape (flat input): {recon_img_flat_sr.shape}")
    expected_sr_flat_shape = (Npts_sr_flat, R_sr) # Output before final reshape to image dims
    assert recon_img_flat_sr.shape == expected_sr_flat_shape, "senserecon: shape mismatch (flat input)"
    # This test actually tests the internal sigout shape before final reshape, 
    # but the function is expected to reshape based on sweights input shape.
    # The function as written will try to infer imshape_dims from sweights_flat_sr which is (Npts, R, Nc)
    # So imshape_dims will be (Npts), and output will be (Npts, R). This is correct.

    # --- Test rmscombine ---
    print("\n--- Testing rmscombine ---")
    Ny_rc, Nx_rc, Nc_rc = 10, 10, 4
    csignals_test_rc = torch.rand((Ny_rc, Nx_rc, Nc_rc), dtype=torch.complex64, device=test_device)

    # Test without csens
    rms_no_sens = reconstructor.rmscombine(csignals_test_rc)
    print(f"rmscombine output shape (no csens): {rms_no_sens.shape}")
    assert rms_no_sens.shape == (Ny_rc, Nx_rc), "rmscombine (no csens): shape mismatch"
    assert rms_no_sens.dtype == torch.float32, "rmscombine (no csens): dtype mismatch"


    # Test with csens
    csens_test_rc = torch.rand((Ny_rc, Nx_rc, Nc_rc), dtype=torch.complex64, device=test_device)
    csens_test_rc[:,:,0] = 0 # Test epsilon for one coil
    rms_with_sens = reconstructor.rmscombine(csignals_test_rc, csens=csens_test_rc, epsilon=1e-9)
    print(f"rmscombine output shape (with csens): {rms_with_sens.shape}")
    assert rms_with_sens.shape == (Ny_rc, Nx_rc), "rmscombine (with csens): shape mismatch"
    assert rms_with_sens.dtype == torch.float32, "rmscombine (with csens): dtype mismatch"
    assert not torch.any(torch.isnan(rms_with_sens)), "NaNs in rms_with_sens"
    assert not torch.any(torch.isinf(rms_with_sens)), "Infs in rms_with_sens"


    # Test rmscombine with mismatched shapes for csens
    try:
        csens_mismatch_rc = torch.rand((Ny_rc, Nx_rc + 1, Nc_rc), dtype=torch.complex64, device=test_device)
        reconstructor.rmscombine(csignals_test_rc, csens=csens_mismatch_rc)
        raise AssertionError("rmscombine should have failed with mismatched shapes but didn't.")
    except ValueError as e:
        print(f"rmscombine correctly failed with mismatched shapes: {e}")

    print("\nAll tests for MRIReconstructor (including new methods) seem to run.")
