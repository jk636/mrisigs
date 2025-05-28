import torch

class GradientKSpaceTools:
    def __init__(self, device=None):
        """
        Initializes GradientKSpaceTools.
        Manages device consistency for tensors created within the class.
        """
        self.device = device if device is not None else torch.device('cpu')
        self.gamma_rad_g_s = 2 * torch.pi * 4258.0  # rad/s/G
        self.gamma_kHz_G = 4.258 # kHz/G, for mingrad convenience

    def _to_tensor(self, data, dtype=torch.float32):
        """Helper to convert data to a PyTorch tensor on self.device."""
        if not isinstance(data, torch.Tensor):
            return torch.tensor(data, dtype=dtype, device=self.device)
        return data.to(device=self.device, dtype=dtype)

    def time2freq(self, t):
        t = self._to_tensor(t, dtype=torch.float32)
        if t.ndim != 1:
            raise ValueError("Input time vector t must be 1D.")
        if t.numel() < 2:
            if t.numel() == 1: return torch.tensor([0.0], device=self.device)
            raise ValueError("Input time vector t must have at least 2 elements for dt calculation.")
        dt = t[1] - t[0]
        num_p = t.numel()
        f = torch.fft.fftshift(torch.fft.fftfreq(num_p, d=dt.item()))
        return f.to(self.device)

    def bvalue(self, gradwave, T_dwell):
        gradwave = self._to_tensor(gradwave, dtype=torch.float32)
        T_dwell_val = self._to_tensor(T_dwell, dtype=torch.float32).item()
        if gradwave.ndim != 1:
            raise ValueError("gradwave must be a 1D tensor.")
        int_g = torch.cumsum(gradwave, dim=0) * T_dwell_val
        b = (self.gamma_rad_g_s**2) * torch.sum(int_g**2) * T_dwell_val
        return b

    def calcgradinfo(self, g, T_dwell=0.000004, k0=0.0, R_ohms=0.35, L_mH=1.4, 
                     eta_inv_sec_per_volt_amp=1/56.0, gamma_hz_g=None):
        current_gamma_rad_g_s = self.gamma_rad_g_s
        if gamma_hz_g is not None:
            current_gamma_rad_g_s = self._to_tensor(gamma_hz_g).item() * 2 * torch.pi

        g_orig_ndim = g.ndim
        g_tensor = self._to_tensor(g, dtype=torch.float32) # Use a different name to return original g
        T_dwell_val = self._to_tensor(T_dwell, dtype=torch.float32).item()
        k0_tensor = self._to_tensor(k0, dtype=torch.float32)
        
        lg = g_tensor.shape[0]
        g_proc = g_tensor.unsqueeze(-1) if g_tensor.ndim == 1 else g_tensor
        
        if k0_tensor.numel() == 1 and g_proc.shape[1] > 1:
            k0_final = k0_tensor.repeat(g_proc.shape[1])
        else:
            k0_final = k0_tensor
        
        k = k0_final + torch.cumsum(g_proc, dim=0) * current_gamma_rad_g_s * T_dwell_val
        t_vec = torch.arange(lg, device=self.device, dtype=torch.float32) * T_dwell_val
        first_g_val = g_proc[0:1, :]
        s_calc = torch.diff(g_proc, dim=0, prepend=first_g_val) / T_dwell_val
        m0_phys = torch.cumsum(g_proc, dim=0) * T_dwell_val
        m1_phys = torch.cumsum(g_proc * t_vec.unsqueeze(-1), dim=0) * T_dwell_val
        m2_integrand = g_proc * (t_vec.unsqueeze(-1)**2 + (T_dwell_val**2 / 12.0))
        m2_phys = torch.cumsum(m2_integrand, dim=0) * T_dwell_val
        m0_phase = m0_phys * current_gamma_rad_g_s
        m1_phase = m1_phys * current_gamma_rad_g_s
        m2_phase = m2_phys * current_gamma_rad_g_s
        L_H = L_mH / 1000.0
        v = (1.0 / eta_inv_sec_per_volt_amp) * (L_H * s_calc + R_ohms * g_proc)
        if g_orig_ndim == 1 and k.ndim > 1 : # Check if k needs squeezing
            k = k.squeeze(-1)
            s_calc = s_calc.squeeze(-1)
            m0_phase = m0_phase.squeeze(-1)
            m1_phase = m1_phase.squeeze(-1)
            m2_phase = m2_phase.squeeze(-1)
            v = v.squeeze(-1)
        return k, g, s_calc, m0_phase, m1_phase, m2_phase, t_vec, v # Return original g

    def ksquare(self, center_coords, square_width, kloc_rad_per_cm, 
                tsamp_s=0.000004, df_hz=0.0):
        center_coords = self._to_tensor(center_coords, dtype=torch.complex64)
        square_width_val = self._to_tensor(square_width, dtype=torch.float32).item()
        kloc_rad_per_cm = self._to_tensor(kloc_rad_per_cm, dtype=torch.complex64)
        tsamp_s_val = self._to_tensor(tsamp_s, dtype=torch.float32).item()
        df_hz_val = self._to_tensor(df_hz, dtype=torch.float32).item()
        sinc_arg_x = square_width_val * torch.real(kloc_rad_per_cm) / (2 * torch.pi)
        sinc_arg_y = square_width_val * torch.imag(kloc_rad_per_cm) / (2 * torch.pi)
        sdata_shape = kloc_rad_per_cm.shape
        sdata_single_obj_at_origin = (square_width_val**2) * \
                                     torch.sinc(sinc_arg_x) * \
                                     torch.sinc(sinc_arg_y)
        sdata_single_obj_at_origin = sdata_single_obj_at_origin.to(torch.complex64)
        kdata = torch.zeros_like(sdata_single_obj_at_origin, dtype=torch.complex64)
        if center_coords.numel() > 0:
            for i in range(center_coords.numel()):
                center = center_coords.flatten()[i]
                phase_shift_exponent = 1j * (torch.real(center) * torch.real(kloc_rad_per_cm) + \
                                            torch.imag(center) * torch.imag(kloc_rad_per_cm))
                phase_shift = torch.exp(phase_shift_exponent)
                kdata += phase_shift * sdata_single_obj_at_origin
        if df_hz_val != 0.0:
            num_k_points = kloc_rad_per_cm.numel()
            t_points = tsamp_s_val * torch.arange(num_k_points, device=self.device, dtype=torch.float32)
            off_res_phase = torch.exp(1j * 2 * torch.pi * df_hz_val * t_points)
            kdata_flat = kdata.flatten() * off_res_phase
            kdata = kdata_flat.reshape(sdata_shape)
        return kdata

    def mingrad(self, kspace_area_cycles_cm, Gmax_G_cm=5.0, Smax_G_cm_ms=20.0, 
                dt_ms=0.004, gamma_kHz_G=None):
        current_gamma_kHz_G = self.gamma_kHz_G
        if gamma_kHz_G is not None:
            current_gamma_kHz_G = self._to_tensor(gamma_kHz_G).item()
        
        kspace_area_cycles_cm = self._to_tensor(kspace_area_cycles_cm, dtype=torch.float32).item()
        Gmax_G_cm = self._to_tensor(Gmax_G_cm, dtype=torch.float32).item()
        Smax_G_cm_ms = self._to_tensor(Smax_G_cm_ms, dtype=torch.float32).item()
        dt_ms = self._to_tensor(dt_ms, dtype=torch.float32).item()

        if abs(kspace_area_cycles_cm) < 1e-9 : # Check for effectively zero area
            return torch.empty(0, device=self.device, dtype=torch.float32), \
                   torch.empty(0, device=self.device, dtype=torch.float32)
        Atri_kspace_cycles_cm = (Gmax_G_cm**2 / Smax_G_cm_ms) * current_gamma_kHz_G
        g = torch.empty(0, device=self.device, dtype=torch.float32)
        if kspace_area_cycles_cm <= Atri_kspace_cycles_cm:
            t_ramp_ms = torch.sqrt(kspace_area_cycles_cm / (Smax_G_cm_ms * current_gamma_kHz_G)).item()
            N_ramp = torch.ceil(self._to_tensor(t_ramp_ms / dt_ms)).int().item()
            if N_ramp == 0: g_ramp_vals = torch.empty(0, device=self.device, dtype=torch.float32)
            elif N_ramp == 1: g_ramp_vals = torch.tensor([Smax_G_cm_ms * t_ramp_ms], device=self.device, dtype=torch.float32)
            else: g_ramp_vals = torch.linspace(0, Smax_G_cm_ms * t_ramp_ms, N_ramp, device=self.device)
            g_ramp_vals = torch.clamp(g_ramp_vals, max=Gmax_G_cm)
            if N_ramp == 0: g = torch.empty(0, device=self.device, dtype=torch.float32)
            elif N_ramp == 1: g = g_ramp_vals
            else: g = torch.cat((g_ramp_vals, torch.flip(g_ramp_vals[:-1], dims=[0])))
        else:
            t_ramp_ms = Gmax_G_cm / Smax_G_cm_ms
            N_ramp = torch.ceil(self._to_tensor(t_ramp_ms / dt_ms)).int().item()
            if N_ramp == 0: g_ramp_up = torch.empty(0, device=self.device, dtype=torch.float32)
            elif N_ramp == 1: g_ramp_up = torch.tensor([Gmax_G_cm], device=self.device, dtype=torch.float32)
            else: g_ramp_up = torch.linspace(0, Gmax_G_cm, N_ramp, device=self.device)
            k_area_ramps = (Gmax_G_cm / Smax_G_cm_ms) * Gmax_G_cm * current_gamma_kHz_G
            k_area_plateau = kspace_area_cycles_cm - k_area_ramps
            t_plateau_ms = k_area_plateau / (Gmax_G_cm * current_gamma_kHz_G)
            N_plateau = max(0, torch.ceil(self._to_tensor(t_plateau_ms / dt_ms)).int().item())
            g_plateau_vals = torch.ones(N_plateau, device=self.device, dtype=torch.float32) * Gmax_G_cm
            if N_ramp == 0: g = g_plateau_vals
            elif N_ramp == 1: g = torch.cat((g_ramp_up, g_plateau_vals, g_ramp_up)) # If ramp is one point, it's Gmax
            else: g = torch.cat((g_ramp_up, g_plateau_vals, torch.flip(g_ramp_up[1:-1], dims=[0]) if N_ramp > 2 else torch.empty(0, device=self.device), g_ramp_up[0:1] if N_ramp > 1 else torch.empty(0, device=self.device) )) # Corrected flip for trapezoid to avoid double Gmax/0
                                                                                                                                                                                                                                # A simpler flip: torch.flip(g_ramp_up, dims=[0]) if g_ramp_up includes 0 and Gmax.
                                                                                                                                                                                                                                # If g_ramp_up = [0, ..., Gmax], then flip is [Gmax, ..., 0].
                                                                                                                                                                                                                                # Corrected cat for trapezoid:
            if N_ramp == 0: g = g_plateau_vals
            elif N_ramp == 1: g = torch.cat((g_ramp_up, g_plateau_vals, g_ramp_up)) # g_ramp_up is [Gmax]
            else: g = torch.cat((g_ramp_up, g_plateau_vals, torch.flip(g_ramp_up, dims=[0])))


        if g.numel() > 0:
            current_k_area = torch.sum(g) * dt_ms * current_gamma_kHz_G
            if torch.abs(current_k_area) > 1e-9:
                g = g * (kspace_area_cycles_cm / current_k_area)
        t_vec_ms = torch.arange(g.numel(), device=self.device, dtype=torch.float32) * dt_ms
        return g, t_vec_ms

    def rotate_spirals(self, g_spiral_complex, k_spiral_complex, N_interleaves):
        g_spiral_complex = self._to_tensor(g_spiral_complex, dtype=torch.complex64)
        k_spiral_complex = self._to_tensor(k_spiral_complex, dtype=torch.complex64)
        if g_spiral_complex.ndim != 1 or k_spiral_complex.ndim != 1:
            raise ValueError("Input spiral g and k must be 1D complex tensors.")
        srot_angles = torch.arange(N_interleaves, device=self.device, dtype=torch.float32) * (2 * torch.pi / N_interleaves)
        srot = torch.exp(1j * srot_angles).to(torch.complex64)
        k_rot = k_spiral_complex.unsqueeze(-1) * srot.unsqueeze(0)
        g_rot = g_spiral_complex.unsqueeze(-1) * srot.unsqueeze(0)
        return g_rot, k_rot

    def _cumint(self, g_data, dt):
        g_data = self._to_tensor(g_data)
        dt_val = self._to_tensor(dt).item()
        return torch.cumsum(g_data, dim=0) * dt_val

    def vecdcf(self, g_waveform_G_cm, dt_s, k_traj_rad_cm=None, 
               N_interleaves=None, FOV_cm=None, res_mm=None, epsilon=1e-9):
        g_waveform_G_cm_tensor = self._to_tensor(g_waveform_G_cm)
        dt_s_val = self._to_tensor(dt_s).item()
        epsilon_val = self._to_tensor(epsilon).item()

        if g_waveform_G_cm_tensor.is_complex():
            if g_waveform_G_cm_tensor.ndim != 1: raise ValueError("Complex g_waveform_G_cm must be 1D.")
            g_xy = torch.stack((torch.real(g_waveform_G_cm_tensor), torch.imag(g_waveform_G_cm_tensor)), dim=1)
        elif g_waveform_G_cm_tensor.ndim == 2 and g_waveform_G_cm_tensor.shape[1] == 2:
            g_xy = g_waveform_G_cm_tensor.float()
        elif g_waveform_G_cm_tensor.ndim == 1:
            g_xy = torch.stack((g_waveform_G_cm_tensor, torch.zeros_like(g_waveform_G_cm_tensor)), dim=1).float()
        else: raise ValueError("g_waveform_G_cm format error.")

        if k_traj_rad_cm is None:
            kx = self._cumint(g_xy[:, 0], dt_s_val) * self.gamma_rad_g_s
            ky = self._cumint(g_xy[:, 1], dt_s_val) * self.gamma_rad_g_s
            k_xy = torch.stack((kx, ky), dim=1)
            k_traj_calculated_complex = (kx + 1j * ky).to(torch.complex64) 
        else:
            k_traj_rad_cm_tensor = self._to_tensor(k_traj_rad_cm)
            if k_traj_rad_cm_tensor.is_complex():
                if k_traj_rad_cm_tensor.ndim != 1: raise ValueError("Complex k_traj_rad_cm must be 1D.")
                k_xy = torch.stack((torch.real(k_traj_rad_cm_tensor), torch.imag(k_traj_rad_cm_tensor)), dim=1)
                k_traj_calculated_complex = k_traj_rad_cm_tensor
            elif k_traj_rad_cm_tensor.ndim == 2 and k_traj_rad_cm_tensor.shape[1] == 2:
                k_xy = k_traj_rad_cm_tensor.float()
                k_traj_calculated_complex = (k_xy[:,0] + 1j * k_xy[:,1]).to(torch.complex64)
            elif k_traj_rad_cm_tensor.ndim == 1:
                k_xy = torch.stack((k_traj_rad_cm_tensor, torch.zeros_like(k_traj_rad_cm_tensor)), dim=1).float()
                k_traj_calculated_complex = (k_xy[:,0] + 1j * k_xy[:,1]).to(torch.complex64)
            else: raise ValueError("k_traj_rad_cm format error.")
        if g_xy.shape[0] != k_xy.shape[0]: raise ValueError("Mismatch in g_xy and k_xy samples.")

        Nsamps = k_xy.shape[0]
        g3 = torch.cat((g_xy, torch.zeros((Nsamps, 1), device=self.device, dtype=g_xy.dtype)), dim=1)
        k3 = torch.cat((k_xy, torch.zeros((Nsamps, 1), device=self.device, dtype=k_xy.dtype)), dim=1)
        dcf = torch.zeros(Nsamps, device=self.device, dtype=torch.float32)
        for m in range(Nsamps):
            cross_prod = torch.cross(g3[m, :], k3[m, :])
            norm_k_m = torch.linalg.norm(k_xy[m, :])
            dcf[m] = torch.linalg.norm(cross_prod) / (norm_k_m + epsilon_val)
        if N_interleaves is not None and N_interleaves > 0:
            dcf_all = dcf.unsqueeze(-1).repeat(1, N_interleaves)
            srot_angles = torch.arange(N_interleaves, device=self.device,dtype=torch.float32) * (2*torch.pi/N_interleaves)
            srot = torch.exp(1j * srot_angles).to(torch.complex64)
            k_traj_out_complex = k_traj_calculated_complex.unsqueeze(-1) * srot.unsqueeze(0)
            return dcf_all, k_traj_out_complex
        else:
            return dcf, k_traj_calculated_complex

    # --- Kaiser-Bessel gridding methods ---
    def _kb(self, u, w, beta):
        u = self._to_tensor(u, dtype=torch.float32) # Ensure u is a tensor on the correct device
        w = self._to_tensor(w, dtype=torch.float32).item()
        beta = self._to_tensor(beta, dtype=torch.float32).item()
        
        y = torch.zeros_like(u, dtype=torch.float32, device=self.device)
        uz_mask = torch.abs(u) < (w / 2.0)
        
        if torch.any(uz_mask):
            u_masked = u[uz_mask]
            sqrt_arg = 1.0 - (2.0 * u_masked / w)**2
            x_beta_term = torch.sqrt(torch.clamp(sqrt_arg, min=0.0))
            # torch.i0 requires PyTorch 1.8+
            try:
                y[uz_mask] = torch.i0(beta * x_beta_term) / w
            except AttributeError:
                raise RuntimeError("torch.i0 not found. Please ensure PyTorch version is 1.8 or later.")
        return y

    def gridmat(self, k_traj_grid_units, k_data_samples, dcf_weights, 
                grid_size=256, kernel_width_grid_units=3.0, kb_beta=4.2, pad_factor=2):
        
        k_traj_grid_units = self._to_tensor(k_traj_grid_units) # Infer dtype
        k_data_samples = self._to_tensor(k_data_samples, dtype=torch.complex64)
        dcf_weights = self._to_tensor(dcf_weights) # Infer dtype, then cast

        grid_size = int(grid_size)
        kernel_width_grid_units = float(kernel_width_grid_units)
        kb_beta = float(kb_beta)
        pad_factor = int(pad_factor)

        padded_grid_size = int(grid_size * pad_factor)
        padgrid = torch.zeros((padded_grid_size, padded_grid_size), dtype=torch.complex64, device=self.device)
        
        k_data_corrected = k_data_samples * dcf_weights.to(k_data_samples.dtype) # Cast dcf to data's dtype

        if k_traj_grid_units.is_complex():
            if k_traj_grid_units.ndim != 1: raise ValueError("Complex k_traj_grid_units must be 1D.")
            k_coords_x = torch.real(k_traj_grid_units)
            k_coords_y = torch.imag(k_traj_grid_units)
        elif k_traj_grid_units.ndim == 2 and k_traj_grid_units.shape[1] == 2:
            k_coords_x = k_traj_grid_units[:, 0].float()
            k_coords_y = k_traj_grid_units[:, 1].float()
        else:
            raise ValueError("k_traj_grid_units must be complex 1D or real 2D (Npts, 2).")

        if not (k_coords_x.shape == k_coords_y.shape == k_data_samples.shape == dcf_weights.shape and k_coords_x.ndim == 1):
            raise ValueError("k_coords_x/y, k_data_samples, and dcf_weights must be 1D and have the same number of elements.")

        k_center_x_on_padgrid = k_coords_x + padded_grid_size / 2.0
        k_center_y_on_padgrid = k_coords_y + padded_grid_size / 2.0
        
        sgrid_kernel_radius = kernel_width_grid_units / 2.0
        # sgrid_pts_extent = torch.ceil(self._to_tensor(sgrid_kernel_radius)).int().item() # Not directly used in loop bounds in this style

        for p in range(k_data_samples.shape[0]):
            kx_sample_center_padded = k_center_x_on_padgrid[p]
            ky_sample_center_padded = k_center_y_on_padgrid[p]
            
            min_gx = torch.floor(kx_sample_center_padded - sgrid_kernel_radius).int().item()
            max_gx = torch.ceil(kx_sample_center_padded + sgrid_kernel_radius).int().item()
            min_gy = torch.floor(ky_sample_center_padded - sgrid_kernel_radius).int().item()
            max_gy = torch.ceil(ky_sample_center_padded + sgrid_kernel_radius).int().item()

            for gy_idx in range(min_gy, max_gy + 1): # Inclusive range
                for gx_idx in range(min_gx, max_gx + 1): # Inclusive range
                    if 0 <= gx_idx < padded_grid_size and 0 <= gy_idx < padded_grid_size:
                        dist_to_sample_center = torch.sqrt(
                            (gx_idx - kx_sample_center_padded)**2 + \
                            (gy_idx - ky_sample_center_padded)**2
                        )
                        if dist_to_sample_center < sgrid_kernel_radius:
                            kernel_val = self._kb(dist_to_sample_center.unsqueeze(0), 
                                                  kernel_width_grid_units, 
                                                  kb_beta)[0] # Get scalar from tensor
                            padgrid[gy_idx, gx_idx] += k_data_corrected[p] * kernel_val
        
        start_idx = (padded_grid_size - grid_size) // 2
        gridded_data = padgrid[start_idx : start_idx + grid_size, start_idx : start_idx + grid_size]
        return gridded_data


if __name__ == '__main__':
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {test_device} for tests ---")
    tools = GradientKSpaceTools(device=test_device)

    # Shortened existing tests
    print("\n--- Testing time2freq ---"); tools.time2freq(torch.linspace(0,1e-2,100,device=test_device)); print("OK.")
    print("\n--- Testing bvalue ---"); tools.bvalue(torch.ones(100,device=test_device),4e-6); print("OK.")
    print("\n--- Testing calcgradinfo ---"); tools.calcgradinfo(torch.sin(torch.linspace(0,2*torch.pi,100,device=test_device))); print("OK.")
    print("\n--- Testing ksquare ---"); tools.ksquare(torch.tensor([0j],device=test_device),0.05,torch.rand(64,64,dtype=torch.complex64,device=test_device)); print("OK.")
    print("\n--- Testing mingrad ---"); tools.mingrad(0.5); tools.mingrad(5.0); print("OK.")
    print("\n--- Testing rotate_spirals ---"); tools.rotate_spirals(torch.rand(100,dtype=torch.complex64,device=test_device),torch.rand(100,dtype=torch.complex64,device=test_device),4); print("OK.")
    print("\n--- Testing _cumint ---"); tools._cumint(torch.ones(5,device=test_device),0.1); print("OK.")
    print("\n--- Testing vecdcf ---"); tools.vecdcf(torch.linspace(0,1,100,device=test_device),4e-6); print("OK.")

    # --- New tests for _kb and gridmat ---
    print("\n--- Testing _kb ---")
    u_test_kb = torch.linspace(-2.0, 2.0, 5, device=test_device) # u values
    w_kb = 3.0 # kernel width
    beta_kb = 4.2 
    kb_vals = tools._kb(u_test_kb, w_kb, beta_kb)
    print(f"_kb inputs u={u_test_kb.cpu().numpy()}, w={w_kb}, beta={beta_kb}")
    print(f"_kb output: {kb_vals.cpu().numpy()}")
    assert kb_vals.shape == u_test_kb.shape, "_kb shape mismatch"
    # Check that values outside w/2 are zero, inside are positive
    assert kb_vals[0] == 0 and kb_vals[-1] == 0, "_kb boundary values incorrect"
    assert torch.all(kb_vals[1:-1] > 0), "_kb internal values should be positive"
    print("_kb test passed.")

    print("\n--- Testing gridmat ---")
    grid_s = 64
    k_pts = 500
    # Random k-space trajectory in grid units [-grid_s/2, grid_s/2]
    k_traj_gm = (torch.rand(k_pts, 2, device=test_device) - 0.5) * grid_s 
    k_data_gm = torch.rand(k_pts, dtype=torch.complex64, device=test_device)
    dcf_gm = torch.rand(k_pts, dtype=torch.float32, device=test_device) + 0.5 # Ensure positive

    gridded_ksp = tools.gridmat(k_traj_gm, k_data_gm, dcf_gm, grid_size=grid_s, kernel_width_grid_units=3.0)
    print(f"gridmat output shape: {gridded_ksp.shape}")
    assert gridded_ksp.shape == (grid_s, grid_s), "gridmat output shape mismatch"
    assert gridded_ksp.is_complex(), "gridded_ksp should be complex"
    
    # Test with complex k_traj input
    k_traj_complex_gm = ( (torch.rand(k_pts, device=test_device) - 0.5) * grid_s + \
                       1j*(torch.rand(k_pts, device=test_device) - 0.5) * grid_s ).to(torch.complex64)
    gridded_ksp_complex_k = tools.gridmat(k_traj_complex_gm, k_data_gm, dcf_gm, grid_size=grid_s)
    assert gridded_ksp_complex_k.shape == (grid_s, grid_s), "gridmat complex k-traj shape mismatch"
    print("gridmat basic tests passed.")

    # Test edge case: one k-space point at center
    k_traj_center = torch.tensor([[0.0, 0.0]], device=test_device)
    k_data_center = torch.tensor([1.0+0j], dtype=torch.complex64, device=test_device)
    dcf_center = torch.tensor([1.0], dtype=torch.float32, device=test_device)
    gridded_center = tools.gridmat(k_traj_center, k_data_center, dcf_center, grid_size=grid_s, kernel_width_grid_units=3.0)
    # The peak should be near the center of the grid
    center_idx = grid_s // 2
    # Sum of values in a small central region should be significant
    assert torch.sum(torch.abs(gridded_center[center_idx-2:center_idx+2, center_idx-2:center_idx+2])) > 0.1, "Gridding single center point failed"
    print("gridmat single center point test passed.")

    print("\nAll tests for GradientKSpaceTools (including final methods) seem to run.")
