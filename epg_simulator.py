import torch
import numpy as np # For consistent testing with original if needed

class EPGSimulator:
    def __init__(self, initial_state=None, device=None):
        """
        Initializes the EPG simulator.
        If initial_state is provided, set self.FZ = initial_state.
        Otherwise, set self.FZ = self.epg_m0().
        Ensures self.FZ is a PyTorch tensor of torch.complex64.
        Device can be 'cpu' or 'cuda' etc.
        """
        self.device = device if device is not None else (initial_state.device if isinstance(initial_state, torch.Tensor) else torch.device('cpu'))

        if initial_state is not None:
            if not isinstance(initial_state, torch.Tensor):
                raise TypeError("initial_state must be a PyTorch tensor.")
            self.FZ = initial_state.to(device=self.device, dtype=torch.complex64)
        else:
            self.FZ = self.epg_m0() # epg_m0 will use self.device

    def epg_m0(self):
        """
        Returns the equilibrium magnetization state as a PyTorch tensor.
        FZ state vector: [F0, F0*, Z0] = [[0], [0], [1]]
        """
        return torch.tensor([[0+0j], [0+0j], [1+0j]], dtype=torch.complex64, device=self.device)

    def _to_float_tensor(self, value):
        """Helper to convert value to a scalar-like float32 tensor on the correct device."""
        if not isinstance(value, torch.Tensor):
            return torch.tensor(float(value), dtype=torch.float32, device=self.device)
        val = value.float().to(self.device)
        if val.numel() > 1:
            return val[0] 
        return val

    def epg_rf(self, alpha_deg, phi_deg):
        alpha_deg_t = self._to_float_tensor(alpha_deg)
        phi_deg_t = self._to_float_tensor(phi_deg)
        alpha_rad = alpha_deg_t * torch.pi / 180.0
        phi_rad = phi_deg_t * torch.pi / 180.0
        ca2 = torch.cos(alpha_rad / 2.0)**2
        sa2 = torch.sin(alpha_rad / 2.0)**2
        sa = torch.sin(alpha_rad)
        exp_jphi = torch.exp(1j * phi_rad)
        exp_2jphi = torch.exp(2j * phi_rad)
        ca2 = ca2.cfloat()
        sa2 = sa2.cfloat()
        sa = sa.cfloat()
        exp_jphi = exp_jphi.cfloat()
        exp_2jphi = exp_2jphi.cfloat()
        RR = torch.tensor([
            [ca2, exp_2jphi * sa2, -1j * exp_jphi * sa],
            [torch.conj(exp_2jphi) * sa2, ca2, 1j * torch.conj(exp_jphi) * sa],
            [-0.5j * torch.conj(exp_jphi) * sa, 0.5j * exp_jphi * sa, torch.cos(alpha_rad).cfloat()]
        ], dtype=torch.complex64, device=self.device)
        self.FZ = torch.matmul(RR, self.FZ)
        return self.FZ

    def epg_relax(self, T1, T2, T):
        T1_t = self._to_float_tensor(T1)
        T2_t = self._to_float_tensor(T2)
        T_t = self._to_float_tensor(T)
        E1 = torch.exp(-T_t / T1_t)
        E2 = torch.exp(-T_t / T2_t)
        EE = torch.diag(torch.stack([E2.cfloat(), E2.cfloat(), E1.cfloat()])).to(self.device)
        RR_recovery = (1.0 - E1).cfloat()
        self.FZ = torch.matmul(EE, self.FZ)
        self.FZ[2, 0] = self.FZ[2, 0] + RR_recovery
        return self.FZ

    def epg_grad(self, noadd=False, positive_lobe=True):
        if not noadd:
            self.FZ = torch.cat((self.FZ, torch.zeros((3, 1), dtype=torch.complex64, device=self.device)), dim=1)
        num_states = self.FZ.shape[1]
        if num_states == 1 and noadd: # Only Z0 state
            if positive_lobe: # F0+ = (F0-)*
                self.FZ[0, 0] = torch.conj(self.FZ[1, 0])
            else: # F0- = (F0+)*
                self.FZ[1, 0] = torch.conj(self.FZ[0, 0])
            return self.FZ
        if positive_lobe:
            self.FZ[0, 1:] = self.FZ[0, :-1].clone()
            self.FZ[1, :-1] = self.FZ[1, 1:].clone()
            self.FZ[1, -1] = 0j 
            self.FZ[0, 0] = torch.conj(self.FZ[1, 0])
        else: # negative_lobe
            self.FZ[1, 1:] = self.FZ[1, :-1].clone()
            self.FZ[0, :-1] = self.FZ[0, 1:].clone()
            self.FZ[0, -1] = 0j
            self.FZ[1, 0] = torch.conj(self.FZ[0, 0])
        return self.FZ

    def epg_zrot(self, angle_deg):
        angle_deg_t = self._to_float_tensor(angle_deg)
        angle_rad = angle_deg_t * torch.pi / 180.0
        phasor = torch.exp(1j * angle_rad).cfloat()
        RR_elementwise = torch.diag(torch.tensor([phasor, torch.conj(phasor), 1+0j], dtype=torch.complex64, device=self.device))
        self.FZ = torch.matmul(RR_elementwise, self.FZ)
        return self.FZ

    def epg_trim(self, threshold=1e-5):
        if self.FZ.shape[1] <= 1:
            return self.FZ
        abs_FZ = torch.abs(self.FZ)
        f_plus_max_order = 0
        significant_f_plus = torch.where(abs_FZ[0, 1:] > threshold)[0]
        if significant_f_plus.numel() > 0:
            f_plus_max_order = torch.max(significant_f_plus) + 1
        f_minus_max_order = 0
        significant_f_minus = torch.where(abs_FZ[1, 1:] > threshold)[0]
        if significant_f_minus.numel() > 0:
            f_minus_max_order = torch.max(significant_f_minus) + 1
        z_k_max_order = 0
        significant_z_k = torch.where(abs_FZ[2, 1:] > threshold)[0]
        if significant_z_k.numel() > 0:
            z_k_max_order = torch.max(significant_z_k) + 1
        max_order_idx = max(f_plus_max_order, f_minus_max_order, z_k_max_order)
        max_order_idx = max(0, max_order_idx)
        self.FZ = self.FZ[:, :max_order_idx + 1]
        return self.FZ

    def epg_mgrad(self, noadd=False):
        return self.epg_grad(noadd=noadd, positive_lobe=False)

    def epg_grelax(self, T1, T2, T, kg=0, D=0, Gon=True, noadd=True):
        self.epg_relax(T1, T2, T)
        if Gon:
            self.epg_grad(noadd=noadd, positive_lobe=True)
        return self.FZ

    def epg_spinlocs(self, nspins=9):
        z = (torch.arange(0, nspins, dtype=torch.float32, device=self.device) - 
             torch.floor(torch.tensor(nspins / 2.0, device=self.device))) / nspins
        return z

    def epg_spins2FZ(self, M, trim_threshold=0.01):
        M = M.to(device=self.device, dtype=torch.complex64)
        N_spins = M.shape[1]
        max_order_Q = int(torch.floor(torch.tensor(N_spins / 2.0)).item()) + 1
        M_shifted = torch.fft.ifftshift(M, dim=1)
        Mxy = M_shifted[0, :] + 1j * M_shifted[1, :]
        Fp = torch.fft.fft(Mxy) / N_spins
        Fm = torch.fft.fft(torch.conj(Mxy)) / N_spins 
        Mz_shifted_complex = M_shifted[2, :].to(torch.complex64)
        Z = torch.fft.fft(Mz_shifted_complex) / N_spins
        FpFmZ_full = torch.stack((Fp[:max_order_Q], Fm[:max_order_Q], Z[:max_order_Q]), dim=0)
        current_FZ_backup = self.FZ
        self.FZ = FpFmZ_full.to(self.device)
        trimmed_FZ = self.epg_trim(threshold=trim_threshold)
        self.FZ = current_FZ_backup
        return trimmed_FZ

    def epg_FZ2spins(self, N_spins=None, frac=0):
        num_states_k = self.FZ.shape[1]
        max_k_order = num_states_k - 1
        if N_spins is None:
            N_spins = 2 * max_k_order + 1
            if N_spins < 1: N_spins = 1
        z_locs = self.epg_spinlocs(N_spins).unsqueeze(0)
        state_indices_k = (torch.arange(-max_k_order, max_k_order + 1, dtype=torch.float32, device=self.device) + frac).unsqueeze(1)
        fourier_matrix = torch.exp(1j * 2.0 * torch.pi * torch.matmul(state_indices_k, z_locs)).to(torch.complex64)
        F_plus_k = self.FZ[0, :max_k_order+1]
        if max_k_order > 0:
            F_minus_k_conj_flipped = torch.flip(torch.conj(self.FZ[1, 1:max_k_order+1]), dims=[0])
            F_all_states = torch.cat((F_minus_k_conj_flipped, F_plus_k), dim=0)
        else:
            F_all_states = F_plus_k
        Mxy = torch.matmul(F_all_states.unsqueeze(0), fourier_matrix)
        Mz_fourier_matrix_rows = fourier_matrix[max_k_order:(2*max_k_order + 1), :]
        Z_coeffs = self.FZ[2, :max_k_order+1]
        Mz = torch.matmul(Z_coeffs.unsqueeze(0), Mz_fourier_matrix_rows)
        Mz = 2.0 * torch.real(Mz)
        if Z_coeffs.numel() > 0 :
             Mz = Mz - torch.real(Z_coeffs[0] * Mz_fourier_matrix_rows[0,:])
        M = torch.cat((torch.real(Mxy), torch.imag(Mxy), Mz.real()), dim=0)
        return M.to(self.device)

    # --- Final methods start here ---

    def show_matrix(self, label=""):
        """ Prints the current self.FZ states. """
        if label:
            print(label)
        # For compact display, convert to numpy and use its formatter.
        # Otherwise, PyTorch tensor printing can be verbose.
        FZ_np = self.FZ.detach().cpu().numpy() 
        # Basic print, can be customized further if complex formatting needed
        # print(FZ_np) # Default numpy print
        # More controlled print:
        for row in FZ_np:
            formatted_row = [f"{x.real:+.2f}{x.imag:+.2f}j" for x in row]
            print(f"[{' '.join(formatted_row)}]")

    def epg_cpmg(self, etl, T1, T2, esp, flipangle_deg_series, initial_rf_phase_deg=90.0, initial_rf_flip_deg=90.0):
        """ Simulates a CPMG pulse sequence. """
        self.FZ = self.epg_m0()
        self.epg_rf(alpha_deg=initial_rf_flip_deg, phi_deg=initial_rf_phase_deg)
        
        echo_signals = torch.zeros(etl, dtype=torch.complex64, device=self.device)
        
        if not isinstance(flipangle_deg_series, torch.Tensor):
            flipangle_deg_series = torch.tensor(flipangle_deg_series, device=self.device)

        for i in range(etl):
            self.epg_grelax(T1, T2, esp / 2.0, Gon=True, noadd=False)
            
            current_refocus_flip_angle = 0.0
            if flipangle_deg_series.numel() == 1:
                current_refocus_flip_angle = flipangle_deg_series.item()
            else:
                current_refocus_flip_angle = flipangle_deg_series[i].item()
            
            self.epg_rf(alpha_deg=current_refocus_flip_angle, phi_deg=0.0) # Refocusing pulses phase = 0
            
            self.epg_grelax(T1, T2, esp / 2.0, Gon=True, noadd=False)
            
            echo_signals[i] = self.FZ[0, 0].clone() # Store F0 state
            
        return echo_signals

    def epg_gradecho(self, T1, T2, TR, TE, flipangle_deg, rf_phase_cycle_deg_initial=0.0, rf_phase_increment_deg=0.0, gspoil_flag=0, delta_freq_hz=0.0, num_reps=200):
        """ Simulates a generic gradient-echo pulse sequence. """
        self.FZ = self.epg_m0()
        current_rf_phase_deg = self._to_float_tensor(rf_phase_cycle_deg_initial) # Ensure tensor for calculations
        
        signals = torch.zeros(num_reps, dtype=torch.complex64, device=self.device)

        for rep_idx in range(num_reps):
            # Relaxation and off-resonance from TE to TR (end of previous TR to start of current RF)
            self.epg_grelax(T1, T2, TR - TE, Gon=False) # No gradient during this period
            self.epg_zrot(360.0 * delta_freq_hz * (TR - TE) / 1000.0) # Time in ms for TR-TE

            if gspoil_flag == 1: # FISP-like: spoiler after TE (applied before next RF effectively)
                self.epg_grad(noadd=False) 
            
            if gspoil_flag == 100: # Perfect spoiling before RF
                if self.FZ.shape[1] > 0: # Check if FZ is not empty
                    self.FZ = self.FZ[:,0:1].clone() # Keep only Z0 state
                    self.FZ[0,0] = 0j # Zero out F0+
                    self.FZ[1,0] = 0j # Zero out F0-
                else: # Handle case where FZ might be empty after excessive trimming
                    self.FZ = self.epg_m0() # Reset to equilibrium if empty
                    self.FZ = self.FZ[:,0:1].clone()
                    self.FZ[0,0] = 0j
                    self.FZ[1,0] = 0j


            rf_phase_for_demod_rad = current_rf_phase_deg * torch.pi / 180.0
            
            # Apply RF pulse (original code added +90 to phase, maintaining for consistency)
            self.epg_rf(alpha_deg=flipangle_deg, phi_deg=(current_rf_phase_deg + 90.0).item())
            
            current_rf_phase_deg = current_rf_phase_deg + rf_phase_increment_deg

            # Relaxation and off-resonance from RF to TE
            self.epg_grelax(T1, T2, TE, Gon=False) # No gradient during this period
            self.epg_zrot(360.0 * delta_freq_hz * TE / 1000.0) # Time in ms for TE

            if gspoil_flag == -1: # PSIF-like: spoiler before RF (applied after RF and TE effectively)
                self.epg_grad(noadd=False)
            
            if gspoil_flag == 100: # Perfect spoiling after RF (crush F+, F- states, only F0 remains)
                self.epg_grad(noadd=True) # noadd=True ensures only F0 from F0*, F0- from F0* are affected.

            signal_val = self.FZ[0, 0] * torch.exp(1j * rf_phase_for_demod_rad)
            signals[rep_idx] = signal_val.clone()
            
        return signals

    def epg_stim_calc(self, flipangle_deg_series, initial_rf_phase_deg=90.0):
        """ Calculates stimulated echo based on a series of RF pulses. """
        self.FZ = self.epg_m0()
        
        if not isinstance(flipangle_deg_series, torch.Tensor):
            flipangle_deg_series = torch.tensor(flipangle_deg_series, device=self.device, dtype=torch.float32)

        for i in range(flipangle_deg_series.numel()):
            current_flip_angle = flipangle_deg_series[i].item()
            self.epg_rf(alpha_deg=current_flip_angle, phi_deg=initial_rf_phase_deg)
            # Using T1=1.0, T2=0.2, T=0 as per original mrsigpy example for this specific function
            # These times are very short, T=0 means no relaxation decay, only gradient effect
            self.epg_grelax(T1=1.0, T2=0.2, T=0.0, Gon=True, noadd=False) 
            
        S = self.FZ[0, 0].clone() # F0 state is the signal
        return S, self.FZ.clone()


if __name__ == '__main__':
    # Default device for tests
    test_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Using device: {test_device} for tests ---")

    sim = EPGSimulator(device=test_device)

    print("\n--- Testing show_matrix ---")
    sim.FZ = sim.epg_m0()
    sim.epg_rf(alpha_deg=90, phi_deg=0)
    sim.show_matrix(label="FZ after 90deg RF:")
    sim.epg_grad(noadd=False)
    sim.show_matrix(label="FZ after gradient:")

    print("\n--- Testing epg_cpmg ---")
    etl_cpmg = 10
    T1_cpmg, T2_cpmg, esp_cpmg = 1000.0, 100.0, 10.0 # ms
    # flip_series_cpmg = torch.ones(etl_cpmg) * 180.0 # All 180s
    flip_series_cpmg = [180.0] * etl_cpmg # More pythonic for list of scalars
    cpmg_echoes = sim.epg_cpmg(etl=etl_cpmg, T1=T1_cpmg, T2=T2_cpmg, esp=esp_cpmg, 
                               flipangle_deg_series=flip_series_cpmg)
    print(f"CPMG Echoes (first 5 of {cpmg_echoes.numel()}) mag: ", torch.abs(cpmg_echoes[:5]))
    assert cpmg_echoes.numel() == etl_cpmg, "CPMG: Incorrect number of echoes"
    assert cpmg_echoes.device == test_device, "CPMG: Device mismatch"


    print("\n--- Testing epg_gradecho ---")
    T1_ge, T2_ge, TR_ge, TE_ge = 1000.0, 100.0, 20.0, 5.0 # ms
    flip_ge = 15.0 # degrees
    num_reps_ge = 50
    
    # Test case 1: Basic GRE (no spoil, no off-res)
    gre_signals_basic = sim.epg_gradecho(T1=T1_ge, T2=T2_ge, TR=TR_ge, TE=TE_ge, 
                                         flipangle_deg=flip_ge, num_reps=num_reps_ge)
    print(f"GRE Basic (first 5 of {gre_signals_basic.numel()}) mag: ", torch.abs(gre_signals_basic[:5]))
    assert gre_signals_basic.numel() == num_reps_ge, "GRE Basic: Incorrect number of signals"
    
    # Test case 2: GRE with RF spoiling and off-resonance
    gre_signals_rfspoil = sim.epg_gradecho(T1=T1_ge, T2=T2_ge, TR=TR_ge, TE=TE_ge, 
                                           flipangle_deg=flip_ge, 
                                           rf_phase_increment_deg=117.0, 
                                           delta_freq_hz=50.0,
                                           num_reps=num_reps_ge)
    print(f"GRE RF Spoiled (first 5 of {gre_signals_rfspoil.numel()}) mag: ", torch.abs(gre_signals_rfspoil[:5]))
    assert gre_signals_rfspoil.numel() == num_reps_ge, "GRE RF Spoiled: Incorrect number of signals"

    # Test case 3: GRE with perfect spoiling (gspoil_flag=100)
    gre_signals_perfect_spoil = sim.epg_gradecho(T1=T1_ge, T2=T2_ge, TR=TR_ge, TE=TE_ge, 
                                                 flipangle_deg=flip_ge, 
                                                 gspoil_flag=100,
                                                 num_reps=num_reps_ge)
    print(f"GRE Perfect Spoiled (first 5 of {gre_signals_perfect_spoil.numel()}) mag: ", torch.abs(gre_signals_perfect_spoil[:5]))
    assert gre_signals_perfect_spoil.numel() == num_reps_ge, "GRE Perfect Spoiled: Incorrect number of signals"
    assert gre_signals_perfect_spoil.device == test_device, "GRE Perfect Spoiled: Device mismatch"


    print("\n--- Testing epg_stim_calc ---")
    # flip_series_stim = torch.tensor([90.0, 90.0, 90.0], device=test_device)
    flip_series_stim = [90.0, 45.0, 30.0]
    stim_signal, stim_FZ_final = sim.epg_stim_calc(flipangle_deg_series=flip_series_stim)
    print(f"Stimulated Echo Signal: {stim_signal} (mag: {torch.abs(stim_signal)})")
    sim.FZ = stim_FZ_final # To show the final state
    sim.show_matrix(label="Final FZ after STIM calc:")
    assert stim_signal.device == test_device, "STIM: Device mismatch on signal"
    assert stim_FZ_final.device == test_device, "STIM: Device mismatch on FZ_final"

    print("\nAll tests for final methods seem to run. Check values for correctness based on theory/other sims.")
    print("\nScript finished.")
