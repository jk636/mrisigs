import torch

class EPGXSimulator:
    def __init__(self, device, model_type='full'):
        self.device = device
        self.model_type = model_type
        self.FZ = None

    def _to_float_tensor(self, value):
        """Converts a value to a float tensor on the specified device."""
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32, device=self.device)
        elif value.device != self.device:
            value = value.to(device=self.device, dtype=torch.float32)
        else:
            value = value.float() # Ensure it's float32
        return value

    def epgx_m0(self, f):
        f_tensor = self._to_float_tensor(f)
        if self.model_type == 'MT':
            self.FZ = torch.tensor([[0], [0], [1-f_tensor], [f_tensor]], dtype=torch.complex64, device=self.device)
        elif self.model_type == 'full':
            self.FZ = torch.tensor([[0], [0], [1-f_tensor], [0], [0], [f_tensor]], dtype=torch.complex64, device=self.device)
        return self.FZ

    def epgx_rf(self, alpha_deg, phi_deg, compartment='both', WT=None):
        alpha_rad = torch.deg2rad(self._to_float_tensor(alpha_deg))
        phi_rad = torch.deg2rad(self._to_float_tensor(phi_deg))

        # Pre-calculate trigonometric terms
        ca2 = torch.cos(alpha_rad / 2)**2
        sa2 = torch.sin(alpha_rad / 2)**2
        sa = torch.sin(alpha_rad)
        exp_j_phi = torch.exp(1j * phi_rad)
        exp_2j_phi = torch.exp(2j * phi_rad)

        # RF rotation matrix (complex64)
        RR = torch.zeros((3, 3), dtype=torch.complex64, device=self.device)
        RR[0, 0] = ca2
        RR[0, 1] = exp_2j_phi * sa2
        RR[0, 2] = -1j * exp_j_phi * sa
        RR[1, 0] = exp_2j_phi.conj() * sa2
        RR[1, 1] = ca2
        RR[1, 2] = 1j * exp_j_phi.conj() * sa
        RR[2, 0] = -1j / 2 * exp_j_phi.conj() * sa
        RR[2, 1] = 1j / 2 * exp_j_phi * sa
        RR[2, 2] = torch.cos(alpha_rad)
        
        if self.model_type == 'full':
            FZ_a = self.FZ[0:3, :]
            FZ_b = self.FZ[3:6, :]
            if compartment == 'a' or compartment == 'both':
                FZ_a = torch.matmul(RR, FZ_a)
            if compartment == 'b' or compartment == 'both':
                FZ_b = torch.matmul(RR, FZ_b)
            self.FZ = torch.cat((FZ_a, FZ_b), dim=0)
        elif self.model_type == 'MT':
            FZ_a_transverse_long = self.FZ[0:3, :]
            FZ_b_long = self.FZ[3:4, :]
            
            FZ_a_transverse_long = torch.matmul(RR, FZ_a_transverse_long)
            
            if WT is not None:
                WT_tensor = self._to_float_tensor(WT)
                FZ_b_long = torch.exp(-WT_tensor) * FZ_b_long
                
            self.FZ = torch.cat((FZ_a_transverse_long, FZ_b_long), dim=0)
        return self.FZ

    def epgx_grad(self, noadd=False):
        if self.FZ is None:
            raise ValueError("FZ states are not initialized. Call epgx_m0 first.")

        if not noadd:
            # Add a new column of zeros for the new highest order state
            self.FZ = torch.cat((self.FZ, torch.zeros((self.FZ.shape[0], 1), dtype=torch.complex64, device=self.device)), dim=1)

        num_states_k = self.FZ.shape[1]

        if num_states_k == 1 and noadd: # Special case for single k-state, no expansion
            # F+[0] = F-[0]* ; F+[0] (conj)-> F-[0] (conj)
            self.FZ[0, 0] = torch.conj(self.FZ[1, 0])
            if self.model_type == 'full':
                self.FZ[3, 0] = torch.conj(self.FZ[4, 0])
            return self.FZ
        
        # Shift F+ states
        # F_k+1 <- F_k (clone to avoid in-place issues)
        self.FZ[0, 1:] = self.FZ[0, :-1].clone() 
        # Update F+[0] using F-[0]
        self.FZ[0, 0] = torch.conj(self.FZ[1, 0]) 
        
        if self.model_type == 'full':
            self.FZ[3, 1:] = self.FZ[3, :-1].clone()
            self.FZ[3, 0] = torch.conj(self.FZ[4, 0])

        # Shift F- states
        # F_k-1 <- F_k (clone to avoid in-place issues)
        self.FZ[1, :-1] = self.FZ[1, 1:].clone()
        # Set the last F- state to 0
        self.FZ[1, -1] = 0j 
        
        if self.model_type == 'full':
            self.FZ[4, :-1] = self.FZ[4, 1:].clone()
            self.FZ[4, -1] = 0j
            
        return self.FZ

    def epgx_relax(self, T_ms, T1_list_ms, T2_list_ms, ka_per_s, f, deltab_hz=0):
        if self.FZ is None:
            raise ValueError("FZ states are not initialized. Call epgx_m0 first.")

        T_s = self._to_float_tensor(T_ms) / 1000.0
        T1_list_s = self._to_float_tensor(T1_list_ms) / 1000.0
        T2_list_s = self._to_float_tensor(T2_list_ms) / 1000.0
        ka_per_s = self._to_float_tensor(ka_per_s)
        f = self._to_float_tensor(f)
        deltab_hz = self._to_float_tensor(deltab_hz)

        R1a = 1.0 / T1_list_s[0]
        R1b = 1.0 / T1_list_s[1]
        R2a = 1.0 / T2_list_s[0]
        R2b = 1.0 / T2_list_s[1]

        if f > 0:
            kb_per_s = ka_per_s * (1.0 - f) / f
        else:
            kb_per_s = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        M0a = 1.0 - f
        M0b = f

        if self.model_type == 'full':
            Za, Zb = 2, 5
            FPa, FMa, FPb, FMb = 0, 1, 3, 4 # F+a, F-a, F+b, F-b

            # Longitudinal Relaxation & Exchange
            L_matrix = torch.tensor([[-R1a - ka_per_s, kb_per_s],
                                     [ka_per_s, -R1b - kb_per_s]], dtype=torch.complex64, device=self.device)
            E_L = torch.matrix_exp(L_matrix * T_s)
            
            # Apply to all k-states
            Z_states = self.FZ[[Za, Zb], :]
            Z_updated = torch.matmul(E_L, Z_states)
            self.FZ[[Za, Zb], :] = Z_updated

            # Recovery for Z0 state (first column)
            # Ensure Z_recovery is complex to match self.FZ dtype
            Z_recovery = torch.tensor([[M0a * (1 - torch.exp(-T_s * R1a))],
                                       [M0b * (1 - torch.exp(-T_s * R1b))]], dtype=torch.complex64, device=self.device)
            self.FZ[[Za, Zb], 0] = self.FZ[[Za, Zb], 0] + Z_recovery.squeeze()


            # Transverse Relaxation & Exchange
            LambdaT = torch.zeros((4, 4), dtype=torch.complex64, device=self.device)
            LambdaT[0,0] = -R2a - ka_per_s
            LambdaT[1,1] = -R2a - ka_per_s
            LambdaT[2,2] = -R2b - kb_per_s - 2j * torch.pi * deltab_hz
            LambdaT[3,3] = -R2b - kb_per_s + 2j * torch.pi * deltab_hz
            LambdaT[0,2] = kb_per_s  # F+a from F+b
            LambdaT[1,3] = kb_per_s  # F-a from F-b
            LambdaT[2,0] = ka_per_s  # F+b from F+a
            LambdaT[3,1] = ka_per_s  # F-b from F-a
            E_T = torch.matrix_exp(LambdaT * T_s)

            F_states = self.FZ[[FPa, FMa, FPb, FMb], :]
            F_updated = torch.matmul(E_T, F_states)
            self.FZ[[FPa, FMa, FPb, FMb], :] = F_updated

        elif self.model_type == 'MT':
            Za, Zb = 2, 3
            FPa, FMa = 0, 1

            # Longitudinal Relaxation & Exchange
            L_matrix = torch.tensor([[-R1a - ka_per_s, kb_per_s],
                                     [ka_per_s, -R1b - kb_per_s]], dtype=torch.complex64, device=self.device)
            E_L = torch.matrix_exp(L_matrix * T_s)

            Z_states = self.FZ[[Za, Zb], :]
            Z_updated = torch.matmul(E_L, Z_states)
            self.FZ[[Za, Zb], :] = Z_updated
            
            # Recovery for Z0 state
            Z_recovery = torch.tensor([[M0a * (1 - torch.exp(-T_s * R1a))],
                                       [M0b * (1 - torch.exp(-T_s * R1b))]], dtype=torch.complex64, device=self.device)
            self.FZ[[Za, Zb], 0] = self.FZ[[Za, Zb], 0] + Z_recovery.squeeze()

            # Transverse Relaxation for F+a, F-a
            E2a = torch.exp(-T_s * R2a)
            self.FZ[FPa, :] *= E2a
            self.FZ[FMa, :] *= E2a
            
        return self.FZ

    def epgx_cpmg(self, flipangle_series_deg, esp_ms, T1_list_ms, T2_list_ms, ka_per_s, f, deltab_hz=0, initial_flip_deg=90.0, initial_phase_deg=90.0):
        if not isinstance(flipangle_series_deg, torch.Tensor):
            flipangle_series_deg = torch.tensor(flipangle_series_deg, dtype=torch.float32, device=self.device)
        elif flipangle_series_deg.device != self.device:
            flipangle_series_deg = flipangle_series_deg.to(device=self.device, dtype=torch.float32)
        else:
            flipangle_series_deg = flipangle_series_deg.float()

        self.epgx_m0(f)
        self.epgx_rf(initial_flip_deg, initial_phase_deg)

        etl = len(flipangle_series_deg)
        if self.model_type == 'full':
            signals = torch.zeros((2, etl), dtype=torch.complex64, device=self.device)
        else: # MT
            signals = torch.zeros((1, etl), dtype=torch.complex64, device=self.device)

        for i in range(etl):
            self.epgx_relax(esp_ms / 2.0, T1_list_ms, T2_list_ms, ka_per_s, f, deltab_hz)
            self.epgx_grad(noadd=False)
            
            current_flip_deg = flipangle_series_deg[i]
            self.epgx_rf(current_flip_deg, 0.0) # Assuming refocusing phase is 0 (about X-axis)
            
            self.epgx_grad(noadd=False) # This was specified as noadd=False in description for CPMG
            self.epgx_relax(esp_ms / 2.0, T1_list_ms, T2_list_ms, ka_per_s, f, deltab_hz)
            
            if self.model_type == 'full':
                signals[0, i] = self.FZ[0, 0].clone()
                signals[1, i] = self.FZ[3, 0].clone()
            else: # MT
                signals[0, i] = self.FZ[0, 0].clone()
        
        return signals

    def epgx_rfspoil(self, flipangle_deg, rf_phase_series_deg, TR_ms, T1_list_ms, T2_list_ms, ka_per_s, f, WT=None, deltab_hz=0):
        if not isinstance(rf_phase_series_deg, torch.Tensor):
            rf_phase_series_deg = torch.tensor(rf_phase_series_deg, dtype=torch.float32, device=self.device)
        elif rf_phase_series_deg.device != self.device:
            rf_phase_series_deg = rf_phase_series_deg.to(device=self.device, dtype=torch.float32)
        else:
            rf_phase_series_deg = rf_phase_series_deg.float()

        self.epgx_m0(f)

        num_pulses = len(rf_phase_series_deg)
        if self.model_type == 'full':
            signals = torch.zeros((2, num_pulses), dtype=torch.complex64, device=self.device)
        else: # MT
            signals = torch.zeros((1, num_pulses), dtype=torch.complex64, device=self.device)

        for n in range(num_pulses):
            self.epgx_relax(TR_ms, T1_list_ms, T2_list_ms, ka_per_s, f, deltab_hz)
            self.epgx_grad(noadd=False) # This was False in CPMG and seems appropriate for RF-spoiled too

            current_rf_phase_deg = rf_phase_series_deg[n]
            self.epgx_rf(flipangle_deg, current_rf_phase_deg, WT=WT)

            if self.model_type == 'full':
                signals[0, n] = self.FZ[0, 0].clone()
                signals[1, n] = self.FZ[3, 0].clone()
            else: # MT
                signals[0, n] = self.FZ[0, 0].clone()

            # Perfect spoiling
            self.FZ = self.FZ[:, 0:1].clone() # Keep only Z0 states (F0, Z0 for A and B)
            self.FZ[0,0] = 0j # Zero F0+ for compartment A
            self.FZ[1,0] = 0j # Zero F0- for compartment A
            if self.model_type == 'full':
                self.FZ[3,0] = 0j # Zero F0+ for compartment B
                self.FZ[4,0] = 0j # Zero F0- for compartment B
        
        return signals
