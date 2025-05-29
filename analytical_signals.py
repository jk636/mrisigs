import torch

# Helper function to ensure inputs are tensors on the correct device
def _to_tensor(value, device, dtype=torch.float32):
    if not isinstance(value, torch.Tensor):
        return torch.tensor(value, dtype=dtype, device=device)
    return value.to(device=device, dtype=dtype)

def calculate_spgr_signal(T1_ms: torch.Tensor | float, 
                          T2_ms: torch.Tensor | float, 
                          TE_ms: torch.Tensor | float, 
                          TR_ms: torch.Tensor | float, 
                          flip_angle_deg: torch.Tensor | float, 
                          device: str = 'cpu') -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the steady-state SPGR (Spoiled Gradient Recalled Echo) signal.
    Based on exrecsignal.m.

    Args:
        T1_ms (torch.Tensor | float): Longitudinal relaxation time (ms).
        T2_ms (torch.Tensor | float): Transverse relaxation time (ms).
        TE_ms (torch.Tensor | float): Echo time (ms).
        TR_ms (torch.Tensor | float): Repetition time (ms).
        flip_angle_deg (torch.Tensor | float): Flip angle (degrees).
        device (str): PyTorch device for tensor creation ('cpu' or 'cuda').

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 
            - sig: The transverse signal magnitude at TE.
            - Mz_ss_at_TE: The longitudinal magnetization magnitude at TE.
    """
    T1 = _to_tensor(T1_ms, device)
    T2 = _to_tensor(T2_ms, device)
    TE = _to_tensor(TE_ms, device)
    TR = _to_tensor(TR_ms, device)
    FA_deg = _to_tensor(flip_angle_deg, device)
    
    epsilon = 1e-9 # Small value to prevent division by zero

    FA_rad = torch.deg2rad(FA_deg)
    s = torch.sin(FA_rad)
    c = torch.cos(FA_rad)

    E_TE = torch.exp(-TE / T2)       # Decay during TE
    R_TR = torch.exp(-TR / T1)       # Recovery during TR

    # Denominator: (1 - E1*cos(alpha))
    den = 1 - R_TR * c
    
    # Handle cases where denominator is very small
    # Matlab original: if (abs(1-R*c)<1e-7) sig = 1; else ...
    # Using epsilon in denominator for PyTorch version for batch processing
    
    sig = (1 - R_TR) * E_TE * s / (den + epsilon)
    
    # Mz_ss_at_TE from Matlab M(3) = (1-R)*E*c/(1-R*c)
    # This is Mz just before the next RF pulse, decayed by TE.
    # More standard Mz_ss (immediately after RF pulse, before TE decay) would be (1-R_TR) / (den+epsilon) * c
    # Mz just after RF pulse: Mz_plus = M0 * (1-E1)/(1-E1*c) * c
    # Mz at TE: Mz_te = Mz_plus * exp(-TE/T1) + M0_recovery_during_TE
    # However, the Matlab script implies Mz at TE derived from the transverse signal context.
    # The line `M = [S;0;(1-R)*E*c/(1-R*c)];` suggests Mz at TE.
    # Let's use Mz_ss right before the RF pulse for the Mz component, then decay by TE
    # Mz_before_rf = M0_initial * (1-R_TR) / (1-R_TR*c) for M0_initial=1
    # Mz_ss_before_rf = (1-R_TR) / (den + epsilon) # This is Mz_eq_prime
    # Mz after RF pulse (Mz+) is Mz_ss_before_rf * c
    # Mz_ss_at_TE = ( (1-R_TR)*c / (den + epsilon) ) * torch.exp(-TE/T1) + (1 - torch.exp(-TE/T1))
    # The Matlab exrecsignal.m seems to use a simplified Mz at TE: (1-R)*E*c/(1-R*c)
    # This is actually Mz_plus * E_TE_T1 (longitudinal component after RF, decayed by T1 over TE) if M0=1
    # Let Mz_ss_at_TE = (1-R_TR)*E_TE_T1*c / (den + epsilon) where E_TE_T1 = exp(-TE/T1)
    E_TE_T1 = torch.exp(-TE / T1)
    Mz_ss_at_TE = (1 - R_TR) * E_TE_T1 * c / (den + epsilon)

    return sig, Mz_ss_at_TE

def calculate_gre_spoiled_signal(T1_ms: torch.Tensor | float, 
                                 T2_ms: torch.Tensor | float, 
                                 TE_ms: torch.Tensor | float, 
                                 TR_ms: torch.Tensor | float, 
                                 flip_angle_deg: torch.Tensor | float, 
                                 device: str = 'cpu') -> torch.Tensor:
    """
    Calculates the steady-state GRE signal with perfect spoiling.
    Based on gresignal.m (Buxton 1989).

    Args:
        T1_ms (torch.Tensor | float): Longitudinal relaxation time (ms).
        T2_ms (torch.Tensor | float): Transverse relaxation time (ms).
        TE_ms (torch.Tensor | float): Echo time (ms).
        TR_ms (torch.Tensor | float): Repetition time (ms).
        flip_angle_deg (torch.Tensor | float): Flip angle (degrees).
        device (str): PyTorch device for tensor creation ('cpu' or 'cuda').

    Returns:
        torch.Tensor: The transverse signal magnitude at TE.
    """
    T1 = _to_tensor(T1_ms, device)
    T2 = _to_tensor(T2_ms, device)
    TE = _to_tensor(TE_ms, device)
    TR = _to_tensor(TR_ms, device)
    FA_deg = _to_tensor(flip_angle_deg, device)
    
    epsilon = 1e-9

    FA_rad = torch.deg2rad(FA_deg)
    s = torch.sin(FA_rad)
    c = torch.cos(FA_rad)

    E1 = torch.exp(-TR / T1)
    E2 = torch.exp(-TR / T2) # Note: Buxton uses E2 for TR, not TE
    E_TE = torch.exp(-TE / T2)

    # Denominators from gresignal.m (Buxton's paper has slightly different grouping)
    # Matlab code: A = sqrt((1-E1.*c-E2.^2.*(E1-c)).^2-(E2.*(E1-1).*(1+c)).^2);
    # B = (1-E1.*c-E2.^2.*(E1-c)); C=(E2.*(E1-1).*(1+c));
    # sig = s.*(1-E1).*E_TE.*(C-E2.*(B-A))./(A.*C);
    
    B_term = 1 - E1*c - E2**2 * (E1-c)
    C_term = E2 * (E1-1) * (1+c)
    
    # Add epsilon inside sqrt for stability if B_term**2 - C_term**2 is negative due to precision
    A_term_squared = B_term**2 - C_term**2
    A_term = torch.sqrt(torch.clamp(A_term_squared, min=epsilon)) # Clamp ensures non-negative for sqrt
                                                                # Or use A_term = torch.sqrt(A_term_squared + epsilon)
                                                                # Original Matlab simply used sqrt, which can lead to NaNs if B_term**2 - C_term**2 is negative.
                                                                # Adding epsilon inside sqrt is safer if A_term_squared can be small negative.
                                                                # If A_term_squared is truly zero, A_term will be small.

    den_sig = A_term * C_term
    
    # Original Matlab sets sig=1 if den_sig is 0.
    # We use epsilon in denominator for batch processing.
    sig_numerator = s * (1-E1) * E_TE * (C_term - E2 * (B_term - A_term))
    sig = sig_numerator / (den_sig + epsilon * torch.sign(den_sig) + epsilon) # Add signed epsilon to avoid changing sign of den_sig if it's non-zero

    # Handle flip_angle_deg = 0 case where s=0, C_term can also be 0
    # if FA_deg is 0, s=0, so sig=0. This is handled.
    # If C_term is 0 (e.g. E1=1 or c=-1), then den_sig is 0.
    # If c = -1 (180 deg), C_term = 0. B_term = 1+E1-E2^2(E1+1). A_term = abs(B_term). den_sig = 0.
    # If FA_deg is very small, s is small.
    # If FA_deg = 0, s = 0 -> sig = 0.
    # The original Matlab code `if (abs(A.*C)==0) S=1;` is problematic as S=1 for 0 flip angle is wrong.
    # For FA=0, s=0, num=0. C_term = E2*(E1-1)*2. A_term = sqrt((1-E1-E2^2*(E1-1))^2 - C_term^2).
    # If FA_deg is 0, then s=0, making sig_numerator=0, so sig=0. This is correct.
    
    return sig

def calculate_bssfp_signal(T1_ms: torch.Tensor | float, 
                           T2_ms: torch.Tensor | float, 
                           TE_ms: torch.Tensor | float, 
                           TR_ms: torch.Tensor | float, 
                           flip_angle_deg: torch.Tensor | float, 
                           delta_freq_hz: torch.Tensor | float, 
                           device: str = 'cpu') -> torch.Tensor:
    """
    Calculates the steady-state bSSFP (balanced Steady-State Free Precession) signal.
    Based on bssfp.m.

    Args:
        T1_ms (torch.Tensor | float): Longitudinal relaxation time (ms).
        T2_ms (torch.Tensor | float): Transverse relaxation time (ms).
        TE_ms (torch.Tensor | float): Echo time (ms).
        TR_ms (torch.Tensor | float): Repetition time (ms).
        flip_angle_deg (torch.Tensor | float): Flip angle (degrees).
        delta_freq_hz (torch.Tensor | float): Off-resonance frequency (Hz).
        device (str): PyTorch device for tensor creation ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Complex tensor representing the bSSFP signal at TE.
    """
    T1 = _to_tensor(T1_ms, device)
    T2 = _to_tensor(T2_ms, device)
    TE = _to_tensor(TE_ms, device)
    TR = _to_tensor(TR_ms, device)
    FA_deg = _to_tensor(flip_angle_deg, device)
    df_hz = _to_tensor(delta_freq_hz, device)
    
    epsilon = 1e-9

    FA_rad = torch.deg2rad(FA_deg)
    sf = torch.sin(FA_rad)
    cf = torch.cos(FA_rad)

    E1 = torch.exp(-TR / T1)
    E2 = torch.exp(-TR / T2)
    
    # Off-resonance phase per TR
    phi_rad_per_TR = 2 * torch.pi * df_hz * TR / 1000.0 # TR in ms

    # Terms from bssfp.m (Freeman and Hill, JMR 1971; also similar to Hennig JMR 1988)
    # These terms build the Mxy_plus component right after RF pulse
    a_term = -(1 - E1) * E2 * sf
    b_term = (1 - E1) * sf
    c_term = E2 * (E1 - 1) * (1 + cf) # Note: (E1-1) makes this negative or zero
    d_term = 1 - E1 * cf - (E1 - cf) * E2**2
    
    # Denominator for Mxy_plus
    den_Mxy_plus = c_term * torch.cos(phi_rad_per_TR) + d_term
    
    # Mxy_plus before TE decay, M0=1 assumed
    # Numerator for Mxy_plus
    # Need to ensure results are complex from this point
    phi_rad_per_TR_complex = _to_tensor(phi_rad_per_TR, device, dtype=torch.complex64) # ensure complex for exp
    a_term_complex = _to_tensor(a_term, device, dtype=torch.complex64)
    b_term_complex = _to_tensor(b_term, device, dtype=torch.complex64)
    den_Mxy_plus_complex = _to_tensor(den_Mxy_plus, device, dtype=torch.complex64)

    Mxy_plus = (a_term_complex * torch.exp(1j * phi_rad_per_TR_complex) + b_term_complex) / (den_Mxy_plus_complex + epsilon)
    
    # Signal at TE: decay Mxy_plus by T2 over TE, and apply phase from off-resonance over TE
    E_TE_T2 = torch.exp(-TE / T2)
    phi_rad_per_TE = 2 * torch.pi * df_hz * TE / 1000.0 # TE in ms
    
    E_TE_T2_complex = _to_tensor(E_TE_T2, device, dtype=torch.complex64)
    phi_rad_per_TE_complex = _to_tensor(phi_rad_per_TE, device, dtype=torch.complex64)

    s = Mxy_plus * E_TE_T2_complex * torch.exp(-1j * phi_rad_per_TE_complex) # bSSFP is typically defined with -phi for TE evolution
                                                                      # The Matlab reference s0 * exp(-TE/T2) * exp(-j*phi*(TE/TR))
                                                                      # matches this if phi in matlab was phi_rad_per_TR.
                                                                      # Here phi_rad_per_TE is df*TE.
    return s
