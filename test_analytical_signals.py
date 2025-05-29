import pytest
import torch
from analytical_signals import (
    calculate_spgr_signal, 
    calculate_gre_spoiled_signal, 
    calculate_bssfp_signal
)

# Define common test parameters
T1_MS = torch.tensor(1000.0)
T2_MS = torch.tensor(100.0)
TE_MS = torch.tensor(10.0)
TR_MS = torch.tensor(50.0) # Relatively short TR for steady-state effects
FLIP_ANGLE_DEG = torch.tensor(30.0)
DELTA_FREQ_HZ = torch.tensor(0.0)
FLIP_ANGLE_ZERO = torch.tensor(0.0)

# Constants for comparison
epsilon = 1e-6

@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_spgr_signal(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Test with standard parameters
    sig, mz = calculate_spgr_signal(T1_MS, T2_MS, TE_MS, TR_MS, FLIP_ANGLE_DEG, device=device)
    assert sig.shape == () # Scalar output
    assert mz.shape == ()
    assert sig.dtype == torch.float32
    assert mz.dtype == torch.float32
    assert sig.device.type == device
    assert mz.device.type == device

    # Test with flip_angle_deg = 0.0
    sig_zero_fa, mz_zero_fa = calculate_spgr_signal(T1_MS, T2_MS, TE_MS, TR_MS, FLIP_ANGLE_ZERO, device=device)
    torch.testing.assert_close(sig_zero_fa, torch.tensor(0.0, device=device, dtype=torch.float32), atol=epsilon, rtol=epsilon)
    # Mz at TE after 0-deg flip: should be M0 * exp(-TE/T1) if starting from M0=1
    # The formula for Mz_ss_at_TE is (1-R)*E_TE_T1*c / (1-R*c + eps). With c=1, R=exp(-TR/T1):
    # (1-R)*E_TE_T1 / (1-R) = E_TE_T1 = exp(-TE/T1)
    expected_mz_zero_fa = torch.exp(-TE_MS / T1_MS)
    torch.testing.assert_close(mz_zero_fa, expected_mz_zero_fa.to(device=device, dtype=torch.float32), atol=epsilon, rtol=epsilon)


    # Test with very long TR_ms (TR = 5 * T1)
    # Mz before pulse should be fully recovered (approx 1.0 if M0=1)
    # sig should approach sin(flip_rad) * exp(-TE/T2)
    # Mz_ss_at_TE should approach cos(flip_rad) * exp(-TE/T1)
    TR_long_ms = 5 * T1_MS
    sig_long_tr, mz_long_tr = calculate_spgr_signal(T1_MS, T2_MS, TE_MS, TR_long_ms, FLIP_ANGLE_DEG, device=device)
    
    flip_rad = torch.deg2rad(FLIP_ANGLE_DEG)
    expected_sig_long_tr = torch.sin(flip_rad) * torch.exp(-TE_MS / T2_MS)
    expected_mz_long_tr = torch.cos(flip_rad) * torch.exp(-TE_MS / T1_MS)
    
    torch.testing.assert_close(sig_long_tr, expected_sig_long_tr.to(device=device, dtype=torch.float32), atol=1e-3, rtol=1e-3) # Lower tolerance due to approximation
    torch.testing.assert_close(mz_long_tr, expected_mz_long_tr.to(device=device, dtype=torch.float32), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_gre_spoiled_signal(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Test with standard parameters
    sig = calculate_gre_spoiled_signal(T1_MS, T2_MS, TE_MS, TR_MS, FLIP_ANGLE_DEG, device=device)
    assert sig.shape == ()
    assert sig.dtype == torch.float32
    assert sig.device.type == device

    # Test with flip_angle_deg = 0.0
    sig_zero_fa = calculate_gre_spoiled_signal(T1_MS, T2_MS, TE_MS, TR_MS, FLIP_ANGLE_ZERO, device=device)
    torch.testing.assert_close(sig_zero_fa, torch.tensor(0.0, device=device, dtype=torch.float32), atol=epsilon, rtol=epsilon)


@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_bssfp_signal(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Test with standard parameters (delta_freq_hz = 0.0)
    s = calculate_bssfp_signal(T1_MS, T2_MS, TE_MS, TR_MS, FLIP_ANGLE_DEG, DELTA_FREQ_HZ, device=device)
    assert s.shape == ()
    assert s.dtype == torch.complex64
    assert s.device.type == device

    # Test with flip_angle_deg = 0.0
    s_zero_fa = calculate_bssfp_signal(T1_MS, T2_MS, TE_MS, TR_MS, FLIP_ANGLE_ZERO, DELTA_FREQ_HZ, device=device)
    # With FA=0, sf=0. b_term = 0. a_term = 0. So s0 = 0. s = 0.
    torch.testing.assert_close(s_zero_fa, torch.tensor(0.0 + 0.0j, device=device, dtype=torch.complex64), atol=epsilon, rtol=epsilon)

    # Test with non-zero delta_freq_hz
    delta_freq_non_zero = torch.tensor(50.0) # 50 Hz off-resonance
    s_offres = calculate_bssfp_signal(T1_MS, T2_MS, TE_MS, TR_MS, FLIP_ANGLE_DEG, delta_freq_non_zero, device=device)
    assert s_offres.shape == ()
    assert s_offres.dtype == torch.complex64
    assert s_offres.device.type == device
    
    # Test with delta_freq_hz = 1 / (2*TR_ms) -> phi_rad = pi
    # This should lead to specific behavior (e.g. signal inversion for on-resonance component)
    # TR_s = TR_MS / 1000.0
    # delta_freq_half_tr_inv = 1.0 / (2.0 * TR_s)
    # s_half_tr_inv = calculate_bssfp_signal(T1_MS, T2_MS, TE_MS, TR_MS, FLIP_ANGLE_DEG, delta_freq_half_tr_inv, device=device)
    # For phi=pi, cos(phi)=-1. 
    # s0_num = -a+b
    # s0_den = -c+d
    # This is a more complex check, ensuring it runs is the primary goal here.
    assert torch.isfinite(s_offres).all()

```
