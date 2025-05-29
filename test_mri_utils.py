import pytest
import torch
from mri_utils import design_time_optimal_gradient

# Constants for tests
GAMMA_HZ_PER_T_test = 42.577478518e6
gamma_eff_test = GAMMA_HZ_PER_T_test * 1e-6 # m^-1 / (mT/m * ms)
GMAX_mT_m = 40.0
SMAX_T_m_s = 150.0
DT_ms = 0.01 # 10 us

@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_triangular_waveform(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    area_cm = 0.05  # Expected to be triangular
    g, t = design_time_optimal_gradient(area_cm, GMAX_mT_m, SMAX_T_m_s, DT_ms, device=device)

    assert isinstance(g, torch.Tensor)
    assert isinstance(t, torch.Tensor)
    assert g.device.type == device
    assert t.device.type == device
    assert g.dtype == torch.float32
    assert t.dtype == torch.float32
    assert g.ndim == 1
    assert t.ndim == 1
    assert g.numel() == t.numel()
    assert g.numel() > 0

    # Check area
    calculated_area_m_inv = gamma_eff_test * torch.sum(g) * DT_ms
    target_area_m_inv = area_cm * 100.0
    torch.testing.assert_close(calculated_area_m_inv, torch.tensor(target_area_m_inv, dtype=torch.float32, device=device), rtol=1e-3, atol=1e-4)

    # Check peak (should be less than Gmax for this area)
    assert torch.max(torch.abs(g)) < GMAX_mT_m + 1e-3 # Add tolerance for normalization adjustments

@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_trapezoidal_waveform(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    area_cm = 0.2  # Expected to be trapezoidal
    g, t = design_time_optimal_gradient(area_cm, GMAX_mT_m, SMAX_T_m_s, DT_ms, device=device)

    assert isinstance(g, torch.Tensor)
    assert isinstance(t, torch.Tensor)
    assert g.device.type == device
    assert t.device.type == device
    assert g.numel() > 0

    # Check area
    calculated_area_m_inv = gamma_eff_test * torch.sum(g) * DT_ms
    target_area_m_inv = area_cm * 100.0
    torch.testing.assert_close(calculated_area_m_inv, torch.tensor(target_area_m_inv, dtype=torch.float32, device=device), rtol=1e-3, atol=1e-4)

    # Check peak (should be close to Gmax)
    torch.testing.assert_close(torch.max(torch.abs(g)), torch.tensor(GMAX_mT_m, dtype=torch.float32, device=device), rtol=1e-3, atol=1e-3) # Peak can be slightly off due to normalization

@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_negative_area(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    area_cm = -0.1
    g, t = design_time_optimal_gradient(area_cm, GMAX_mT_m, SMAX_T_m_s, DT_ms, device=device)
    assert isinstance(g, torch.Tensor)
    assert g.numel() > 0
    
    calculated_area_m_inv = gamma_eff_test * torch.sum(g) * DT_ms
    target_area_m_inv = area_cm * 100.0
    torch.testing.assert_close(calculated_area_m_inv, torch.tensor(target_area_m_inv, dtype=torch.float32, device=device), rtol=1e-3, atol=1e-4)
    assert torch.all(g <= 1e-6) # Should be negative or zero

@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_zero_area(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    area_cm = 0.0
    g, t = design_time_optimal_gradient(area_cm, GMAX_mT_m, SMAX_T_m_s, DT_ms, device=device)
    assert isinstance(g, torch.Tensor)
    assert g.numel() > 0 # Current implementation returns [0.0]
    
    calculated_area_m_inv = gamma_eff_test * torch.sum(g) * DT_ms
    torch.testing.assert_close(calculated_area_m_inv, torch.tensor(0.0, dtype=torch.float32, device=device), rtol=1e-5, atol=1e-7)
    assert torch.allclose(g, torch.tensor([0.0], device=device, dtype=torch.float32))

@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_area_equals_atri(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    s_max_mT_m_ms = SMAX_T_m_s / 1000.0
    t_ramp_to_g_max_ms = GMAX_mT_m / s_max_mT_m_ms
    a_tri_m_inv_val = gamma_eff_test * GMAX_mT_m * t_ramp_to_g_max_ms
    area_cm_eq_atri = a_tri_m_inv_val / 100.0

    g, t = design_time_optimal_gradient(area_cm_eq_atri, GMAX_mT_m, SMAX_T_m_s, DT_ms, device=device)
    assert g.numel() > 0

    calculated_area_m_inv = gamma_eff_test * torch.sum(g) * DT_ms
    target_area_m_inv = area_cm_eq_atri * 100.0
    torch.testing.assert_close(calculated_area_m_inv, torch.tensor(target_area_m_inv, dtype=torch.float32, device=device), rtol=1e-3, atol=1e-4)
    
    torch.testing.assert_close(torch.max(torch.abs(g)), torch.tensor(GMAX_mT_m, dtype=torch.float32, device=device), rtol=1e-3, atol=1e-3)
