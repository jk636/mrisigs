import torch

def design_time_optimal_gradient(area_cm_inv: float, 
                                 g_max_mT_m: float, 
                                 s_max_T_m_s: float, 
                                 dt_ms: float,
                                 device: str = 'cpu'):
    """
    Designs a time-optimal gradient waveform (triangular or trapezoidal)
    for a given area, Gmax, Smax, and sampling interval dt.

    Args:
        area_cm_inv (float): Desired gradient area in 1/cm.
        g_max_mT_m (float): Maximum gradient amplitude in mT/m.
        s_max_T_m_s (float): Maximum slew rate in T/m/s.
        dt_ms (float): Sampling interval in ms.
        device (str): PyTorch device for tensor creation ('cpu' or 'cuda').

    Returns:
        tuple (g_mT_m, t_ms):
            - g_mT_m: PyTorch tensor representing the gradient waveform (mT/m).
            - t_ms: PyTorch tensor representing the time vector (ms).
    """
    GAMMA_HZ_PER_T = 42.577478518e6  # Hz/T
    epsilon = 1e-9 # For floating point comparisons

    # Unit conversions
    area_m_inv = area_cm_inv * 100.0
    s_max_mT_m_ms = s_max_T_m_s / 1000.0  # mT/m/ms
    gamma_eff = GAMMA_HZ_PER_T * 1e-6    # m^-1 / (mT/m * ms)

    g_max_mT_m_t = torch.tensor(g_max_mT_m, dtype=torch.float32, device=device)
    s_max_mT_m_ms_t = torch.tensor(s_max_mT_m_ms, dtype=torch.float32, device=device)
    dt_ms_t = torch.tensor(dt_ms, dtype=torch.float32, device=device)
    area_m_inv_t = torch.tensor(area_m_inv, dtype=torch.float32, device=device)

    if abs(area_m_inv) < epsilon:
        g_mT_m = torch.tensor([0.0], dtype=torch.float32, device=device)
        t_ms = torch.tensor([0.0], dtype=torch.float32, device=device)
        return g_mT_m, t_ms

    # Area of full triangle if limited by Smax to reach Gmax
    # t_ramp_to_g_max_ms = Gmax / Smax_eff
    t_ramp_to_g_max_ms = g_max_mT_m_t / s_max_mT_m_ms_t
    # Area = gamma_eff * Gmax * t_ramp_to_g_max (one full ramp up, one full ramp down)
    a_tri_m_inv = gamma_eff * g_max_mT_m_t * t_ramp_to_g_max_ms

    g_mT_m = torch.tensor([], dtype=torch.float32, device=device)

    if torch.abs(area_m_inv_t) <= a_tri_m_inv + epsilon : # Triangular waveform
        # Peak gradient for the triangle to achieve the desired area
        # area = gamma_eff * g_peak * t_ramp_peak
        # g_peak = s_max * t_ramp_peak => t_ramp_peak = g_peak / s_max
        # area = gamma_eff * g_peak^2 / s_max  => g_peak^2 = area * s_max / gamma_eff
        g_peak_mT_m = torch.sqrt(torch.abs(area_m_inv_t) * s_max_mT_m_ms_t / gamma_eff)
        if g_peak_mT_m > g_max_mT_m_t: # Should not happen if a_tri_m_inv is correct
             g_peak_mT_m = g_max_mT_m_t

        t_ramp_ms = g_peak_mT_m / s_max_mT_m_ms_t
        n_ramp = torch.ceil(t_ramp_ms / dt_ms_t).int().item()
        n_ramp = max(1, n_ramp) # Ensure at least one point for non-zero area

        # Construct ramp using actual discrete points
        # Peak might be slightly different due to discretization
        g_ramp_up = (torch.arange(1, n_ramp + 1, device=device, dtype=torch.float32) / n_ramp) * g_peak_mT_m
        
        if n_ramp == 1 :
             g_mT_m = g_ramp_up # Single point peak
        else:
             g_mT_m = torch.cat((g_ramp_up, torch.flip(g_ramp_up[:-1], dims=[0])))
        
    else: # Trapezoidal waveform
        t_ramp_ms = g_max_mT_m_t / s_max_mT_m_ms_t
        n_ramp = torch.ceil(t_ramp_ms / dt_ms_t).int().item()
        n_ramp = max(1, n_ramp) # Ramp must have at least one point if Gmax > 0

        g_ramp_up = (torch.arange(1, n_ramp + 1, device=device, dtype=torch.float32) / n_ramp) * g_max_mT_m_t
        g_ramp_down = torch.flip(g_ramp_up, dims=[0])
        
        # Area of the ramps (summing discrete points)
        # For ramp_down, if n_ramp=1, g_ramp_down[:-1] is empty, sum is 0. Correct.
        area_ramps_m_inv = gamma_eff * (torch.sum(g_ramp_up) + torch.sum(g_ramp_down[:-1])) * dt_ms_t
        
        area_plat_needed_m_inv = torch.abs(area_m_inv_t) - area_ramps_m_inv
        
        if area_plat_needed_m_inv < 0: # Should not happen if logic for tri vs trap is correct
            area_plat_needed_m_inv = torch.tensor(0.0, device=device)

        t_plat_ms = area_plat_needed_m_inv / (gamma_eff * g_max_mT_m_t + epsilon) # add epsilon to avoid div by zero if g_max_mT_m_t is zero
        n_plat = torch.ceil(t_plat_ms / dt_ms_t).int().item()
        n_plat = max(0, n_plat)

        g_plateau = torch.ones(n_plat, device=device, dtype=torch.float32) * g_max_mT_m_t
        
        if n_ramp == 1 and n_plat == 0: # Essentially a triangle peaking at Gmax
            g_mT_m = g_ramp_up 
        elif n_plat == 0: # Triangle that hits Gmax but area_plat_needed is zero/negative
             g_mT_m = torch.cat((g_ramp_up, g_ramp_down[1:])) # Avoid Gmax duplication
        else:
             g_mT_m = torch.cat((g_ramp_up, g_plateau, g_ramp_down))

    if g_mT_m.numel() == 0: # Should be caught by zero area check, but as safety
         g_mT_m = torch.tensor([0.0], dtype=torch.float32, device=device)

    # Normalize area
    current_area_m_inv = gamma_eff * torch.sum(g_mT_m) * dt_ms_t
    if torch.abs(current_area_m_inv) > epsilon:
        g_mT_m = g_mT_m * (torch.abs(area_m_inv_t) / current_area_m_inv)

    # Assign sign based on input area
    if area_m_inv < 0:
        g_mT_m = -g_mT_m
        
    t_ms = torch.arange(g_mT_m.numel(), device=device, dtype=torch.float32) * dt_ms_t
    
    return g_mT_m, t_ms

def calculate_b_value(grad_waveform_mT_m: torch.Tensor, 
                        dt_s: float, 
                        device: str = 'cpu') -> torch.Tensor:
    """
    Calculates the b-value for a given gradient waveform.

    Args:
        grad_waveform_mT_m (torch.Tensor): 1D tensor of gradient amplitudes (mT/m).
        dt_s (float): Sampling interval of the gradient waveform in seconds.
        device (str): PyTorch device for tensor creation ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Scalar tensor representing the b-value in s/mm^2.
    """
    GAMMA_RAD_S_T = 2 * torch.pi * 42.577478518e6  # rad s^-1 T^-1

    if not isinstance(grad_waveform_mT_m, torch.Tensor):
        grad_waveform_mT_m = torch.tensor(grad_waveform_mT_m, dtype=torch.float32, device=device)
    elif grad_waveform_mT_m.device.type != device or grad_waveform_mT_m.dtype != torch.float32:
        grad_waveform_mT_m = grad_waveform_mT_m.to(device=device, dtype=torch.float32)

    dt_s_t = torch.tensor(dt_s, dtype=torch.float32, device=device)

    # Convert gradient from mT/m to T/m
    g_T_m = grad_waveform_mT_m * 1e-3

    # Calculate integral of G(t) using cumsum
    # k(t) = integral_0^t G(tau) dtau
    # Here, k_integral_G_dt represents samples of k(t) * dt, but more accurately,
    # it's samples of integral G(tau)dtau up to that point.
    # The cumsum approach here is: k_n = sum_{i=0 to n} G_i * dt
    k_integral_G_dt_T_s_m = torch.cumsum(g_T_m * dt_s_t, dim=0)
    
    # b = gamma^2 * integral_0^T (k(t))^2 dt
    # Approximated by sum(k_n^2 * dt)
    b_s_m2 = (GAMMA_RAD_S_T**2) * torch.sum(k_integral_G_dt_T_s_m**2) * dt_s_t
    
    # Convert b-value from s/m^2 to s/mm^2
    b_s_mm2 = b_s_m2 * 1e-6
    
    return b_s_mm2

def calculate_b_value_refocused(grad_waveform_T_m: torch.Tensor, 
                                  dt_s: float, 
                                  t_inversion_s: float, 
                                  device: str = 'cpu') -> torch.Tensor:
    """
    Calculates the b-value for a gradient waveform with an effective inversion
    (e.g., due to a 180-degree refocusing pulse).

    Args:
        grad_waveform_T_m (torch.Tensor): 1D tensor of gradient amplitudes (T/m).
        dt_s (float): Sampling interval of the gradient waveform in seconds.
        t_inversion_s (float): Time of the effective inversion pulse (seconds)
                               from the start of the waveform.
        device (str): PyTorch device for tensor creation ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Scalar tensor representing the b-value in s/mm^2.
    """
    GAMMA_RAD_S_T = 2 * torch.pi * 42.577478518e6  # rad s^-1 T^-1

    if not isinstance(grad_waveform_T_m, torch.Tensor):
        g_wf_T_m = torch.tensor(grad_waveform_T_m, dtype=torch.float32, device=device)
    elif grad_waveform_T_m.device.type != device or grad_waveform_T_m.dtype != torch.float32:
        g_wf_T_m = grad_waveform_T_m.to(device=device, dtype=torch.float32)
    else:
        g_wf_T_m = grad_waveform_T_m

    dt_s_t = torch.tensor(dt_s, dtype=torch.float32, device=device)
    t_inversion_s_t = torch.tensor(t_inversion_s, dtype=torch.float32, device=device)

    num_pts = g_wf_T_m.numel()
    inv_idx = torch.floor(t_inversion_s_t / dt_s_t).long().item()
    inv_idx = max(0, min(inv_idx, num_pts)) # Clip inv_idx

    inv_factor = torch.ones_like(g_wf_T_m)
    if inv_idx < num_pts: # If inversion happens within or at the end of the waveform
        inv_factor[inv_idx:] = -1.0
    
    g_effective_T_m = g_wf_T_m * inv_factor
    
    # Calculate integral of effective G(t) using cumsum
    k_integral_G_eff_dt_T_s_m = torch.cumsum(g_effective_T_m * dt_s_t, dim=0)
    
    # b = gamma^2 * integral_0^T (k(t))^2 dt
    b_s_m2 = (GAMMA_RAD_S_T**2) * torch.sum(k_integral_G_eff_dt_T_s_m**2) * dt_s_t
    
    # Convert b-value from s/m^2 to s/mm^2
    b_s_mm2 = b_s_m2 * 1e-6
    
    return b_s_mm2

def calculate_gradient_moments(grad_waveform_T_m: torch.Tensor, 
                                 dt_s: float, 
                                 t_inversion_s: float, 
                                 num_moments: int = 5, 
                                 calculate_absolute: bool = True, 
                                 device: str = 'cpu') -> torch.Tensor:
    """
    Calculates the moments of a gradient waveform, considering an effective inversion.

    Args:
        grad_waveform_T_m (torch.Tensor): 1D tensor of gradient amplitudes (T/m).
        dt_s (float): Sampling interval of the gradient waveform in seconds.
        t_inversion_s (float): Time of the effective inversion pulse (seconds)
                               from the start of the waveform.
        num_moments (int): Number of moments to calculate (M0 to M(num_moments-1)).
        calculate_absolute (bool): If True, returns absolute values of moments.
        device (str): PyTorch device for tensor creation ('cpu' or 'cuda').

    Returns:
        torch.Tensor: 1D tensor of size `num_moments` containing the moments.
                      Units are T/m * s^(n+1) for the n-th moment.
    """
    if not isinstance(grad_waveform_T_m, torch.Tensor):
        g_wf_T_m = torch.tensor(grad_waveform_T_m, dtype=torch.float32, device=device)
    elif grad_waveform_T_m.device.type != device or grad_waveform_T_m.dtype != torch.float32:
        g_wf_T_m = grad_waveform_T_m.to(device=device, dtype=torch.float32)
    else:
        g_wf_T_m = grad_waveform_T_m

    dt_s_t = torch.tensor(dt_s, dtype=torch.float32, device=device)
    t_inversion_s_t = torch.tensor(t_inversion_s, dtype=torch.float32, device=device)

    num_pts = g_wf_T_m.numel()
    inv_idx = torch.floor(t_inversion_s_t / dt_s_t).long().item()
    inv_idx = max(0, min(inv_idx, num_pts)) # Clip inv_idx

    inv_factor = torch.ones_like(g_wf_T_m)
    if inv_idx < num_pts:
        inv_factor[inv_idx:] = -1.0
    
    g_effective_T_m = g_wf_T_m * inv_factor
    
    t_s = torch.arange(num_pts, device=device, dtype=torch.float32) * dt_s_t
    
    moments_val = torch.zeros(num_moments, device=device, dtype=torch.float32)
    
    for n in range(num_moments):
        t_pow_n = t_s ** n
        integrand = t_pow_n * g_effective_T_m
        moments_val[n] = torch.sum(integrand) * dt_s_t
        
    if calculate_absolute:
        return torch.abs(moments_val)
    else:
        return moments_val

if __name__ == '__main__':
    # Example usage for design_time_optimal_gradient:
    area_cm = 0.05  # 1/cm
    Gmax = 40.0    # mT/m
    Smax = 150.0   # T/m/s
    dt = 0.01      # ms (e.g. 10 us)

    print(f"Designing for Area: {area_cm} 1/cm, Gmax: {Gmax} mT/m, Smax: {Smax} T/m/s, dt: {dt} ms")
    
    # Test 1: Triangular
    g_tri, t_tri = design_time_optimal_gradient(area_cm, Gmax, Smax, dt)
    GAMMA_HZ_PER_T_test = 42.577478518e6
    gamma_eff_test = GAMMA_HZ_PER_T_test * 1e-6
    actual_area_tri = torch.sum(g_tri) * dt * gamma_eff_test / 100.0
    print(f"Triangular Waveform (first 10 points): {g_tri[:10]}")
    print(f"Time vector (first 10 points): {t_tri[:10]}")
    print(f"Number of points: {g_tri.numel()}")
    print(f"Target Area: {area_cm} 1/cm, Actual Area: {actual_area_tri.item()} 1/cm")
    print(f"Max G: {torch.max(torch.abs(g_tri))} mT/m")

    # Test 2: Trapezoidal
    area_cm_trap = 0.2 # Larger area
    g_trap, t_trap = design_time_optimal_gradient(area_cm_trap, Gmax, Smax, dt)
    actual_area_trap = torch.sum(g_trap) * dt * gamma_eff_test / 100.0
    print(f"\nTrapezoidal Waveform (first 10 points): {g_trap[:10]}")
    print(f"Time vector (first 10 points): {t_trap[:10]}")
    print(f"Number of points: {g_trap.numel()}")
    print(f"Target Area: {area_cm_trap} 1/cm, Actual Area: {actual_area_trap.item()} 1/cm")
    print(f"Max G: {torch.max(torch.abs(g_trap))} mT/m")

    # Test 3: Negative Area
    area_cm_neg = -0.1
    g_neg, t_neg = design_time_optimal_gradient(area_cm_neg, Gmax, Smax, dt)
    actual_area_neg = torch.sum(g_neg) * dt * gamma_eff_test / 100.0
    print(f"\nNegative Area Waveform (first 10 points): {g_neg[:10]}")
    print(f"Number of points: {g_neg.numel()}")
    print(f"Target Area: {area_cm_neg} 1/cm, Actual Area: {actual_area_neg.item()} 1/cm")
    print(f"Max G value (should be negative): {torch.min(g_neg)} mT/m")

    # Test 4: Zero Area
    g_zero, t_zero = design_time_optimal_gradient(0.0, Gmax, Smax, dt)
    actual_area_zero = torch.sum(g_zero) * dt * gamma_eff_test / 100.0
    print(f"\nZero Area Waveform: {g_zero}")
    print(f"Time vector: {t_zero}")
    print(f"Target Area: 0.0 1/cm, Actual Area: {actual_area_zero.item()} 1/cm")

    # Test 5: Very small area (should be triangular)
    area_cm_vsmall = 1e-5
    g_vsmall, t_vsmall = design_time_optimal_gradient(area_cm_vsmall, Gmax, Smax, dt)
    actual_area_vsmall = torch.sum(g_vsmall) * dt * gamma_eff_test / 100.0
    print(f"\nVery Small Area Waveform (points): {g_vsmall.numel()}")
    print(f"Target Area: {area_cm_vsmall} 1/cm, Actual Area: {actual_area_vsmall.item()} 1/cm")
    print(f"Max G: {torch.max(torch.abs(g_vsmall))} mT/m")

    # Test 6: Area requires Gmax but is still triangular (exactly a_tri_m_inv)
    # a_tri_m_inv = gamma_eff * g_max_mT_m * (g_max_mT_m / s_max_mT_m_ms)
    # area_cm_eq_atri = (gamma_eff_test * Gmax * (Gmax / (Smax/1000.0))) / 100.0
    # print(f"Calculated a_tri_cm_inv: {area_cm_eq_atri}")
    # g_eq, t_eq = design_time_optimal_gradient(area_cm_eq_atri, Gmax, Smax, dt)
    # actual_area_eq = torch.sum(g_eq) * dt * gamma_eff_test / 100.0
    # print(f"\nArea = a_tri Waveform (points): {g_eq.numel()}")
    # print(f"Target Area: {area_cm_eq_atri} 1/cm, Actual Area: {actual_area_eq.item()} 1/cm")
    # print(f"Max G: {torch.max(torch.abs(g_eq))} mT/m")

    print("\n--- Testing calculate_b_value ---")
    # Example: Rectangular gradient
    G_rect_mT_m = 10.0  # mT/m
    T_total_ms = 10.0   # ms
    dt_rect_ms = 0.01   # ms
    
    num_points_rect = int(T_total_ms / dt_rect_ms)
    grad_rect_mT_m = torch.ones(num_points_rect) * G_rect_mT_m
    dt_rect_s = dt_rect_ms * 1e-3

    b_val_rect_calc = calculate_b_value(grad_rect_mT_m, dt_rect_s)
    print(f"Calculated b-value for rectangular gradient: {b_val_rect_calc.item()} s/mm^2")

    # Analytical b-value for rectangular pulse G over duration T_total:
    # b = gamma^2 * G^2 * T_total^3 / 3  (where G is in T/m, T_total in s)
    GAMMA_RAD_S_T_test = 2 * torch.pi * 42.577478518e6
    G_rect_T_m = G_rect_mT_m * 1e-3
    T_total_s = T_total_ms * 1e-3
    b_val_rect_analytical = (GAMMA_RAD_S_T_test**2) * (G_rect_T_m**2) * (T_total_s**3) / 3.0 * 1e-6 # to s/mm^2
    print(f"Analytical b-value for rectangular gradient: {b_val_rect_analytical} s/mm^2")

    # Example: Triangular gradient (using design_time_optimal_gradient output)
    area_cm_tri_bval = 0.05
    g_tri_bval, _ = design_time_optimal_gradient(area_cm_tri_bval, Gmax, Smax, dt_rect_ms) # dt is dt_rect_ms
    b_val_tri_calc = calculate_b_value(g_tri_bval, dt_rect_s) # g_tri_bval is in mT/m
    print(f"Calculated b-value for a triangular gradient: {b_val_tri_calc.item()} s/mm^2")

    print("\n--- Testing calculate_b_value_refocused ---")
    # Symmetric bipolar gradient
    grad_bipolar_T_m_bvr = torch.tensor([10e-3, 10e-3, -10e-3, -10e-3], dtype=torch.float32) # T/m
    dt_bipolar_s_bvr = 1e-3 # 1 ms
    
    # Inversion at midpoint (after 2ms)
    t_inv_mid_s_bvr = 2 * dt_bipolar_s_bvr 
    b_val_ref_mid = calculate_b_value_refocused(grad_bipolar_T_m_bvr, dt_bipolar_s_bvr, t_inv_mid_s_bvr)
    print(f"Refocused (mid) b-value for bipolar: {b_val_ref_mid.item()} s/mm^2")

    # Inversion after waveform
    t_inv_after_s_bvr = 5 * dt_bipolar_s_bvr
    b_val_ref_after = calculate_b_value_refocused(grad_bipolar_T_m_bvr, dt_bipolar_s_bvr, t_inv_after_s_bvr)
    b_val_no_ref = calculate_b_value(grad_bipolar_T_m_bvr * 1000, dt_bipolar_s_bvr) # convert T/m to mT/m
    print(f"Refocused (after) b-value for bipolar: {b_val_ref_after.item()} s/mm^2 (should match non-refocused)")
    print(f"Non-refocused b-value for bipolar: {b_val_no_ref.item()} s/mm^2")

    # Inversion at beginning
    t_inv_start_s_bvr = 0.0
    b_val_ref_start = calculate_b_value_refocused(grad_bipolar_T_m_bvr, dt_bipolar_s_bvr, t_inv_start_s_bvr)
    b_val_neg_no_ref = calculate_b_value(-grad_bipolar_T_m_bvr * 1000, dt_bipolar_s_bvr) # convert T/m to mT/m
    print(f"Refocused (start) b-value for bipolar: {b_val_ref_start.item()} s/mm^2 (should match non-refocused with inverted G)")
    print(f"Non-refocused b-value for inverted bipolar: {b_val_neg_no_ref.item()} s/mm^2")

    print("\n--- Testing calculate_gradient_moments ---")
    # Simple rectangular pulse
    grad_rect_T_m_mom = torch.ones(100, dtype=torch.float32) * 10e-3 # 10 T/m
    dt_s_mom = 1e-5 # 0.01 ms -> total duration 1ms
    t_total_s_mom = grad_rect_T_m_mom.numel() * dt_s_mom

    # No inversion (t_inversion_s after duration)
    m_no_inv = calculate_gradient_moments(grad_rect_T_m_mom, dt_s_mom, t_total_s_mom + dt_s_mom, num_moments=3, calculate_absolute=False)
    print(f"Rectangular (no inv) M0, M1, M2: {m_no_inv}")
    # M0 = G * T = 10e-3 * 1e-3 = 10e-6
    # M1 = G * T^2 / 2 = 10e-3 * (1e-3)^2 / 2 = 0.5 * 10e-9

    # Inversion at midpoint
    m_mid_inv = calculate_gradient_moments(grad_rect_T_m_mom, dt_s_mom, t_total_s_mom / 2.0, num_moments=3, calculate_absolute=False)
    print(f"Rectangular (mid inv) M0, M1, M2: {m_mid_inv}")
    # M0 should be 0

    # Balanced bipolar
    grad_bipolar_T_m_mom = torch.cat((torch.ones(50) * 10e-3, torch.ones(50) * -10e-3))
    # No inversion
    m_bipolar_no_inv = calculate_gradient_moments(grad_bipolar_T_m_mom, dt_s_mom, t_total_s_mom + dt_s_mom, num_moments=3, calculate_absolute=False)
    print(f"Bipolar (no inv) M0, M1, M2: {m_bipolar_no_inv}") # M0 should be 0

    # Inversion at center (makes it effectively unipolar)
    m_bipolar_mid_inv = calculate_gradient_moments(grad_bipolar_T_m_mom, dt_s_mom, t_total_s_mom / 2.0, num_moments=3, calculate_absolute=False)
    print(f"Bipolar (mid inv) M0, M1, M2: {m_bipolar_mid_inv}") # M0 should be G*T (like first case)

```python
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
    
    # Peak should be GMAX_mT_m
    torch.testing.assert_close(torch.max(torch.abs(g)), torch.tensor(GMAX_mT_m, dtype=torch.float32, device=device), rtol=1e-3, atol=1e-3)

    # Check for plateau (should be minimal or zero for triangular)
    # Count how many points are at GMAX_mT_m
    # num_plat_points = torch.sum(torch.isclose(torch.abs(g), torch.tensor(GMAX_mT_m, device=device, dtype=torch.float32), atol=1e-3))
    # This logic is tricky because normalization can shift values slightly.
    # Instead, check if it's more triangular than trapezoidal by ramp duration.
    # Expected ramp duration for full triangle:
    # n_ramp_expected = torch.ceil(torch.tensor(t_ramp_to_g_max_ms / DT_ms)).int().item()
    # Expected total points ~ 2 * n_ramp_expected - 1
    # This test is more about ensuring it handles this boundary case correctly.
    # The primary check is that the area is correct and peak is Gmax.

```
