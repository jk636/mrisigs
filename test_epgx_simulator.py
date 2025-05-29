import torch
from epgx_simulator import EPGXSimulator

def test_met_simulation():
    """
    Based on epgx/Test_MET.m
    Tests the EPGXSimulator with model_type='full' using epgx_cpmg.
    """
    print("\n--- Running test_met_simulation ---")
    # Define parameters
    f = 0.20
    ka_per_s = 5e-2  # Assuming ka is per second as typically used in these models
    T1_list_ms = [1000.0, 500.0]
    T2_list_ms = [100.0, 20.0]
    esp_ms = 5.0
    # angles_rad = torch.ones(50) * torch.pi # epgx_cpmg expects degrees
    angles_deg = torch.ones(50, dtype=torch.float32) * 180.0
    deltab_hz = 0.0

    # Instantiate simulator
    # Assuming 'full' model as Test_MET.m implies epg_X_CMPG.m which returns 2 signals (water and MET)
    simulator = EPGXSimulator(model_type='full', device='cpu')

    # Call epgx_cpmg
    signals = simulator.epgx_cpmg(
        flipangle_series_deg=angles_deg,
        esp_ms=esp_ms,
        T1_list_ms=T1_list_ms,
        T2_list_ms=T2_list_ms,
        ka_per_s=ka_per_s,
        f=f,
        deltab_hz=deltab_hz
    )

    # Print results
    print("MET signals shape:", signals.shape)
    if signals.numel() > 0:
        print("MET 1st compartment (abs, first 5):", torch.abs(signals[0, :5]))
        print("MET 2nd compartment (abs, first 5):", torch.abs(signals[1, :5]))
    else:
        print("MET signals tensor is empty.")

def test_mt_saturation_simulation():
    """
    Based on epgx/test_saturation_MT.m
    Tests the EPGXSimulator with model_type='MT' using epgx_rfspoil and varying WT.
    """
    print("\n--- Running test_mt_saturation_simulation ---")
    # Define parameters
    TR_ms = 40.0
    alpha_deg = 15.0
    phi0_rad = 117.0 * torch.pi / 180.0  # Phase increment in radians
    T1_list_ms = [1200.0, 1200.0] # T1a, T1b
    f_MT = 0.35
    k_MT_per_s = 4.3e-3 # This is ka for the MT model (exchange from free to bound)
    
    # T2b for MT model is typically for lineshape, not direct relaxation of Zb.
    # epgx_relax for MT model uses T2a for free pool transverse relaxation.
    # We'll pass [T2a, T2b_model_parameter] to T2_list_ms.
    # T2b_model_parameter might be used if epgx_relax's MT part evolves to include it for Zb.
    # For now, only T2a (T2_list_ms[0]) is used for F states in MT model.
    T2_list_ms = [350.0, 12e-3] # T2a_ms, T2b_superlorentzian_ms (12 microseconds = 0.012 ms)
    
    num_pulses = 50  # Reduced from 300 for faster testing
    
    # Calculate RF phase series in degrees
    rf_phase_series_deg = torch.fmod(torch.arange(num_pulses, dtype=torch.float32) * (phi0_rad * 180.0 / torch.pi), 360.0)
    deltab_hz = 0.0 # Off-resonance of the free pool (water)

    # Instantiate simulator
    simulator = EPGXSimulator(model_type='MT', device='cpu')

    # Simplified WT calculation: loop through a few fixed values
    WT_values = [0.1, 0.5, 1.0, 5.0] # Representing different saturation levels

    for WT in WT_values:
        print(f"\nTesting with WT = {WT}")
        # Call epgx_rfspoil
        s = simulator.epgx_rfspoil(
            flipangle_deg=alpha_deg,
            rf_phase_series_deg=rf_phase_series_deg,
            TR_ms=TR_ms,
            T1_list_ms=T1_list_ms,
            T2_list_ms=T2_list_ms,
            ka_per_s=k_MT_per_s,
            f=f_MT,
            WT=WT,
            deltab_hz=deltab_hz
        )

        # Print results
        print(f"MT saturation signals shape (WT={WT}):", s.shape)
        if s.numel() > 0:
            print(f"MT final signal (abs, WT={WT}):", torch.abs(s[0, -1]))
        else:
            print(f"MT signals tensor (WT={WT}) is empty.")

if __name__ == '__main__':
    print("Starting EPGXSimulator tests...")
    test_met_simulation()
    test_mt_saturation_simulation()
    print("\nEPGXSimulator tests finished.")
