# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:39:31 2019

@authors: Joshua Kaggie, Brian Hargreaves
"""
import numpy as np
import matplotlib.pyplot as plt
# import mrsigpy as mrs # Removed

import torch
from mri_relaxation import MRIRelaxation
from mri_rotation import MRIRotation
import plotting_utils 
# Removed numpy and specific matplotlib imports that are covered or unused
# from numpy import mean, std, exp, diag, matmul, pi, cos, sin, zeros, ones, shape, floor, ceil
# from maplotlib.pyplot import figure, plot, xlabel, ylabel, title, imshow, subplot 

# Define global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conceptB4_2(sigmean = 0):
    # -- Show distributions Gaussian, Rician (& Rayleigh)
    # This function is not part of the current refactoring task.
    # Original plotting code was complex and relied on potentially undefined helper functions.
    # Kept as a placeholder.
    pass


# def exampleA1_63():
#     # Plot of Mz for IR Question.    
#     t = np.arange(0, 1, .01)
#     Mz = -1.63*np.exp(-t/0.5)+1;
#     Mz[51:] = 1-np.exp(-(t[51:]-.5)/.5);
#     plt.plot(t,Mz)
#     mrs.lplot('Time (s)','M_z / M_0','Simple IR Signal Example',[0, 1, -1, 1]);
#     mrs.setprops()

def exampleA1_63():
    # Plot of Mz for IR Question.
    t = torch.arange(0, 1, .01, device=device, dtype=torch.float32)
    Mz = torch.zeros_like(t)
    
    # Part 1: Mz = -1.63*exp(-t/0.5)+1
    exp_term1 = torch.exp(-t / 0.5)
    Mz_part1 = -1.63 * exp_term1 + 1.0
    
    # Part 2: Mz = 1-exp(-(t-0.5)/0.5) for t >= 0.5
    # Find the split index more robustly
    split_index = torch.searchsorted(t, 0.5).item() # .item() to get Python int

    Mz[:split_index] = Mz_part1[:split_index]
    
    if split_index < t.shape[0]: # If there are any time points >= 0.5
        t_for_part2 = t[split_index:]
        # Mz_part2 calculation needs to align with the length of Mz[split_index:]
        t_offset_part2 = t_for_part2 - 0.5 
        exp_term2 = torch.exp(-t_offset_part2 / 0.5)
        Mz_part2_calculated = 1.0 - exp_term2
        Mz[split_index:] = Mz_part2_calculated

    plt.figure() 
    plt.plot(t.cpu().numpy(), Mz.cpu().numpy())
    plotting_utils.lplot('Time (s)', 'M_z / M_0', 'Simple IR Signal Example (PyTorch)', ax_limits=[0, 1, -1, 1])
    plotting_utils.setprops()
    plt.show(block=False)
    plt.pause(0.1)


# def exampleB1_13(TR = 1, TI = 0.5, TE = 0.05, T1 = 0.5, T2 = 0.1):
#     #%	Short-TR IR Signal Example
#     # ... (original numpy code commented out) ...

# def exampleB1_15(TR = 1, TI = 0.5, TE = 0.05, T1 = 0.5, T2 = 0.1):
#     #%	Short-TR IR Signal Example
#     # ... (original numpy code commented out) ...

def exampleB1_15(TR=1, TI=0.5, TE=0.05, T1=0.5, T2=0.1):
    # Uses new relaxation and rotation modules to replicate abprop logic
    # Assuming MRIRelaxation and MRIRotation constructors were updated to accept `device`
    relaxation_module = MRIRelaxation() # device=device) # If constructor takes device
    rotation_module = MRIRotation()   # device=device) # If constructor takes device

    ops = [
        ('relax', {'t': TE, 'T1': T1, 'T2': T2}), 
        ('rotate', {'angle_deg': 90}),
        ('relax', {'t': TI, 'T1': T1, 'T2': T2}),
        ('rotate', {'angle_deg': 180}),
        ('relax', {'t': TR - TE - TI, 'T1': T1, 'T2': T2})
    ]
    ops.reverse() 

    A_total = torch.eye(3, device=device, dtype=torch.complex64)
    B_total = torch.zeros((3, 1), device=device, dtype=torch.complex64)

    for op_type, params in ops:
        A_op_uncast = None
        B_op_uncast = None
        if op_type == 'relax':
            # relax returns combined A (3x4) if combine=True
            AB_op_combined = relaxation_module.relax(t=params['t'], T1=params['T1'], T2=params['T2'], combine=True)
            A_op_uncast = AB_op_combined[:, :3]
            B_op_uncast = AB_op_combined[:, 3].unsqueeze(1)
        elif op_type == 'rotate':
            # xrot returns 3x3 rotation matrix
            A_op_uncast = rotation_module.xrot(angle_deg=params['angle_deg']) 
            B_op_uncast = torch.zeros((3, 1), device=device, dtype=torch.float32) # B is float initially
        else:
            raise ValueError(f"Unknown operation type: {op_type}")

        A_op = A_op_uncast.to(dtype=torch.complex64, device=device)
        B_op = B_op_uncast.to(dtype=torch.complex64, device=device)

        A_total = A_op @ A_total
        B_total = A_op @ B_total + B_op
            
    I_minus_A = torch.eye(3, device=device, dtype=torch.complex64) - A_total
    
    try:
        Mss = torch.linalg.solve(I_minus_A, B_total)
    except Exception as e:
        # Check for singularity explicitly for better message
        if torch.det(I_minus_A).abs() < 1e-9:
             print("Warning: Matrix (I - A_total) is singular or near-singular. Using pseudo-inverse (lstsq).")
        else:
            print(f"linalg.solve failed: {e}. Using pseudo-inverse (lstsq).")
        Mss = torch.linalg.lstsq(I_minus_A, B_total).solution

    return Mss


# --- Placeholder for other functions from original examples.py ---
def exampleB1_13(TR = 1, TI = 0.5, TE = 0.05, T1 = 0.5, T2 = 0.1):
    print("exampleB1_13 is not refactored in this step.")
    pass

def exampleB1_17():
    print("exampleB1_17 is not refactored in this step.")
    pass

def exampleB1_19():
    print("exampleB1_19 is not refactored in this step.")
    pass

def exampleB2_2(D = 1e-6, G = 0.04, T = 10, gamma = 42.58, dt = 20):
    print("exampleB2_2 is not refactored in this step.")
    pass

def exampleB2_23(T=1, T2 = 9.4877, T1 = 19.4932, Q = None): 
    if Q is None: Q = torch.tensor([[0j],[0j],[1+0j]], device=device, dtype=torch.complex64) 
    print("exampleB2_23 is not refactored in this step.")
    pass

def exampleB2_43():
    print("exampleB2_43 is not refactored in this step.")
    pass

def exampleB2_5(G=40, T= 1.174, gamma = 42.58, alpha = None): 
    if alpha is None: alpha = np.arange(180,60,-1) # Keep np.arange if alpha is for looping/display not tensor math
    print("exampleB2_5 is not refactored in this step.")
    pass
    
def exampleB4_14(R=2, Na = 1000, cwid = 25):
    print("exampleB4_14 is not refactored in this step.")
    pass

def exampleE1_vds():
    print("exampleE1_vds is not refactored in this step.")
    pass

def exampleE3_spiral():
    print("exampleE3_spiral is not refactored in this step.")
    pass

def sense1d():
    print("sense1d is not refactored in this step.")
    pass
    
def examplenoise():
    print("examplenoise is not refactored in this step.")
    pass

if __name__ == '__main__':
    plotting_utils.setprops() # Set global plot properties once
    print(f"Using device: {device}")

    print("\n--- Running PyTorch exampleA1_63 ---")
    exampleA1_63()
    
    print("\n--- Running PyTorch exampleB1_15 ---")
    Mss_pytorch = exampleB1_15(TR=1, TI=0.5, TE=0.05, T1=0.5, T2=0.1)
    print("Mss from PyTorch exampleB1_15:\n", Mss_pytorch.cpu().numpy())
    assert Mss_pytorch.shape == (3,1), f"Mss_pytorch shape is {Mss_pytorch.shape}, expected (3,1)"
    print("exampleB1_15 finished.")

    # Attempt to show all non-blocking figures at the end, then close
    try:
        plt.show() # This will process all figures created with block=False
    except Exception as e:
        print(f"Final plt.show() failed or not applicable: {e}")
    
    plt.close('all') # Ensure all figures are closed after script execution
    print("\nExamples finished.")
