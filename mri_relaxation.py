import torch

class MRIRelaxation:
    def relax(self, t, T1=4.0, T2=0.1, combine=False):
        """
        Calculates the relaxation matrix A and vector B.

        Args:
            t (float or torch.Tensor): Interval over which relaxation is being evaluated.
            T1 (float or torch.Tensor): Longitudinal relaxation time.
            T2 (float or torch.Tensor): Transverse relaxation time.
            combine (bool): If True, stacks A and B horizontally.

        Returns:
            If combine is True:
                torch.Tensor: Combined matrix A and B (3x4).
            If combine is False:
                tuple[torch.Tensor, torch.Tensor]: Matrix A (3x3) and vector B (3x1).
        """
        # Convert inputs to PyTorch float tensors
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        elif t.dtype != torch.float32:
            t = t.float()

        if not isinstance(T1, torch.Tensor):
            T1 = torch.tensor(T1, dtype=torch.float32)
        elif T1.dtype != torch.float32:
            T1 = T1.float()

        if not isinstance(T2, torch.Tensor):
            T2 = torch.tensor(T2, dtype=torch.float32)
        elif T2.dtype != torch.float32:
            T2 = T2.float()

        # Ensure t, T1, T2 are scalar-like (0-dim or 1-element tensors) for this formulation
        # If they are tensors with more than one element, this specific matrix formulation
        # might not be what's intended, as E1 and E2 would become vectors.
        # For now, proceeding with the assumption they result in scalar E1, E2.
        if t.numel() != 1 or T1.numel() != 1 or T2.numel() != 1:
            # This warning can be adjusted based on expected behavior for non-scalar t
            # print("Warning: t, T1, or T2 have more than one element. "
            #       "E1 and E2 will be calculated element-wise, "
            #       "but only the first element will be used for matrix A and vector B construction "
            #       "if they are not already scalars.")
            # To ensure E1 and E2 are scalars, we might take .item() or ensure inputs are scalar.
            # For now, let's assume inputs are such that E1 and E2 become scalar after computation.
            # If t, T1, T2 are multi-element, exp will be element-wise.
            # The original np.diag([E2, E2, E1]) implies E2 and E1 are scalars.
            # Let's ensure this by taking the first element if they are not 0-dim.
            # This behavior should be clarified if t can be an array for this function.
            pass


        E1 = torch.exp(-t / T1)
        E2 = torch.exp(-t / T2)

        # If E1 or E2 are tensors with more than one element due to broadcasting or multi-element t, T1, T2
        # we need to ensure they are scalar for torch.diag and B vector construction as per original code.
        # Example: if t = torch.tensor([1.0, 2.0]), E1 will be a 2-element tensor.
        # The original code np.diag([E2, E2, E1]) would fail if E1, E2 are not scalars.
        # Assuming t, T1, T2 are effectively scalars for this matrix math.
        # If t is an array, the concept of "the" relaxation matrix A becomes more complex.
        # Sticking to the most direct interpretation: t, T1, T2 are scalars or treated as such.
        
        if E1.numel() > 1: E1 = E1[0] # Take the first element if not scalar
        if E2.numel() > 1: E2 = E2[0] # Take the first element if not scalar


        A = torch.diag(torch.stack([E2, E2, E1]))

        # Ensure B_scalar_term is a scalar for the tensor constructor
        B_scalar_term = 1.0 - E1
        if not isinstance(B_scalar_term, float): # if E1 was a tensor, B_scalar_term might be too
            B_scalar_term = B_scalar_term.item() 

        B = torch.tensor([0., 0., B_scalar_term], dtype=torch.float32)
        B = B.reshape((3, 1))

        if combine:
            # A is 3x3, B is 3x1. Concatenate along dimension 1 to make A 3x4.
            A_combined = torch.cat((A, B), dim=1)
            return A_combined
        else:
            return A, B

if __name__ == '__main__':
    # Example Usage (optional - for testing during development)
    relaxer = MRIRelaxation()

    # Test case 1: combine=False (default inputs for T1, T2)
    t_val = 0.05 # example time interval
    A_matrix, B_vector = relaxer.relax(t_val)
    print(f"--- Test Case 1: t={t_val}, combine=False ---")
    print("Matrix A:\n", A_matrix)
    print("Vector B:\n", B_vector)
    print("A shape:", A_matrix.shape)
    print("B shape:", B_vector.shape)

    # Test case 2: combine=True
    A_combined_matrix = relaxer.relax(t_val, T1=3.0, T2=0.08, combine=True)
    print(f"\n--- Test Case 2: t={t_val}, T1=3.0, T2=0.08, combine=True ---")
    print("Combined Matrix (A|B):\n", A_combined_matrix)
    print("Combined Matrix shape:", A_combined_matrix.shape)

    # Test case 3: Using tensor inputs
    t_tensor = torch.tensor(0.1)
    T1_tensor = torch.tensor(4.5)
    T2_tensor = torch.tensor(0.15)
    A_matrix_tensor_input, B_vector_tensor_input = relaxer.relax(t_tensor, T1=T1_tensor, T2=T2_tensor)
    print(f"\n--- Test Case 3: Tensor inputs, t={t_tensor.item()}, combine=False ---")
    print("Matrix A:\n", A_matrix_tensor_input)
    print("Vector B:\n", B_vector_tensor_input)

    # Test case 4: Scalar inputs for T1, T2 for diag
    t_val_s = 0.05
    E1_s = torch.exp(-torch.tensor(t_val_s) / torch.tensor(4.0))
    E2_s = torch.exp(-torch.tensor(t_val_s) / torch.tensor(0.1))
    print(f"\n--- Debug: E1_s ({E1_s.shape}), E2_s ({E2_s.shape}) from scalar t, T1, T2 ---")
    A_diag_test = torch.diag(torch.stack([E2_s, E2_s, E1_s])) # torch.stack needed if E1, E2 are 0-dim
    print("A_diag_test:\n", A_diag_test)
    
    # What if t is a multi-element tensor?
    # The current implementation will use the first element of t, T1, T2 if they are not scalar.
    # This is based on the interpretation of np.diag([E2, E2, E1]) requiring scalar E1, E2.
    # If element-wise matrices are needed, the function signature or logic would need to change.
    print("\n--- Example with non-scalar t (current behavior: uses t[0]) ---")
    # t_array = torch.tensor([0.05, 0.1])
    # A_arr, B_arr = relaxer.relax(t_array) # This would cause issues if not handled
    # print("A from t_array[0]:\n", A_arr) # This shows behavior if E1, E2 are forced scalar

    # Correct handling of torch.diag with 0-dim tensors:
    # torch.diag(torch.tensor([scalar1, scalar2, scalar3])) works.
    # torch.diag(torch.stack([scalar_tensor1, scalar_tensor2, scalar_tensor3])) also works.
    # My use of torch.stack([E2, E2, E1]) is robust for E1, E2 being 0-dim tensors.

    print("\nScript finished.")
