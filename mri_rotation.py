import torch

class MRIRotation:
    def _to_tensor(self, value):
        """Helper to convert value to a float32 tensor if it's not already."""
        if not isinstance(value, torch.Tensor):
            return torch.tensor(value, dtype=torch.float32)
        return value.float()

    def xrot(self, angle_deg=0.0):
        """
        Returns a 3x3 PyTorch tensor for left-handed rotation about the x-axis.
        Input angle_deg: Rotation angle in degrees.
        """
        angle_deg = self._to_tensor(angle_deg)
        angle_rad = angle_deg * torch.pi / 180.0
        
        c = torch.cos(angle_rad)
        s = torch.sin(angle_rad)
        
        M = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, c.item(), s.item()],
            [0.0, -s.item(), c.item()]
        ], dtype=torch.float32)
        return M

    def yrot(self, angle_deg=0.0):
        """
        Returns a 3x3 PyTorch tensor for left-handed rotation about the y-axis.
        Input angle_deg: Rotation angle in degrees.
        """
        angle_deg = self._to_tensor(angle_deg)
        angle_rad = angle_deg * torch.pi / 180.0
        
        c = torch.cos(angle_rad)
        s = torch.sin(angle_rad)
        
        M = torch.tensor([
            [c.item(), 0.0, -s.item()],
            [0.0, 1.0, 0.0],
            [s.item(), 0.0, c.item()]
        ], dtype=torch.float32)
        return M

    def zrot(self, angle_deg=0.0):
        """
        Returns a 3x3 PyTorch tensor for left-handed rotation about the z-axis.
        Input angle_deg: Rotation angle in degrees.
        """
        angle_deg = self._to_tensor(angle_deg)
        angle_rad = angle_deg * torch.pi / 180.0
        
        c = torch.cos(angle_rad)
        s = torch.sin(angle_rad)
        
        M = torch.tensor([
            [c.item(), s.item(), 0.0],
            [-s.item(), c.item(), 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)
        return M

    def throt(self, theta_deg=0.0, phi_deg=0.0):
        """
        Returns a 3x3 PyTorch tensor for rotation about an axis 
        in the x-y plane, phi_deg away from x, by theta_deg.
        Inputs theta_deg, phi_deg: Rotation angles in degrees.
        Uses the formula from the original mrsigpy.py.
        """
        theta_deg = self._to_tensor(theta_deg)
        phi_deg = self._to_tensor(phi_deg)

        theta_rad = theta_deg * torch.pi / 180.0
        phi_rad = phi_deg * torch.pi / 180.0
        
        ca = torch.cos(theta_rad)
        sa = torch.sin(theta_rad)
        cp = torch.cos(phi_rad)
        sp = torch.sin(phi_rad)
        
        # Using .item() to ensure scalar values for tensor construction
        # if inputs were scalar. If inputs are tensors, this will error.
        # The expectation for rotation matrices is usually scalar angles.
        # If angle_deg were a tensor, c and s would be tensors.
        # torch.tensor([[...]]) expects Python numbers or tensors that can be converted.
        # Let's ensure ca, sa, cp, sp are Python floats if they are 0-dim tensors.
        
        ca_val = ca.item() if ca.numel() == 1 else ca 
        sa_val = sa.item() if sa.numel() == 1 else sa
        cp_val = cp.item() if cp.numel() == 1 else cp
        sp_val = sp.item() if sp.numel() == 1 else sp

        M = torch.tensor([
            [cp_val*cp_val + sp_val*sp_val*ca_val, cp_val*sp_val*(1-ca_val), -sp_val*sa_val],
            [cp_val*sp_val - sp_val*cp_val*ca_val, sp_val*sp_val + cp_val*cp_val*ca_val, cp_val*sa_val], # Direct translation from mrsigpy.py
            [sa_val*sp_val, -sa_val*cp_val, ca_val]
        ], dtype=torch.float32)
        return M

if __name__ == '__main__':
    # Example Usage (optional - for testing during development)
    rotator = MRIRotation()

    print("--- Testing xrot ---")
    x_rot_90 = rotator.xrot(angle_deg=90.0)
    print("xrot(90):\n", x_rot_90)
    
    x_rot_0 = rotator.xrot(angle_deg=0.0)
    print("xrot(0):\n", x_rot_0) # Should be identity in y-z
    
    print("\n--- Testing yrot ---")
    y_rot_90 = rotator.yrot(angle_deg=90.0)
    print("yrot(90):\n", y_rot_90)

    print("\n--- Testing zrot ---")
    z_rot_90 = rotator.zrot(angle_deg=90.0)
    print("zrot(90):\n", z_rot_90)

    print("\n--- Testing throt ---")
    # Test case 1: phi=0 (rotation around x-axis by theta)
    # Should be equivalent to xrot(theta_deg)
    th_rot_phi0 = rotator.throt(theta_deg=90.0, phi_deg=0.0)
    print("throt(theta=90, phi=0):\n", th_rot_phi0)
    print("Compare with xrot(90):\n", rotator.xrot(90))

    # Test case 2: theta=0 (should be identity matrix)
    th_rot_theta0 = rotator.throt(theta_deg=0.0, phi_deg=45.0)
    print("throt(theta=0, phi=45):\n", th_rot_theta0) # Should be identity

    # Test case 3: General case
    th_rot_general = rotator.throt(theta_deg=30.0, phi_deg=45.0)
    print("throt(theta=30, phi=45):\n", th_rot_general)

    # Test with tensor inputs (e.g., for batch operations, though matrices here are single)
    # The .item() calls would need adjustment if inputs are multi-element tensors
    # and element-wise matrix generation is expected.
    # Current setup assumes scalar angle inputs for single matrix output.
    # If angle_deg = torch.tensor([90.0, 45.0]), c and s would be tensors.
    # Then c.item() would fail.
    # The helper _to_tensor ensures it's a tensor, but .item() in matrix construction
    # assumes it's a 0-dim tensor (scalar). This is typical for rotation matrix functions.
    
    # Re-evaluating .item() usage:
    # If angle_deg is a scalar float (e.g. 90.0), _to_tensor converts to 0-dim tensor.
    # angle_rad becomes 0-dim. c, s become 0-dim. c.item() is fine.
    # This design is okay for scalar inputs.
    
    print("\nScript finished.")
