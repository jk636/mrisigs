import torch
import numpy as np
import matplotlib.pyplot as plt

class ImageManipulation:
    def __init__(self, device=None):
        """
        Initializes ImageManipulation.
        Manages device consistency for tensors created within the class.
        """
        self.device = device if device is not None else torch.device('cpu')

    def _to_tensor(self, data, dtype=None):
        """Helper to convert data to a PyTorch tensor on self.device."""
        if not isinstance(data, torch.Tensor):
            return torch.tensor(data, dtype=dtype, device=self.device)
        return data.to(device=self.device, dtype=dtype if dtype is not None else data.dtype)

    def cropim(self, im_tensor, sx=None, sy=None):
        im_tensor = self._to_tensor(im_tensor)
        if im_tensor.ndim != 2:
            raise ValueError(f"im_tensor must be a 2D tensor. Got {im_tensor.ndim} dimensions.")
        sz = torch.tensor(im_tensor.shape, device=self.device, dtype=torch.float32)
        if sx is None: sx_calc = torch.floor(sz[1] / 2.0)
        elif sx < 1.0: sx_calc = torch.floor(sz[1] * sx)
        else: sx_calc = torch.tensor(float(sx), device=self.device, dtype=torch.float32)
        if sy is None: sy_calc = sx_calc
        elif sy < 1.0: sy_calc = torch.floor(sz[0] * sy)
        else: sy_calc = torch.tensor(float(sy), device=self.device, dtype=torch.float32)
        sx_calc, sy_calc = sx_calc.int(), sy_calc.int()
        sx_calc = torch.clamp(torch.minimum(sx_calc, sz[1].int()), min=1)
        sy_calc = torch.clamp(torch.minimum(sy_calc, sz[0].int()), min=1)
        stx = torch.clamp(torch.floor(sz[1] / 2.0 - sx_calc.float() / 2.0).int(), min=0)
        sty = torch.clamp(torch.floor(sz[0] / 2.0 - sy_calc.float() / 2.0).int(), min=0)
        endx = torch.min(stx + sx_calc, sz[1].int())
        endy = torch.min(sty + sy_calc, sz[0].int())
        im_cropped = im_tensor[sty:endy, stx:endx]
        return im_cropped

    def zpadcrop(self, im_tensor, new_shape_yx):
        im_tensor = self._to_tensor(im_tensor)
        if im_tensor.ndim != 2:
            raise ValueError(f"im_tensor must be a 2D tensor. Got {im_tensor.ndim} dimensions.")
        old_sz_yx = torch.tensor(im_tensor.shape, device=self.device, dtype=torch.long)
        new_sz_yx = self._to_tensor(new_shape_yx, dtype=torch.long)
        if new_sz_yx.numel() != 2:
            raise ValueError("new_shape_yx must be a tuple or list of two elements (rows, cols).")
        new_im = torch.zeros(new_sz_yx.tolist(), dtype=im_tensor.dtype, device=self.device)
        cz_old = torch.floor(old_sz_yx.float() / 2.0).long()
        ncz_new = torch.floor(new_sz_yx.float() / 2.0).long()
        min_sz_yx = torch.minimum(old_sz_yx, new_sz_yx)
        old_st_y = cz_old[0] - torch.floor(min_sz_yx[0].float() / 2.0).long()
        old_en_y = old_st_y + min_sz_yx[0]
        old_st_x = cz_old[1] - torch.floor(min_sz_yx[1].float() / 2.0).long()
        old_en_x = old_st_x + min_sz_yx[1]
        new_st_y = ncz_new[0] - torch.floor(min_sz_yx[0].float() / 2.0).long()
        new_en_y = new_st_y + min_sz_yx[0]
        new_st_x = ncz_new[1] - torch.floor(min_sz_yx[1].float() / 2.0).long()
        new_en_x = new_st_x + min_sz_yx[1]
        clamped_old_st_y = torch.clamp(old_st_y, min=0, max=old_sz_yx[0])
        clamped_old_en_y = torch.clamp(old_en_y, min=0, max=old_sz_yx[0])
        clamped_old_st_x = torch.clamp(old_st_x, min=0, max=old_sz_yx[1])
        clamped_old_en_x = torch.clamp(old_en_x, min=0, max=old_sz_yx[1])
        rows_to_copy = torch.clamp(clamped_old_en_y - clamped_old_st_y, min=0)
        cols_to_copy = torch.clamp(clamped_old_en_x - clamped_old_st_x, min=0)
        clamped_new_st_y = torch.clamp(new_st_y, min=0, max=new_sz_yx[0])
        clamped_new_st_x = torch.clamp(new_st_x, min=0, max=new_sz_yx[1])
        clamped_new_en_y = torch.clamp(clamped_new_st_y + rows_to_copy, min=0, max=new_sz_yx[0])
        clamped_new_en_x = torch.clamp(clamped_new_st_x + cols_to_copy, min=0, max=new_sz_yx[1])
        actual_rows_copied = clamped_new_en_y - clamped_new_st_y
        actual_cols_copied = clamped_new_en_x - clamped_new_st_x
        final_src_en_y = clamped_old_st_y + actual_rows_copied
        final_src_en_x = clamped_old_st_x + actual_cols_copied
        if actual_rows_copied > 0 and actual_cols_copied > 0:
            new_im[clamped_new_st_y:clamped_new_en_y, clamped_new_st_x:clamped_new_en_x] = \
                im_tensor[clamped_old_st_y:final_src_en_y, clamped_old_st_x:final_src_en_x]
        return new_im

    # --- New display methods ---
    def dispim(self, im_tensor, low=0.0, high=None, new_figure=True, title=None):
        im_tensor = self._to_tensor(im_tensor) # Ensure it's a tensor first
        im_np = im_tensor.cpu().numpy()
        
        # Ensure low is float
        low = float(low)

        if high is None:
            im_abs_np = np.abs(im_np) # Use abs for stats if it might be complex phase-like data
            immax = np.max(im_abs_np)
            imstd = np.std(im_abs_np)
            high_calc = immax - 0.5 * imstd
        else:
            high_calc = float(high)
        
        # Handle case where low might be >= high_calc after calculation
        if low >= high_calc:
            if high_calc > 0 : low = high_calc / 2.0 # Or some other sensible default
            elif high_calc == 0 and low == 0 : high_calc = 1.0 # Avoid vmin=vmax=0
            else : low = 0.0 # Default fallback

        if new_figure:
            plt.figure()
        
        plt.imshow(np.abs(im_np), cmap="gray", vmin=low, vmax=high_calc)
        plt.axis("off")
        if title:
            plt.title(title)
        plt.show(block=False)
        plt.pause(0.05) # Allow plot to render in non-blocking mode

    def dispkspim(self, ksp_tensor, new_figure=True):
        ksp_tensor = self._to_tensor(ksp_tensor, dtype=torch.complex64)
        
        if new_figure:
            fig, axes = plt.subplots(2, 2, figsize=(10,10)) # Adjusted size
        else:
            fig = plt.gcf()
            axes_list = fig.get_axes()
            if len(axes_list) >= 4: # Assuming they are in order or we can reshape
                axes = np.array(axes_list).reshape(2,2) # Be cautious with this assumption
            else: # Not enough axes, create new figure and axes
                fig, axes = plt.subplots(2, 2, figsize=(10,10))


        im_tensor = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(ksp_tensor, dim=(-2,-1)), dim=(-2,-1)), dim=(-2,-1))
        
        ksp_np = ksp_tensor.cpu().numpy()
        im_np = im_tensor.cpu().numpy()
        
        log_ksp_np = 10 * np.log10(np.abs(ksp_np) + 1e-9)
        khigh = np.max(log_ksp_np) - 0.5 * np.std(log_ksp_np)
        klow = np.min(log_ksp_np) # Use actual min for log ksp
        if klow >= khigh: klow = khigh -1 # ensure klow < khigh

        im_abs_np = np.abs(im_np)
        imhigh = np.max(im_abs_np) - 0.5 * np.std(im_abs_np)
        if 0 >= imhigh: imhigh = np.max(im_abs_np) # if std is large or max is 0

        axes[0,0].imshow(log_ksp_np, cmap="gray", vmin=klow, vmax=khigh)
        axes[0,0].set_title('k-space log-magnitude')
        axes[0,0].axis("off")
        
        axes[0,1].imshow(np.angle(ksp_np), cmap="gray")
        axes[0,1].set_title('k-space phase')
        axes[0,1].axis("off")
        
        axes[1,0].imshow(im_abs_np, cmap="gray", vmin=0, vmax=imhigh)
        axes[1,0].set_title('Image Magnitude')
        axes[1,0].axis("off")
        
        axes[1,1].imshow(np.angle(im_np), cmap="gray")
        axes[1,1].set_title('Image Phase')
        axes[1,1].axis("off")
        
        plt.show(block=False)
        plt.pause(0.05)
        return im_tensor

    def dispangle(self, complex_tensor, new_figure=True):
        complex_tensor = self._to_tensor(complex_tensor, dtype=torch.complex64)
        angle_tensor = torch.angle(complex_tensor) + torch.pi # Shift to [0, 2*pi]
        self.dispim(angle_tensor, low=0.0, high=2.0 * np.pi, new_figure=new_figure, title="Phase Image [0, 2pi]")

    def displogim(self, im_tensor, low_exp_thresh=None, new_figure=True):
        # low_exp_thresh is not directly used in this simplified version,
        # but could be used to clamp abs(im_tensor) before log.
        im_tensor = self._to_tensor(im_tensor)
        
        abs_im = torch.abs(im_tensor)
        if low_exp_thresh is not None:
            abs_im = torch.clamp(abs_im, min=float(low_exp_thresh))
            
        log_im_tensor = torch.log(abs_im + 1e-9) # Add epsilon for log(0)
        
        # Let dispim determine high, but set low based on min of log data
        min_val = torch.min(log_im_tensor).item()
        self.dispim(log_im_tensor, low=min_val, high=None, new_figure=new_figure, title="Log-Magnitude Image")


if __name__ == '__main__':
    plt.ioff() # Turn off interactive mode for automated testing
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {test_device} for tests ---")
    manipulator = ImageManipulation(device=test_device)

    # --- Test cropim (existing) ---
    print("\n--- Testing cropim ---")
    test_image_c = torch.arange(100, device=test_device).reshape(10, 10).float()
    manipulator.cropim(test_image_c) 
    print("cropim called.")

    # --- Test zpadcrop (existing) ---
    print("\n--- Testing zpadcrop ---")
    manipulator.zpadcrop(test_image_c, (6,6))
    print("zpadcrop called.")

    # --- New tests for display methods ---
    print("\n--- Testing dispim ---")
    mag_image = torch.abs(torch.randn(64, 64, device=test_device))
    manipulator.dispim(mag_image, title="Test Magnitude Image")
    print("dispim called.")
    plt.close('all')


    print("\n--- Testing dispkspim ---")
    # Create a simple k-space (e.g., point at center)
    ksp_data = torch.zeros(64, 64, dtype=torch.complex64, device=test_device)
    ksp_data[32, 32] = 1.0
    reconstructed_img = manipulator.dispkspim(ksp_data)
    assert reconstructed_img.shape == ksp_data.shape, "dispkspim output shape mismatch"
    print("dispkspim called.")
    plt.close('all')


    print("\n--- Testing dispangle ---")
    complex_image = torch.randn(64, 64, dtype=torch.complex64, device=test_device)
    manipulator.dispangle(complex_image)
    print("dispangle called.")
    plt.close('all')


    print("\n--- Testing displogim ---")
    manipulator.displogim(mag_image)
    print("displogim called.")
    # Test with low_exp_thresh
    manipulator.displogim(mag_image, low_exp_thresh=0.1)
    print("displogim with low_exp_thresh called.")
    plt.close('all')
    

    print("\nAll tests for ImageManipulation (including display methods) seem to run.")
    # Note: Visual output cannot be verified here, only lack of runtime errors.
    # For actual visual inspection, run in an environment that supports Matplotlib GUI.
    # plt.ion() # Turn interactive mode back on if needed at end of a script
    # plt.show() # To show all figures if block=False was used throughout.
