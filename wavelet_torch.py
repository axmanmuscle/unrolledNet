import torch
import torch.nn.functional as F
import torch.nn as nn

class WaveletTransform(nn.Module):
    def __init__(self, levels=1, mode='periodization', dtype=torch.float32):
        super().__init__()
        self.levels = levels
        self.mode = mode
        self.dtype = dtype

        # Initialize the transform
        self.wt = Daubechies4Transform(dtype=dtype)
        self.add_module("wt", self.wt)

    def forward(self, x):
        # Forward multi-level wavelet decomposition
        coeffs = self.wt.multilevel_dwt_2d(x, self.levels, self.mode)
        return self.wt.coeffs_to_im(coeffs, x.shape, x.dtype)

    def inverse(self, x):
        coeffs = self.wt.im_to_coeffs(x, self.levels)
        return self.wt.multilevel_idwt_2d(coeffs, self.levels, self.mode)

class Daubechies4Transform(nn.Module):
    """
    Improved Daubechies-4 wavelet transformation for PyTorch tensors.
    Uses proper filter relationships for perfect reconstruction.
    """
    
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self._get_filters()
        
    def _get_filters(self):
        """Get Daubechies-4 analysis and synthesis filters with correct relationships."""
        # Daubechies-4 scaling function coefficients (h0)
        sqrt3 = (3.0) ** 0.5
        h0_vals = [
            (1 + sqrt3) / (4 * (2.0) ** 0.5),
            (3 + sqrt3) / (4 * (2.0) ** 0.5),
            (3 - sqrt3) / (4 * (2.0) ** 0.5),
            (1 - sqrt3) / (4 * (2.0) ** 0.5)
        ]
        
        # Analysis filters
        h0 = torch.tensor(h0_vals, dtype=self.dtype)
        h1 = torch.tensor([(-1)**k * h0[3-k] for k in range(4)], dtype=self.dtype)
        
        # Synthesis filters (time-reversed analysis filters)
        g0 = h0.flip(0)  # g0[n] = h0[-n]
        g1 = h1.flip(0)  # g1[n] = h1[-n]
        
        # Reshape for conv1d: (out_channels, in_channels, kernel_length)
        self.register_buffer('h0', h0.view(1, 1, -1))
        self.register_buffer('h1', h1.view(1, 1, -1))
        self.register_buffer('g0', g0.view(1, 1, -1))
        self.register_buffer('g1', g1.view(1, 1, -1))

        return 
    
    def dwt_1d_step(self, x, mode='periodization'):
        """
        Single level 1D DWT step.
        Args:
            x: (B, C, L) input signal
            mode: boundary condition ('periodization' or 'zero')
        Returns:
            (low, high): each (B, C, L//2)
        """
        B, C, L = x.shape
        
        # Handle odd lengths by truncating
        if L % 2 == 1:
            x = x[..., :-1]
            L = L - 1
            
        # Expand filters for grouped convolution
        h0_exp = self.h0.repeat(C, 1, 1)
        h1_exp = self.h1.repeat(C, 1, 1)
        
        filter_len = self.h0.shape[-1]  # 4 for Daubechies-4
        
        if mode == 'periodization':
            # For periodization, we need to pad to ensure output length is exactly L//2
            # Pad symmetrically: (filter_len - 1) total padding
            pad_left = (filter_len - 1) // 2
            pad_right = filter_len - 1 - pad_left
            x_padded = F.pad(x, (pad_left, pad_right), mode='circular')
        else:
            # Zero padding - pad to ensure proper output size
            pad_left = filter_len - 1
            pad_right = 0
            x_padded = F.pad(x, (pad_left, pad_right), mode='constant', value=0)
            
        # Apply filters
        low = F.conv1d(x_padded, h0_exp, groups=C)
        high = F.conv1d(x_padded, h1_exp, groups=C)
        
        # Downsample by 2 - take every other sample starting from 0
        low = low[..., ::2]
        high = high[..., ::2]
        
        # Ensure we get exactly L//2 samples
        target_len = L // 2
        if low.shape[-1] > target_len:
            low = low[..., :target_len]
            high = high[..., :target_len]
        
        return low, high
    
    def idwt_1d_step(self, low, high, mode='periodization'):
        """
        Single level 1D inverse DWT step.
        Args:
            low, high: (B, C, L//2) low and high frequency components
        Returns:
            x: (B, C, L) reconstructed signal
        """
        B, C, L_half = low.shape
        
        # Upsample by inserting zeros
        low_up = torch.zeros(B, C, 2*L_half, dtype=low.dtype, device=low.device)
        high_up = torch.zeros(B, C, 2*L_half, dtype=high.dtype, device=high.device)
        
        low_up[..., ::2] = low
        high_up[..., ::2] = high
        
        # Expand synthesis filters
        g0_exp = self.g0.repeat(C, 1, 1) 
        g1_exp = self.g1.repeat(C, 1, 1)
        
        filter_len = self.g0.shape[-1]  # 4 for Daubechies-4
        
        if mode == 'periodization':
            # Circular padding for synthesis
            pad_right = (filter_len - 1) // 2
            pad_left = filter_len - 1 - pad_right
            # pad_left = (filter_len - 1) // 2
            # pad_right = filter_len - 1 - pad_left
            low_padded = F.pad(low_up, (pad_left, pad_right), mode='circular')
            high_padded = F.pad(high_up, (pad_left, pad_right), mode='circular')
        else:
            # Zero padding for synthesis
            pad_left = filter_len - 1
            pad_right = 0
            low_padded = F.pad(low_up, (pad_left, pad_right), mode='constant', value=0)
            high_padded = F.pad(high_up, (pad_left, pad_right), mode='constant', value=0)
        
        # Apply synthesis filters and sum
        recon_low = F.conv1d(low_padded, g0_exp, groups=C)
        recon_high = F.conv1d(high_padded, g1_exp, groups=C)
        
        result = recon_low + recon_high
        
        # Ensure output length is exactly 2*L_half
        target_len = 2 * L_half
        if result.shape[-1] > target_len:
            result = result[..., :target_len]
        elif result.shape[-1] < target_len:
            # Pad if too short
            pad_needed = target_len - result.shape[-1]
            result = F.pad(result, (0, pad_needed), mode='constant', value=0)
        
        return result
    
    def dwt_2d(self, x, mode='periodization'):
        """
        2D DWT using separable 1D transforms.
        Args:
            x: (B, C, H, W) input
        Returns:
            LL, LH, HL, HH: each (B, C, H//2, W//2)
        """
        B, C, H, W = x.shape
        
        # Transform along rows (width dimension)
        x_rows = x.permute(0, 2, 1, 3).reshape(B*H, C, W)
        L_rows, H_rows = self.dwt_1d_step(x_rows, mode)
        
        # Reshape back
        L_rows = L_rows.view(B, H, C, W//2).permute(0, 2, 1, 3)
        H_rows = H_rows.view(B, H, C, W//2).permute(0, 2, 1, 3)
        
        # Transform along columns (height dimension)  
        L_cols = L_rows.permute(0, 3, 1, 2).reshape(B*(W//2), C, H)
        H_cols = H_rows.permute(0, 3, 1, 2).reshape(B*(W//2), C, H)
        
        LL, LH = self.dwt_1d_step(L_cols, mode)
        HL, HH = self.dwt_1d_step(H_cols, mode)
        
        # Reshape to final form
        LL = LL.view(B, W//2, C, H//2).permute(0, 2, 3, 1)
        LH = LH.view(B, W//2, C, H//2).permute(0, 2, 3, 1)
        HL = HL.view(B, W//2, C, H//2).permute(0, 2, 3, 1)
        HH = HH.view(B, W//2, C, H//2).permute(0, 2, 3, 1)
        
        return LL, LH, HL, HH
    
    def idwt_2d(self, LL, LH, HL, HH, mode='periodization'):
        """
        2D inverse DWT.
        Args:
            LL, LH, HL, HH: (B, C, H//2, W//2) subbands
        Returns:
            x: (B, C, H, W) reconstructed image
        """
        B, C, H_half, W_half = LL.shape
        
        # Inverse transform along columns first
        L_cols = LL.permute(0, 3, 1, 2).reshape(B*W_half, C, H_half)
        H_cols = LH.permute(0, 3, 1, 2).reshape(B*W_half, C, H_half)
        L_rows = self.idwt_1d_step(L_cols, H_cols, mode)
        
        L_cols = HL.permute(0, 3, 1, 2).reshape(B*W_half, C, H_half)
        H_cols = HH.permute(0, 3, 1, 2).reshape(B*W_half, C, H_half)
        H_rows = self.idwt_1d_step(L_cols, H_cols, mode)
        
        # Reshape back
        L_rows = L_rows.view(B, W_half, C, 2*H_half).permute(0, 2, 3, 1)
        H_rows = H_rows.view(B, W_half, C, 2*H_half).permute(0, 2, 3, 1)
        
        # Inverse transform along rows
        L_final = L_rows.permute(0, 2, 1, 3).reshape(B*2*H_half, C, W_half)
        H_final = H_rows.permute(0, 2, 1, 3).reshape(B*2*H_half, C, W_half)
        
        x_recon = self.idwt_1d_step(L_final, H_final, mode)
        
        # Final reshape
        x_recon = x_recon.view(B, 2*H_half, C, 2*W_half).permute(0, 2, 1, 3)
        
        return x_recon
        
    def multilevel_dwt_2d(self, x, levels, mode='periodization'):
        """Multi-level 2D DWT."""
        coeffs = {}
        current = x
        
        for level in range(levels):
            LL, LH, HL, HH = self.dwt_2d(current, mode)
            coeffs[f'LH_{level}'] = LH
            coeffs[f'HL_{level}'] = HL  
            coeffs[f'HH_{level}'] = HH
            current = LL
            
        coeffs['LL'] = LL
        return coeffs
    
    def multilevel_idwt_2d(self, coeffs, levels, mode='periodization'):
        """Multi-level inverse 2D DWT."""
        current = coeffs['LL']
        
        for level in reversed(range(levels)):
            LH = coeffs[f'LH_{level}']
            HL = coeffs[f'HL_{level}']
            HH = coeffs[f'HH_{level}']
            current = self.idwt_2d(current, LH, HL, HH, mode)
            
        return current
    
    def coeffs_to_im(self, coeffs, in_shape, dtype=torch.float32):
      """
      Construct a visual representation of multi-level DWT coefficients.

      Places LL in the top-left, and recursively places LH, HL, HH
      in the appropriate quadrants at decreasing resolution.
      """
      B, C, H, W = in_shape
      out = torch.zeros((B, C, H, W), dtype=dtype, device=coeffs['LL'].device)

      # Start from top-left
      size = coeffs['LL'].shape[-2:]

      # Place LL block
      out[..., :size[0], :size[1]] = coeffs['LL']

      for level in reversed(range(len([k for k in coeffs if k.startswith('HH_')]))):
          # Determine size of the current level subbands
          h, w = coeffs[f'HH_{level}'].shape[-2:]

          # Place HH
          out[..., :h, size[1]:size[1]+w] = coeffs[f'LH_{level}']   # Top-right
          out[..., size[0]:size[0]+h, :w] = coeffs[f'HL_{level}']   # Bottom-left
          out[..., size[0]:size[0]+h, size[1]:size[1]+w] = coeffs[f'HH_{level}']  # Bottom-right

          # Update size to include this level’s bands
          size = (size[0] + h, size[1] + w)

      return out
    
    def im_to_coeffs(self, img, levels):
        """
        input in image space and get the coeffs in the form the inverse transform expects
        """

        B, C, H, W = img.shape
        coeffs = {}

        H_sizes = [H // (2**i) for i in range(1,levels+1)]
        W_sizes = [W // (2**i) for i in range(1,levels+1)]

        coeffs['LL'] = img[..., :H_sizes[-1], :W_sizes[-1]]

        coeffs[f'LH_{levels-1}'] = img[..., :H_sizes[-1], W_sizes[-1]:2*W_sizes[-1]]
        coeffs[f'HL_{levels-1}'] = img[..., H_sizes[-1]:2*H_sizes[-1], :W_sizes[-1]]
        coeffs[f'HH_{levels-1}'] = img[..., H_sizes[-1]:2*H_sizes[-1], W_sizes[-1]:2*W_sizes[-1]]

        for i in reversed(range(levels-1)):
            hi, wi = H_sizes[i], W_sizes[i]
            coeffs[f'LH_{i}'] = img[..., :hi, wi:2*wi]
            coeffs[f'HL_{i}'] = img[..., hi:2*hi, :wi]
            coeffs[f'HH_{i}'] = img[..., hi:2*hi, wi:2*wi]

        return coeffs


        
def test_wavelet_transform():
    """Test the wavelet transform for perfect reconstruction."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create transform object
    wt = Daubechies4Transform().to(device)
    # wt = WaveletTransform()
    
    # Test with smaller, power-of-2 sized input for cleaner results
    torch.manual_seed(42)
    x = torch.randn(1, 2, 640, 320, device=device)
    
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Input norm: {x.norm():.3f}")
    
    # Test 1D first
    print("\n=== 1D Test ===")
    x_1d = torch.randn(1, 1, 16, device=device)
    print(f"1D Input shape: {x_1d.shape}")
    
    low, high = wt.dwt_1d_step(x_1d)
    print(f"Low shape: {low.shape}, High shape: {high.shape}")
    
    x_1d_recon = wt.idwt_1d_step(low, high)
    print(f"1D Reconstruction shape: {x_1d_recon.shape}")
    
    error_1d = (x_1d - x_1d_recon).abs().mean()
    rel_error_1d = error_1d / x_1d.abs().mean()
    print(f"1D Reconstruction error: {error_1d:.2e}")
    print(f"1D Relative error: {rel_error_1d:.2e}")
    
    # Single level 2D test
    print("\n=== Single Level 2D Test ===")
    LL, LH, HL, HH = wt.dwt_2d(x)
    print(f"LL shape: {LL.shape}")
    
    x_recon = wt.idwt_2d(LL, LH, HL, HH)
    print(f"Reconstruction shape: {x_recon.shape}")
    
    error = (x - x_recon).abs().mean()
    relative_error = error / x.abs().mean()
    
    print(f"2D Reconstruction error: {error:.2e}")
    print(f"2D Relative error: {relative_error:.2e}")
    
    # Multi-level test
    print("\n=== Multi-Level Test ===")
    levels = 4  # Reduced levels for 32x32 input
    coeffs = wt.multilevel_dwt_2d(x, levels)
    
    print("Coefficient shapes:")
    for key, coeff in coeffs.items():
        print(f"  {key}: {coeff.shape}")
    
    x_recon_multi = wt.multilevel_idwt_2d(coeffs, levels)
    error_multi = (x - x_recon_multi).abs().mean()
    relative_error_multi = error_multi / x.abs().mean()
    
    print(f"Multi-level reconstruction error: {error_multi:.2e}")
    print(f"Multi-level relative error: {relative_error_multi:.2e}")
    
    return wt, x, coeffs

def test_coeffs():
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cpu')
    
    # Create transform object
    wt = Daubechies4Transform()

    # Test with smaller, power-of-2 sized input for cleaner results
    torch.manual_seed(42)
    x = torch.randn(1, 2, 640, 320, device=device, dtype=torch.float32)

    levels = 4
    coeffs = wt.multilevel_dwt_2d(x, levels)
    c1 = wt.im_to_coeffs(x, levels)
    x1 = wt.coeffs_to_im(c1, x.shape, x.dtype)

    print(torch.norm(x1 - x))
    
def test_wt():
    wt = WaveletTransform(levels=4)
    x = torch.randn(1, 2, 640, 320, dtype=torch.float32)

    wx = wt(x)
    wtx = wt.inverse(wx)

    print(torch.norm(wtx - x))

if __name__ == "__main__":
    # test_wt()
    # test_coeffs()
    wt, x, coeffs = test_wavelet_transform()