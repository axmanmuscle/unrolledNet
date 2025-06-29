"""
model for testing the supervised learning set up
we're going to start with literally just the unet and the data consistency layer
"""
import torch
import torch.nn as nn
import numpy as np
from unet import build_unet ,build_unet_small, build_unet_smaller
from torchinfo import summary
import gc
import utils
import math_utils
from wavelet_torch import WaveletTransform

class unrolled_block(nn.Module):
    """
    we probably need A matrix free huh
    I might need to actually make that
    """
    def __init__(self, shape, wavSplit, device, dc=True):
        super(unrolled_block, self).__init__()
        self.device = device
        self.nn = build_unet_smaller(shape[1])
        self.wavSplit = wavSplit
        self.dc = dc

    def applyW(self, x, op='notransp'):
        """
        wavelet transform here
        going to assume that x is [1, 1, *sImg]
        """
        out = torch.zeros_like(x)
        if op == 'transp':
            out[..., :, :] = math_utils.iwtDaubechies2(torch.squeeze(x), self.wavSplit)
        else: 
            out[..., :, :] = math_utils.wtDaubechies2(torch.squeeze(x), self.wavSplit)

        del x
        gc.collect()
        
        return out
    
    def applyS(self, x, sMaps, op='notransp'):
        nCoils = sMaps.shape[-1]
        if op == 'transp':
            out = torch.sum( torch.conj(sMaps) * x, -1 )
        else:
            # out = torch.zeros(size=[nBatch, 1, *sMaps.shape], dtype=sMaps.dtype)
            out = torch.zeros(size = [*x.shape, nCoils], dtype=sMaps.dtype, device=self.device)
            if len(x.shape) == 4:
                sm = sMaps.unsqueeze(0)
            else:
                sm = sMaps
            for i in range(nCoils):
                out[..., i] = sm[...,i] * x

        return out

    def applyF(self, x, op='notransp'):
        if op == 'transp':
            out = torch.fft.fftshift( torch.fft.ifftn( torch.fft.ifftshift( x, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
        else:
            out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( x, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
        return out

    def grad_desc(self, x, A, b):
        """
        expecting x is the wavelet coefficients
        """
        def obj(xi):
           return 0.5 * torch.norm(A(xi) - b)**2
        
        def grad(xi):
            xt = A(xi) - b
            xt = A(xt, 'transp')
            return xt
        
        gx = grad(x)
        gxNorm = torch.norm(gx.reshape(-1, 1))**2
        alpha = 1e-4 # TODO this may need to get changed
        rho = 0.9
        c = 0.9
        max_linesearch_iters = 250
        obj_x = obj(x)

        linesearch_iter = 0
        while linesearch_iter < max_linesearch_iters:
            linesearch_iter += 1
            xNew = x - alpha*gx
            obj_xnew = obj(xNew)
            if obj_xnew < obj_x - alpha * c * gxNorm:
                break
            alpha *= rho

        # print(f'grad_descent line search finished after {linesearch_iter} iters')
        return xNew
    
    def prox(self, x, mask, b, sMaps):
        """
        apply sensitivity maps
        apply fourier transform
        replace data at mask with b
        roemer recon
        """
        nCoils = sMaps.shape[-1]
        # out = torch.zeros(size=[self.nBatch, *self.sMaps.shape], dtype=self.sMaps.dtype)
        out = torch.zeros(size = [*x.shape, nCoils], dtype=sMaps.dtype, device=self.device)
        if len(x.shape) == 4:
            sm = sMaps.unsqueeze(0)
        else:
            sm = sMaps
        for i in range(nCoils):
            out[...,i] = sm[...,i] * x

        out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( out, dim=(2,3) ), norm='ortho',\
                                                 dim = (2,3) ), dim=(2,3) )

        if len(mask.shape) < 3:
          out[..., mask, :] = b[..., mask, :]
        else:
          out[mask] = b[mask]
        
        out = torch.fft.fftshift( torch.fft.ifftn( torch.fft.ifftshift( out, dim=(2,3) ), norm='ortho',\
                                                 dim = (2,3) ), dim=(2,3) )

        out = torch.sum( torch.conj(sm) * out, -1 ) #roemer

        return out

    def forward(self, inputs):
        """
        Input will be [Nbatch x Nx x Ny]
        sMaps shouldn't change
        fourier won't change
        mask will change
        
        So if we want A = MFS, we can fix F and S at construct time
        and just vary M on the forward method
        """
        x, mask, b, sMaps = inputs # unpack
        def applyM(x):
          out = x
          if len(mask.shape) < 3:
            out[..., ~mask, :] = 0
          else:
            out[~mask] = 0
          return out

        def applyA(x, op='notransp'):
          if op == 'transp':
            out = applyM(x)
            # apply f transpose
            out = self.applyF(out, 'transp')
            # apply S transpose
            out = self.applyS(out, sMaps, 'transp')
            # wavelet transform
            out = self.applyW(out)
            
          else:
            # inverse wavelet transform
            out = self.applyW(x, 'transp')
            # apply S
            out = self.applyS(out, sMaps)
            # apply f
            out = self.applyF(out)
            # apply mask
            out = applyM(out)

          return out

        # adjoint testing
        # self.sMaps = self.sMaps.to(torch.complex128)
        # err = math_utils.test_adjoint_torch(x.to(torch.complex128), applyA)
        with torch.no_grad():
            out = self.grad_desc(x, applyA, b)

            out = self.applyW(out, 'transp')
            if self.dc:
                out = self.prox(out, mask, b, sMaps)

            # wavelets to image space
            # out = self.applyW(out, 'transp')

        out = torch.view_as_real(out)
        n = out.shape[-3]
        out_r = torch.cat((out[..., 0], out[..., 1]), dim=2)

        del out # memory management
        # gc.collect()
        torch.cuda.empty_cache()

        post_unet = self.nn(out_r)
        post_unet_r = post_unet[..., :n, :]
        post_unet_im = post_unet[..., n:, :]

        post_unet = torch.stack((post_unet_r, post_unet_im), dim=-1)
        
        del post_unet_r, post_unet_im
        # gc.collect()
        torch.cuda.empty_cache()

        out = torch.view_as_complex(post_unet)

        del post_unet
        # gc.collect()
        torch.cuda.empty_cache()

        # back to wavelet coeffs
        out = self.applyW(out)

        return out, mask, b, sMaps

class grad_desc(nn.Module):
    """
    apply gradient descent to 
    0.5 * || Ax - b ||_2^2
    so gradf(x) = A^*(Ax - b)
    A = MFS
    """
    def __init__(self, alpha, linesearch=True, w = lambda x: x, wt = lambda x: x):
        super(grad_desc, self).__init__()
        self.alpha = alpha
        self.ls = linesearch
        self.W = w
        self.Wt = wt

    def applyW(self, x, op='notransp'):
        # wx = torch.squeeze(x)

        if op == 'transp':
            x = self.Wt(x)
        else:
            x = self.W(x)

        # x = wx[None, :]
        return x

    def applyM(self, x, mask, op='notransp'):
        # x: [B, H, W, C] complex
        if mask.ndim == 2:
            mask = mask[None, ...]              # → [1, H, W]
        mask = mask[..., None]                  # → [B, H, W, 1]
        return x * mask
    
    def applyF(self, x, op='notransp'):
        # forward mode - x is [B, H, W, C] and we want the fourier transorm
        # backward mode - x is [B, H, W, C] and we want the inverse fourier transform
        if op == 'transp':
            return torch.fft.fftshift( torch.fft.ifftn( torch.fft.ifftshift( x, dim=(-3,-2) ), norm='ortho',\
                                                 dim = (-3,-2)  ), dim=(-3,-2)  )
        else:
            return torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( x, dim=(-3,-2) ), norm='ortho',\
                                                 dim = (-3,-2)  ), dim=(-3,-2)  )\
            
    def applyS(self, x, sMaps, op='notransp'):
        if op == 'transp':
            # roemer recon
            return torch.sum(torch.conj(sMaps) * x, -1) # -1 is coil dim
        else:
            # apply coil sensitivities
            # x is [B, H, W]
            # smaps are [B, H, W, C]
            return x.unsqueeze(-1) * sMaps # [B, H, W, C]

    def applyA(self, x, sMaps, mask, op='notransp'):
        if op == 'transp':
            out = self.applyM(x, mask, op)
            out = self.applyF(out, op)
            out = self.applyS(out, sMaps, op)
            out = self.applyW(out)

        else:
            out = self.applyW(x, 'transp')
            out = self.applyS(out, sMaps)
            out = self.applyF(out)
            out = self.applyM(out, mask)

        return out
    
    def forward(self, x, sMaps, mask, b):

        A = lambda x_in: self.applyA(x_in, sMaps, mask)
        At = lambda x_in: self.applyA(x_in, sMaps, mask, op='transp')

        def obj(xi):
           return 0.5 * torch.norm(A(xi) - b)**2
        
        def grad(xi):
            xt = A(xi) - b
            xt = At(xt)
            return xt

        # Ax = self.applyA(x, sMaps, mask) # [B, H, W, C]
        # resid = Ax - b # [B, H, W, C]
        # gradf = self.applyA(resid, sMaps, mask, 'transp') # [B, H, W]

        # x_new = x - self.alpha * gradf

        gx = grad(x)
        gxNorm = torch.norm(gx.reshape(-1, 1))**2
        # print(f'grad norm: {gxNorm}')
        if self.ls:
            alpha = 0.5 # TODO this may need to get changed
            rho = 0.9
            c = 0.9
            max_linesearch_iters = 250
            obj_x = obj(x)

            linesearch_iter = 0
            while linesearch_iter < max_linesearch_iters:
                linesearch_iter += 1
                xNew = x - alpha*gx
                obj_xnew = obj(xNew)
                if obj_xnew < obj_x - alpha * c * gxNorm:
                    break
                alpha *= rho

            # print(f'grad_descent line search finished after {linesearch_iter} iters at alpha {alpha}')

        else:
            xNew = x - self.alpha*gx

        return xNew

class supervised_net(nn.Module):
    def __init__(self, sImg, device, dc=True, grad=False, linesearch = True, wavelets=False, n=1, alpha=1e-3, share_weights=False):
        super(supervised_net, self).__init__()
        self.device = device
        self.dc = dc
        self.alpha = alpha
        self.ls = linesearch

        self.n = n
        self.share_weights = share_weights
        if share_weights:
            self.unet = build_unet_smaller(sImg[-1])
        else:
            self.unet_list = nn.ModuleList([build_unet_smaller(sImg[-1]) for _ in range(n)])

        self.grad = grad
        self.wav = wavelets
        if self.wav:
            self.wavSplit = torch.tensor(math_utils.makeWavSplit(sImg))
            # self.wavLevels = self.wavSplit.shape[-1]
            self.wavLevels = 4
            self.wavTrans = WaveletTransform(levels=self.wavLevels).to(self.device)
            
            def forwardW(x_in):
                x_chan = utils.complex_to_channels(x_in)
                x_out = self.wavTrans(x_chan)
                return utils.channels_to_complex(x_out)
            
            def invW(x_in):
                x_chan = utils.complex_to_channels(x_in)
                x_out = self.wavTrans.inverse(x_chan)
                return utils.channels_to_complex(x_out)
            
            self.W = forwardW
            self.Wt = invW

            self.grad_step = grad_desc(self.alpha, self.ls, w = self.W, wt = self.Wt)
        elif self.grad and not self.wav:
            self.grad_step = grad_desc(self.alpha, self.ls)

    def apply_dc(self, x, mask, b, sMaps):
        """
        just the data consistency layer
      
        apply sensitivity maps
        apply fourier transform
        replace data at mask with b
        roemer recon
        """
        # sMaps: [B, H, W, C], x: [B, H, W]
        x_exp = x.unsqueeze(-1) # [B, H, W, 1]
        coil_ims = x_exp * sMaps # [B, H, W, C]

        kSpace = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( coil_ims, dim=(-3,-2) ), norm='ortho',\
                                                 dim = (-3,-2)  ), dim=(-3,-2)  )

        # Safe masking
        if mask.ndim == 2:
            mask = mask[None, ...]  # [1, H, W]
        mask = mask[..., None]     # [B, H, W, 1]
        kSpace = kSpace * (~mask) + b * mask
        
        coil_ims_dc = torch.fft.fftshift( torch.fft.ifftn( torch.fft.ifftshift( kSpace, dim=(-3,-2) ), norm='ortho',\
                                                 dim =(-3,-2) ), dim=(-3,-2) )

        out = torch.sum( torch.conj(sMaps) * coil_ims_dc, -1 ) #roemer

        return out

    def forward(self, x, mask = None, b = None, sMaps = None):
      """
      here we assume the inputs are:
      x - [batchsize x 1 x Nx x Ny] complex image
      mask - [1 x 1 x Nx x Ny] masks of 0s and 1s
      b - [1 x nCoils x Nx x Ny] complex float measured kspace data
      sMaps - [1 x nCoils x Nx x Ny] complex float estimated sensitivity maps
      
      we split x into real and imag channels before applying unet
      """
      assert x.ndim == 3, "x at input must be [B, H, W] (complex)"
      assert b is not None and sMaps is not None and mask is not None

      for iter in range(self.n):
        if self.wav:
            # convert to wavelet coefficients
            # wx = torch.squeeze(x)
            x = self.W(x)
            # x = wx[None, :]

        # step 1: gradient descent
        if self.grad:
            # with torch.no_grad():
            xf = self.grad_step(x, sMaps, mask, b)
            # print(f'norm diff grad step: {torch.norm(x - xf)}')
            x = xf

        if self.wav:
            # wx = torch.squeeze(x)
            x = self.Wt(x)
            # x = wx[None, :]

        # step 2: (optionally) apply data consistency
        if self.dc:              
            x = self.apply_dc(x, mask, b, sMaps)

        # step 3: convert to channels and apply unet
        in_norm = x.norm(dim=(-2,-1), keepdim=True)
        x = utils.complex_to_channels(x)
        if self.share_weights:
            nn = self.unet
        else:
            nn = self.unet_list[iter]
        
        x = nn(x)
        x = utils.channels_to_complex(x)
        out_norm = x.norm(dim=(-2,-1), keepdim=True)
        x = x / (out_norm + 1e-8) * in_norm

        
      # finally do DC before output
      if self.dc:              
            x = self.apply_dc(x, mask, b, sMaps)

      return x
    
if __name__ == "__main__":
    device = torch.device("cpu")
    sMaps = torch.randn([1,640,320,16]) + 1j*torch.randn([1,640,320,16])
    # kSpace = torch.randn([1,1,640,320,16]) + 1j*torch.randn([1,1,640,320,16])
    im_in = torch.randn([1, 640, 320]) + 1j*torch.randn([1, 640, 320])
    mask = torch.randn([640,320])
    mask = mask > 0
    model = supervised_net([640, 320], device, dc=True, grad=True, linesearch = True, wavelets=False, n=10)

    print(summary(model, [im_in.shape, mask.shape, sMaps.shape, sMaps.shape], dtypes=[torch.float32, torch.bool, torch.complex64, torch.complex64], device="cpu"))