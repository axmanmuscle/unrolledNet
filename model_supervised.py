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

import logging

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

    def apply_dc_onecoil(self, coil_ks, mask, b, eps):
        """
        individual coil helper function
        coil_ks is k-space
        """
        mz = coil_ks * mask
        bmz = b - mz
        l = torch.norm(bmz)
        alpha = torch.min(torch.tensor([1, 1 - torch.sqrt(eps) / l]))

        out = coil_ks + alpha * bmz
        return out

    def apply_dc(self, x, mask, b, sMaps, eps):
        """
        apply data consistency with a nonzero epsilon
        we'll apply FS to the image to get per-coil data in k-space
        then project each individual coil to its counterpart in b via eps
        then roemer reconstruct
        """
        nCoils = sMaps.shape[-1]
        # sMaps: [B, H, W, C], x: [B, H, W]
        x_exp = x.unsqueeze(-1) # [B, H, W, 1]
        coil_ims = x_exp * sMaps # [B, H, W, C]

        kSpace = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( coil_ims, dim=(-3,-2) ), norm='ortho',\
                                                dim = (-3,-2)  ), dim=(-3,-2)  )
        
        coil_ks_dc = torch.zeros_like(coil_ims)
        for i in range(nCoils):
            coili = torch.squeeze(kSpace[..., i])
            coil_ks_dc[..., i] = self.apply_dc_onecoil(coili, mask, torch.squeeze(b[..., i]), eps[i])

        coil_ims_dc = torch.fft.fftshift( torch.fft.ifftn( torch.fft.ifftshift( coil_ks_dc, dim=(-3,-2) ), norm='ortho',\
                                                dim =(-3,-2) ), dim=(-3,-2) )

        out = torch.sum( torch.conj(sMaps) * coil_ims_dc, -1 ) #roemer

        return out
    
    def apply_dc_zero(self, x, mask, b, sMaps):
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

    def forward(self, x, mask = None, b = None, sMaps = None, eps = []):
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
      logger = logging.getLogger(__name__)

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
            if len(eps) > 0:
                x = self.apply_dc(x, mask, b, sMaps, eps)
            else:
                logger.info('applying dc zero since eps is empty')
                x = self.apply_dc_zero(x, mask, b, sMaps)

        # step 3: convert to channels and apply unet
        # in_norm = x.norm(dim=(-2,-1), keepdim=True)
        x = utils.complex_to_channels(x)
        if self.share_weights:
            nn = self.unet
        else:
            nn = self.unet_list[iter]
        
        x = nn(x)
        x = utils.channels_to_complex(x)
        # out_norm = x.norm(dim=(-2,-1), keepdim=True)
        # x = x / (out_norm + 1e-8) * in_norm

        
      # finally do DC before output
      if self.dc:              
            if len(eps) > 0:
                x = self.apply_dc(x, mask, b, sMaps, eps)
            else:
                logger.info('applying dc zero since eps is empty')
                x = self.apply_dc_zero(x, mask, b, sMaps)

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