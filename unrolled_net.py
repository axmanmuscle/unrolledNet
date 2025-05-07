"""
formulation of the unrolled network for supervised learning

In this case we'll unroll proximal gradient descent and replace the proximal steps with a neural network (?)
in fact we're going to do FISTA, which is proximal gradient descent with momentum

FISTA solves problems of the form 
min f(x) + g(x)
 x
where f is smooth and g has a simple proximal operator
FISTA iterations:
y_k+1 = y_k - alpha_k(gradf(y_k))
z_k+1 = prox_(gamma_k g)(y_k+1)
t_k+1 = 0.5 * (1 + sqrt(1 + 4t_k^2))
x_k+1 = z_k+1 + (t_k - 1)/t_k+1 * (z_k+1 - z_k)

we're going to solve
min 0.5 * || Ax - b ||_2^2 st || Ax - b ||^2 <= eps

with A = MFS
Seems strange but the grad descent will impact all of the image x and the constraint will enforce data consistency
"""
import torch
import torch.nn as nn
import numpy as np
from unet import build_unet ,build_unet_small
from torchinfo import summary
import gc
import utils
import math_utils

class prox_block(nn.Module):
    """
    just a module for the proximal block
    """

    def __init__(self, device, wavSplit):
        super(prox_block, self).__init__()
        self.device = device
        self.wavSplit = wavSplit

    def forward(self, inputs):
        """
        forward method for prox block
        simply to do a proximal step at the very end for data consistency
        """
        x, mask, b, sMaps = inputs

        nCoils = sMaps.shape[-1]
        # wavelet coefficients to image space
        x = math_utils.iwtDaubechies2(torch.squeeze(x), self.wavSplit)

        x = x[None, None, :, :]

        out = torch.zeros(size = [*x.shape, nCoils], dtype=sMaps.dtype, device = self.device)

        if len(x.shape) == 4:
            sm = sMaps.unsqueeze(0)
        else:
            sm = sMaps

        for i in range(nCoils):
            out[...,i] = sm[...,i] * x

        out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( out, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )

        if len(mask.shape) < 3:
          out[..., mask, :] = b[..., mask, :]
        else:
          out[mask] = b[mask]

        out = torch.fft.fftshift( torch.fft.ifftn( torch.fft.ifftshift( out, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
        out = torch.sum( torch.conj(sm) * out, -1 ) #roemer

        #out = torch.view_as_real(out)

        return out


class final_block_nodc(nn.Module):
    """
    final block for the wavelet network
    """

    def __init__(self, device, wavSplit):
        super(final_block_nodc, self).__init__()
        self.device = device
        self.wavSplit = wavSplit

    def forward(self, inputs):
        """
        forward method for the final block
        just return the wavelet coefficients to image space
        """
        x, mask, b = inputs

        # wavelet coefficients to image space
        x = math_utils.iwtDaubechies2(torch.squeeze(x), self.wavSplit)

        return x

class unrolled_block(nn.Module):
    """
    we probably need A matrix free huh
    I might need to actually make that
    """
    def __init__(self, shape, wavSplit, device, dc=True):
        super(unrolled_block, self).__init__()
        self.device = device
        self.nn = build_unet(shape[1])
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
        out = self.grad_desc(x, applyA, b)

        if self.dc:
            out = self.prox(out, mask, b, sMaps)

        # wavelets to image space
        out = self.applyW(out, 'transp')

        out = torch.view_as_real(out)
        n = out.shape[-3]
        out_r = torch.cat((out[..., 0], out[..., 1]), dim=2)

        del out # memory management
        gc.collect()

        post_unet = self.nn(out_r)
        post_unet_r = post_unet[..., :n, :]
        post_unet_im = post_unet[..., n:, :]

        post_unet = torch.stack((post_unet_r, post_unet_im), dim=-1)
        
        del post_unet_r, post_unet_im
        gc.collect()

        out = torch.view_as_complex(post_unet)

        del post_unet
        gc.collect()

        # back to wavelet coeffs
        out = self.applyW(out)

        return out, mask, b, sMaps


class unrolled_net(nn.Module):
    def __init__(self, sImg, device, n=10, dc=True, mc=True):
        super(unrolled_net, self).__init__()
        self.n = n
        self.device = device
        self.wavSplit = torch.tensor(math_utils.makeWavSplit(sImg))
        self.dc = dc
        mod = []
        if not mc: # single coil
            assert False, 'single coil not implemented for wavelets yet'
            for i in range(n):
                mod.append(unrolled_block_sc(sImg, device))
            mod.append(prox_block_sc(device))
        else: # multicoil
            for i in range(n):
                mod.append(unrolled_block(sImg, self.wavSplit, device, dc))
            if dc:
                mod.append(prox_block(device, self.wavSplit))
            else:
                mod.append(final_block_nodc(device, self.wavSplit))

        self.model = nn.Sequential(*mod)

    def forward(self, inputs, mask, b, sMaps):
      return self.model((inputs, mask, b, sMaps))
    
if __name__ == "__main__":
    device = torch.device("cpu")
    sMaps = torch.randn([1,640,320,16]) + 1j*torch.randn([1,640,320,16])
    kSpace = torch.randn([1,1,640,320,16]) + 1j*torch.randn([1,1,640,320,16])
    mask = torch.randn([1,1,640,320])
    mask = mask > 0
    model = unrolled_net([640, 320], device, n=2 )

    print(summary(model, [kSpace.shape[0:4], mask.shape[2:4], kSpace.shape, sMaps.shape], dtypes=[torch.complex64, torch.bool, torch.complex64, torch.complex64], device="cpu"))