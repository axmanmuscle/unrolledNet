import torch
import torch.nn as nn
import numpy as np
from unet import build_unet,build_unet_small,build_unet_smaller
from torchinfo import summary
import gc
import utils
import math_utils

class prox_block(nn.Module):
    """
    just a module for the proximal block
    """

    def __init__(self, sMaps, device, wavSplit, cornerOrigin):
        super(prox_block, self).__init__()
        self.device = device
        self.sMaps = sMaps
        self.nCoils = sMaps.shape[-1]
        self.wavSplit = wavSplit
        self.cornerOrigin = cornerOrigin

    def forward(self, inputs):
        """
        forward method for prox block
        simply to do a proximal step at the very end for data consistency
        """
        x, mask, b, nn = inputs

        # wavelet coefficients to image space
        x = math_utils.iwtDaubechies2(torch.squeeze(x), self.wavSplit)

        x = x[None, None, :, :]

        out = torch.zeros(size = [*x.shape, self.nCoils], dtype=self.sMaps.dtype, device = self.device)

        if len(x.shape) == 4:
            sm = self.sMaps.unsqueeze(0)
        else:
            sm = self.sMaps

        for i in range(self.nCoils):
            out[...,i] = sm[...,i] * x

        if self.cornerOrigin:
          out = torch.fft.fftshift( torch.fft.fftn( out , norm='ortho', dim=(2,3) ), dim=(2,3) )
        else:
          out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( out, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )

        if len(mask.shape) < 3:
          out[..., mask, :] = b[..., mask, :]
        else:
          out[mask] = b[mask]

        if self.cornerOrigin:
          out = torch.fft.ifftn( torch.fft.ifftshift( out, dim=(2,3) ), norm='ortho', dim=(2,3) )
        else:
          out = torch.fft.fftshift( torch.fft.ifftn( torch.fft.ifftshift( out, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
        out = torch.sum( torch.conj(sm) * out, -1 ) #roemer

        #out = torch.view_as_real(out)

        return out

class final_block_nodc(nn.Module):
    """
    final block for the wavelet network
    """

    def __init__(self, sMaps, device, wavSplit):
        super(final_block_nodc, self).__init__()
        self.device = device
        self.sMaps = sMaps
        self.nCoils = sMaps.shape[-1]
        self.wavSplit = wavSplit

    def forward(self, inputs):
        """
        forward method for the final block
        just return the wavelet coefficients to image space
        """
        x, mask, b, _ = inputs

        # wavelet coefficients to image space
        x = math_utils.iwtDaubechies2(torch.squeeze(x), self.wavSplit)

        return x

class unrolled_block(nn.Module):
    """
    unrolled block for single shared network
    """
    def __init__(self, sMaps, wavSplit, device, dc=True, cornerOrigin=False):
        super(unrolled_block, self).__init__()
        self.sMaps = sMaps
        self.nCoils = sMaps.shape[-1]
        self.device = device
        self.wavSplit = wavSplit
        self.dc = dc
        self.cornerOrigin = cornerOrigin

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
        torch.cuda.empty_cache()
        # gc.collect()
        
        return out
    
    def applyS(self, x, op='notransp'):
        if op == 'transp':
            out = torch.sum( torch.conj(self.sMaps) * x, -1 )
        else:
            # out = torch.zeros(size=[nBatch, 1, *sMaps.shape], dtype=sMaps.dtype)
            out = torch.zeros(size = [*x.shape, self.nCoils], dtype=self.sMaps.dtype, device=self.device)
            if len(x.shape) == 4:
                sm = self.sMaps.unsqueeze(0)
            else:
                sm = self.sMaps
            for i in range(self.nCoils):
                out[..., i] = sm[...,i] * x

        return out

    def applyF(self, x, op='notransp'):
        if op == 'transp':
          if self.cornerOrigin:
            out = torch.fft.ifftn( torch.fft.ifftshift( x, dim=(2,3) ), norm='ortho', dim=(2,3) )
          else:
            out = torch.fft.fftshift( torch.fft.ifftn( torch.fft.ifftshift( x, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
        else:
          if self.cornerOrigin:
            out = torch.fft.fftshift( torch.fft.fftn( x, norm='ortho', dim=(2,3) ), dim=(2,3) )
          else:
            out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( x, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
        return out

    def grad_desc(self, x, A, b):
        """
        i think A has to downselect here?
        """
        def obj(xi):
           return 0.5 * torch.norm(A(xi) - b)**2
        
        def grad(xi):
            xt = A(xi) - b
            xt = A(xt, 'transp')
            return xt
        
        gx = grad(x)
        gxNorm = torch.norm(gx.reshape(-1, 1))**2
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

        # print(f'grad_descent line search finished after {linesearch_iter} iters')
        return xNew
    
    def prox(self, x, mask, b):
        """
        apply sensitivity maps
        apply fourier transform
        replace data at mask with b
        roemer recon
        """
        # out = torch.zeros(size=[self.nBatch, *self.sMaps.shape], dtype=self.sMaps.dtype)
        out = torch.zeros(size = [*x.shape, self.nCoils], dtype=self.sMaps.dtype, device=self.device)
        if len(x.shape) == 4:
            sm = self.sMaps.unsqueeze(0)
        else:
            sm = self.sMaps
        for i in range(self.nCoils):
            out[...,i] = sm[...,i] * x

        if self.cornerOrigin:
          out = torch.fft.fftshift( torch.fft.fftn( out, norm='ortho',\
                                                 dim = (2,3) ), dim=(2,3) )
        else:
          out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( out, dim=(2,3) ), norm='ortho',\
                                                 dim = (2,3) ), dim=(2,3) )

        if len(mask.shape) < 3:
          out[..., mask, :] = b[..., mask, :]
        else:
          out[mask] = b[mask]
        
        if self.cornerOrigin:
          out = torch.fft.ifftn( torch.fft.ifftshift( out, dim=(2,3) ), norm='ortho',\
                                                 dim = (2,3) )
        else:
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
        x, mask, b, nn = inputs # unpack
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
            out = self.applyS(out, 'transp')
            # wavelet transform
            out = self.applyW(out)
            
          else:
            # inverse wavelet transform
            out = self.applyW(x, 'transp')
            # apply S
            out = self.applyS(out)
            # apply f
            out = self.applyF(out)
            # apply mask
            out = applyM(out)

          return out

        # adjoint testing
        # self.sMaps = self.sMaps.to(torch.complex128)
        # err = math_utils.test_adjoint_torch(x.to(torch.complex128), applyA)
        with torch.no_grad(): ## helps immensely 
            out = self.grad_desc(x, applyA, b)

            if self.dc:
                out = self.prox(out, mask, b)

            # wavelets to image space
            out = self.applyW(out, 'transp')

        out = torch.view_as_real(out)
        n = out.shape[-3]
        out_r = torch.cat((out[..., 0], out[..., 1]), dim=2)

        del out # memory management
        # gc.collect()
        torch.cuda.empty_cache()

        post_unet = nn(out_r)
        post_unet_r = post_unet[..., :n, :]
        post_unet_im = post_unet[..., n:, :]

        post_unet = torch.stack((post_unet_r, post_unet_im), dim=-1)
        
        del post_unet_r, post_unet_im
        # gc.collect()
        torch.cuda.empty_cache()

        out = torch.view_as_complex(post_unet)

        # don't know if we need this or not
        # mval = torch.max(torch.abs(out))
        # out = out / mval

        del post_unet
        # gc.collect()
        torch.cuda.empty_cache()

        # back to wavelet coeffs
        out = self.applyW(out)

        return out, mask, b, nn


class ZS_Unrolled_Network_onenet(nn.Module):
    def __init__(self, sImg, device, sMaps=[], n=10, dc=True, cornerOrigin=False):
        super(ZS_Unrolled_Network_onenet, self).__init__()
        self.n = n
        self.device = device
        self.wavSplit = torch.tensor(math_utils.makeWavSplit(sImg))
        self.dc = dc
        self.unet = build_unet(sImg[1])
        # self.unet = build_unet_smaller(sImg[1])
        self.cornerOrigin = cornerOrigin # the fetal data origin in image space is in the corner
        mod = []
        if len(sMaps) == 0: # single coil
            assert False, 'single coil not implemented for wavelets yet'
        else: # multicoil
            for i in range(n):
                mod.append(unrolled_block(sMaps, self.wavSplit, device, dc, cornerOrigin))
            if dc:
                mod.append(prox_block(sMaps, device, self.wavSplit, cornerOrigin))
            else:
                mod.append(final_block_nodc(sMaps, device, self.wavSplit))

        self.model = nn.Sequential(*mod)

    def forward(self, inputs, mask, b):
      return self.model((inputs, mask, b, self.unet))
    

if __name__ == "__main__":
    import scipy.io as sio

    # test size of model
    data = sio.loadmat('/home/mcmanus/code/unrolledNet/brain_data_newsmap.mat')
    # data = sio.loadmat('/Users/alex/Documents/School/Research/Dwork/dataConsistency/brain_data_newsmap.mat')
    kSpace = data['d2']
    kSpace = kSpace / np.max(np.abs(kSpace))
    sMaps = data['sm2']
    sMaps = sMaps / np.max(np.abs(sMaps))

    sImg = kSpace.shape[0:2]
    sMaps = torch.tensor(sMaps)
    kSpace = torch.tensor(kSpace)
    kSpace = kSpace.unsqueeze(0)
    kSpace = kSpace.unsqueeze(0)
    kSpace = kSpace.to(torch.complex64)
    sMaps = sMaps.to(torch.complex64)

    mask = utils.vdSampleMask(sImg, [30, 30], np.round(np.prod(sImg) * 0.4))
    # b = kSpace * mask
    d = torch.device("cuda")
    sMaps = sMaps.to(d)

    model = ZS_Unrolled_Network_onenet([256, 256], d, sMaps, n=10 )

    print(summary(model, [kSpace.shape[0:4], mask.shape, sMaps.shape], dtypes=[torch.complex64, torch.bool, torch.complex64], device="cuda"))