"""
different formulation of the unrolled network for zero-shot
since at each forward step we need to accept a different mask
we need to _not_ have a monolithic A operator
"""
import torch
import torch.nn as nn
import numpy as np
from unet import build_unet
from torchsummary import summary
import gc
import utils


class prox_block(nn.Module):
    """
    just a module for the proximal block
    """

    def __init__(self, sMaps, device):
        super(prox_block, self).__init__()
        self.device = device
        self.sMaps = sMaps
        self.nCoils = sMaps.shape[-1]

    def forward(self, inputs):
        """
        forward method for prox block
        simply to do a proximal step at the very end for data consistency
        """
        x, mask, b = inputs
        out = torch.zeros(size = [*x.shape, self.nCoils], dtype=self.sMaps.dtype, device = self.device)
        if len(x.shape) == 4:
            sm = self.sMaps.unsqueeze(0)
        else:
            sm = self.sMaps

        for i in range(self.nCoils):
            out[...,i] = sm[...,i] * x
        out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( out, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )

        if len(mask.shape) < 3:
          out[..., mask, :] = b[..., mask, :]
        else:
          out[mask] = b[mask]

        out = torch.fft.ifftshift( torch.fft.ifftn( torch.fft.fftshift( out, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
        out = torch.sum( torch.conj(sm) * out, -1 ) #roemer

        #out = torch.view_as_real(out)

        return out

class unrolled_block(nn.Module):
    """
    we probably need A matrix free huh
    I might need to actually make that
    """
    def __init__(self, sMaps, shape, device):
        super(unrolled_block, self).__init__()
        self.sMaps = sMaps
        self.nCoils = sMaps.shape[-1]
        self.device = device
        self.nn = build_unet(shape[1])
    
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
            out = torch.fft.ifftshift( torch.fft.ifftn( torch.fft.fftshift( x, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
        else:
            out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( x, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
        return out

    def grad_desc(self, x, A, b):
        """
        i think A has to downselect here?
        """
        xt = A(x) - b
        xt = A(xt, 'transp')
        x1 = x - xt
        return x1
    
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
        out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( out, dim=(2,3) ), norm='ortho',\
                                                 dim = (2,3) ), dim=(2,3) )

        if len(mask.shape) < 3:
          out[..., mask, :] = b[..., mask, :]
        else:
          out[mask] = b[mask]
        
        out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( out, dim=(2,3) ), norm='ortho',\
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
        x, mask, b = inputs # unpack
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
          else:
            # apply S
            out = self.applyS(x)
            # apply f
            out = self.applyF(out)
            # apply mask
            out = applyM(out)

          return out

        out = self.grad_desc(x, applyA, b)
        out = self.prox(out, mask, b)

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

        return out, mask, b
        
class unrolled_block_sc(nn.Module):
    """
    single coil version of the unrolled block
    no more sensitivity maps
    """
    def __init__(self, shape, device):
        super(unrolled_block, self).__init__()
        self.device = device
        self.nn = build_unet(shape[1])
    
    def applyF(self, x, op='notransp'):
        if op == 'transp':
            out = torch.fft.ifftshift( torch.fft.ifftn( torch.fft.fftshift( x, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
        else:
            out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( x, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
        return out

    def grad_desc(self, x, A, b):
        """
        i think A has to downselect here?
        """
        xt = A(x) - b
        xt = A(xt, 'transp')
        x1 = x - xt
        return x1
    
    def prox(self, x, mask, b):
        """
        apply fourier transform
        replace data at mask with b
        roemer recon
        """
        # out = torch.zeros(size=[self.nBatch, *self.sMaps.shape], dtype=self.sMaps.dtype)
        out = torch.zeros(size = [*x.shape], dtype=self.sMaps.dtype, device=self.device)

        out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( out ), norm='ortho' ) )

        out[mask] = b[mask] # projection
        
        out = torch.fft.ifftshift( torch.fft.ifftn( torch.fft.fftshift( out ), norm='ortho' ) )

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
        x, mask, b = inputs # unpack
        def applyM(x):
          out = x
          out[~mask] = 0
          return out

        def applyA(x, op='notransp'):
          if op == 'transp':
            out = applyM(x)
            # apply f transpose
            out = self.applyF(out, 'transp')
          else:
            # apply f
            out = self.applyF(out)
            # apply mask
            out = applyM(out)

          return out

        out = self.grad_desc(x, applyA, b)
        out = self.prox(out, mask, b)

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

        return out, mask, b

class ZS_Unrolled_Network(nn.Module):
    def __init__(self, sMaps, sImg, device, n=10):
        super(ZS_Unrolled_Network, self).__init__()
        self.n = n
        mod = []
        for i in range(n):
            mod.append(unrolled_block(sMaps, sImg, device))
        mod.append(prox_block(sMaps, device))
        self.model = nn.Sequential(*mod)

    def forward(self, inputs, mask, b):
        return self.model((inputs, mask, b))
