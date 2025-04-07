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
import math_utils


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

        out = torch.fft.fftshift( torch.fft.ifftn( torch.fft.ifftshift( out, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
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
            out = torch.fft.fftshift( torch.fft.ifftn( torch.fft.ifftshift( x, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
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
        alpha = 1 # TODO this may need to get changed
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

        out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( out, dim=(2,3) ), norm='ortho',\
                                                 dim = (2,3) ), dim=(2,3) )

        if len(mask.shape) < 3:
          out[..., mask, :] = b[..., mask, :]
        else:
          out[mask] = b[mask]
        
        # should this fourier transform be here? I don't think so
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

class unrolled_block_wav(nn.Module):
    """
    we probably need A matrix free huh
    I might need to actually make that
    """
    def __init__(self, sMaps, shape, wavSplit, device):
        super(unrolled_block, self).__init__()
        self.sMaps = sMaps
        self.nCoils = sMaps.shape[-1]
        self.device = device
        self.nn = build_unet(shape[1])
        self.wavSplit = wavSplit

    def applyW(self, x, op='notransp'):
        """
        wavelet transform here
        """
        if op == 'transp':
            out = math_utils.iwtDaubechies2(x, self.wavSplit)
        else: 
            out = math_utils.wtDaubechies2(x, self.wavSplit)
        
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
            out = torch.fft.fftshift( torch.fft.ifftn( torch.fft.ifftshift( x, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
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
        alpha = 1 # TODO this may need to get changed
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

        out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( out, dim=(2,3) ), norm='ortho',\
                                                 dim = (2,3) ), dim=(2,3) )

        if len(mask.shape) < 3:
          out[..., mask, :] = b[..., mask, :]
        else:
          out[mask] = b[mask]
        
        # should this fourier transform be here? I don't think so
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
            # wavelet transform
            out = self.applyW(out)
            
          else:
            # inverse wavelet transform
            out = self.applyW(x, 'transp')
            # apply S
            out = self.applyS(x)
            # apply f
            out = self.applyF(out)
            # apply mask
            out = applyM(out)

          return out

        out = self.grad_desc(x, applyA, b)
        out = self.prox(out, mask, b)

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

        return out, mask, b

class unrolled_block_gd(nn.Module):
    """
    unrolled block but its only gradient descent
    """
    def __init__(self, sMaps, shape, device):
        super(unrolled_block_gd, self).__init__()
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
            out = torch.fft.fftshift( torch.fft.ifftn( torch.fft.ifftshift( x, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
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
        alpha = 1 # TODO this may need to get changed
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

        print(f'grad_descent line search finished after {linesearch_iter} iters')
        return xNew

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

        return out, mask, b

class prox_block_sc(nn.Module):
    """
    just a module for the proximal block
    """

    def __init__(self, device):
        super(prox_block_sc, self).__init__()
        self.device = device

    def forward(self, inputs):
        """
        forward method for prox block
        simply to do a proximal step at the very end for data consistency
        """
        x, mask, b = inputs

        out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( x, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )

        out[..., mask] = b[..., mask]

        out = torch.fft.ifftshift( torch.fft.ifftn( torch.fft.fftshift( out, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )

        return out

class unrolled_block_sc(nn.Module):
    """
    single coil version of the unrolled block
    no more sensitivity maps
    """
    def __init__(self, shape, device):
        super(unrolled_block_sc, self).__init__()
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

        print(f'grad_descent line search finished after {linesearch_iter} iters')
        return xNew
    
    def prox(self, x, mask, b):
        """
        apply fourier transform
        replace data at mask with b
        roemer recon
        """
        out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( x ), norm='ortho' ) )

        out[..., mask] = b[..., mask] # projection
        
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
          x[..., ~mask] = 0
          return x

        def applyA(x, op='notransp'):
          if op == 'transp':
            out = applyM(x)
            # apply f transpose
            out = self.applyF(out, 'transp')
          else:
            # apply f
            out = self.applyF(x)
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
    
class unrolled_block_sc_gd(nn.Module):
    """
    single coil version of the unrolled block
    gradient descent only
    """
    def __init__(self, shape, device):
        super(unrolled_block_sc_gd, self).__init__()
        self.device = device
        self.nn = build_unet(shape[1])
    
    def applyF(self, x, op='notransp'):
        if op == 'transp':
            out = torch.fft.ifftshift( torch.fft.ifftn( torch.fft.fftshift( x, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
        else:
            out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( x, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
        return out
    
    def applyF2(self, x, op='notransp'):
        if op == 'transp':
            out = (x.shape[2] * x.shape[3]) * torch.fft.ifftshift( torch.fft.ifftn( torch.fft.fftshift( x, dim=(2,3) ), dim=(2,3) ), dim=(2,3) )
        else:
            out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( x, dim=(2,3) ), dim=(2,3) ), dim=(2,3) )
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
        alpha = 1 # TODO this may need to get changed
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

        print(f'grad_descent line search finished after {linesearch_iter} iters')
        return xNew

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
        def applyM(x, op=None):
          x[..., ~mask] = 0
          return x

        def applyA(x, op='notransp'):
          if op == 'transp':
            out = applyM(x)
            # apply f transpose
            out = self.applyF(out, 'transp')
          else:
            # apply f
            out = self.applyF(x)
            # apply mask
            out = applyM(out)

          return out
        
        def applyA2(x, op='notransp'):
          if op == 'transp':
            out = applyM(x)
            # apply f transpose
            out = self.applyF2(out, 'transp')
          else:
            # apply f
            out = self.applyF2(x)
            # apply mask
            out = applyM(out)

          return out

        # math_utils.test_adjoint_torch(x, applyM)
        # math_utils.test_adjoint_torch(x, self.applyF)
        # math_utils.test_adjoint_torch(x, self.applyF2)
        # math_utils.test_adjoint_torch(x, applyA)
        out = self.grad_desc(x, applyA, b)

        return out, mask, b

class ZS_Unrolled_Network(nn.Module):
    def __init__(self, sImg, device, sMaps=[], n=10):
        super(ZS_Unrolled_Network, self).__init__()
        self.n = n
        self.device = device
        mod = []
        if len(sMaps) == 0: # single coil
            for i in range(n):
                mod.append(unrolled_block_sc(sImg, device))
            mod.append(prox_block_sc(device))
        else: # multicoil
            for i in range(n):
                mod.append(unrolled_block(sMaps, sImg, device))
            mod.append(prox_block(sMaps, device))

        self.model = nn.Sequential(*mod)

    def forward(self, inputs, mask, b):
        return self.model((inputs, mask, b))
    
class ZS_Unrolled_Network_gd(nn.Module):
    def __init__(self, sImg, device, sMaps=[], n=10):
        super(ZS_Unrolled_Network_gd, self).__init__()
        self.n = n
        self.device = device
        mod = []
        if len(sMaps) == 0: # single coil
            for i in range(n):
                mod.append(unrolled_block_sc_gd(sImg, device))
        else: # multicoil
            for i in range(n):
                mod.append(unrolled_block_gd(sMaps, sImg, device))

        self.model = nn.Sequential(*mod)

    def forward(self, inputs, mask, b):
        o, m, b = self.model((inputs, mask, b))
        return o

class ZS_Unrolled_Network_wavelets(nn.Module):
    def __init__(self, sImg, device, sMaps=[], n=10):
        super(ZS_Unrolled_Network, self).__init__()
        self.n = n
        self.device = device
        self.wavSplit = math_utils.makeWavSplit(sImg)
        mod = []
        if len(sMaps) == 0: # single coil
            assert False, 'single coil not implemented for wavelets yet'
            for i in range(n):
                mod.append(unrolled_block_sc(sImg, device))
            mod.append(prox_block_sc(device))
        else: # multicoil
            for i in range(n):
                mod.append(unrolled_block_wav(sMaps, sImg, device))
            mod.append(prox_block(sMaps, device))

        self.model = nn.Sequential(*mod)

    def forward(self, inputs, mask, b):
        return self.model((inputs, mask, b))