"""
a class for an unrolled network
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

we're going so solve
min 0.5 * || Ax - b ||_2^2 st || Ax - b ||^2 <= eps

with A = MFS
Seems strange but the grad descent will impact all of the image x and the constraint will enforce data consistency
"""

import torch
import torch.nn as nn
import numpy as np
from unet import build_unet
from torchsummary import summary
import gc

class prox_block(nn.Module):
    """
    just a module for the proximal block
    """

    def __init__(self, b, mask, sMaps):
        super(prox_block, self).__init__()
        self.mask = mask
        self.b = b
        self.sMaps = sMaps
        self.nBatch = mask.shape[0]
        self.nCoils = sMaps.shape[-1]

    def forward(self, x):
        """
        forward method for prox block
        simply to do a proximal step at the very end for data consistency
        """
        out = torch.zeros(size = [*x.shape, self.nCoils], dtype=self.sMaps.dtype)
        if len(x.shape) == 4:
            sm = self.sMaps.unsqueeze(0)
        else:
            sm = self.sMaps

        for i in range(self.nCoils):
            out[...,i] = sm[...,i] * x
        out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( out ), norm='ortho' ) )
        out[self.mask] = self.b[self.mask]
        out = torch.fft.ifftshift( torch.fft.ifftn( torch.fft.fftshift( out ), norm='ortho' ) )
        out = torch.sum( torch.conj(sm) * out, -1 ) #roemer

        out = torch.view_as_real(out)

        return out

class unrolled_block(nn.Module):
    """
    we probably need A matrix free huh
    I might need to actually make that
    """
    def __init__(self, A, b, mask, sMaps, shape):
        super(unrolled_block, self).__init__()
        self.A = A
        self.b = b
        self.mask = mask
        self.nBatch = mask.shape[0]
        self.sMaps = sMaps
        self.nCoils = sMaps.shape[-1]
        self.nn = build_unet(shape[1])
    
    def grad_desc(self, x):
        """
        i think A has to downselect here?
        """
        xt = self.A(x) - self.b
        xt = self.A(xt, 'transp')
        x1 = x - xt
        return x1
    
    def prox(self, x):
        """
        TODO: write prox operator
        apply sensitivity maps
        apply fourier transform
        replace data at mask with b
        roemer recon (?)
        """
        # out = torch.zeros(size=[self.nBatch, *self.sMaps.shape], dtype=self.sMaps.dtype)
        out = torch.zeros(size = [*x.shape, self.nCoils], dtype=self.sMaps.dtype)
        if len(x.shape) == 4:
            sm = self.sMaps.unsqueeze(0)
        else:
            sm = self.sMaps
        for i in range(self.nCoils):
            out[...,i] = sm[...,i] * x
        out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( out ), norm='ortho' ) )
        out[self.mask] = self.b[self.mask]
        out = torch.fft.ifftshift( torch.fft.ifftn( torch.fft.fftshift( out ), norm='ortho' ) )
        out = torch.sum( torch.conj(sm) * out, -1 ) #roemer

        return out

    def forward(self, inputs):
        """
        maybe let inputs be the N x Y image and reshape it before the neural net?
        """
        out = self.grad_desc(inputs)
        out = self.prox(out)

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

        return out
        

class Unrolled_Network(nn.Module):
    def __init__(self, A, b, mask, sMaps, sImg, n=10):
        super(Unrolled_Network, self).__init__()
        self.n = n
        mod = []
        for i in range(n):
            mod.append(unrolled_block(A, b, mask, sMaps, sImg))
        mod.append(prox_block(b, mask, sMaps))
        self.model = nn.Sequential(*mod)

    def forward(self, inputs):
        return self.model(inputs)

def Afun(A, x, op='notransp'):
    """
    TODO this needs to be in torch
    """
    out = torch.zeros(x.shape)
    nDims = len(x.shape)

    if nDims == 2:
        if op == 'transp':
            out = torch.dot(A.T, x)
        else:
            out = torch.dot(A, x)
    
    if nDims == 3:
        out = torch.zeros((x.shape[0], A.shape[0], x.shape[2]))

    if nDims == 4:
        out = x

    return out

def newA(x, mask, sMaps, op='notransp'):
    """
    this needs to be MFS

    the questions is do we want A : [Nx x Ny] -> [Nx x Ny x Ncoils]
    or A : [Nx x Ny] -> [Nx*Ny*Ncoils*p x 1] ?

    okay we have to assume that x is minibatched
    so [Nbatch x Nx x Ny] -> [Nbatch x Nx x Ny x Ncoils]
    """
    sKspace = sMaps.shape
    sImg = x.shape
    assert len(sImg) > 2, "x must either have 3 or 4 dimensions"
    nBatch = x.shape[0]
    sImg = sKspace[1:3] # this may need to change if we do batching?
    nCoils = sKspace[-1]

    def applyM(x, op='notransp'):
        """
        image space is size [Nx x Ny]
        k-space is size [Nx x Ny x Ncoils]
        total number of collected samples is Nx*Ny*Ncoils*p for some (0 < p < 1) undersampling factor
        so M applies to k-space after F, S and should be [Nx x Ny x Ncoils] -> [Nx*Ny*Ncoils*p x 1]
        and M^* applies to k-space before F, S and should be [Nx*Ny*Ncoils*p x 1] -> [Nx x Ny x Ncoils] 
        """
        # total number of uncollected samples is sum of mask * nCoils
        nCollected = torch.sum(mask) * sMaps.shape[-1]
        nColperCoil = torch.sum(mask)
        if op == 'transp':
            # # if you want in to be a column vector [Nx*Ny*Ncoils*p x 1]
            # out = torch.zeros_like(sMaps)
            # for i in range(nCoils):
            #     oi = out[:, :, i]
            #     oi[mask] = x[i*nColperCoil:(i+1)*nColperCoil]
            #     out[:,:,i] = oi
            out = x 
            out[~mask] = 0
        else:
            # out = x[mask].flatten() # if we want out to be a column vector [Nx*Ny*Ncoils*p x 1]
            out = x
            out[~mask] = 0 # if we want out to be [Nx x Ny x Ncoils] but zero filled
        return out

    def applyF(x, op='notransp'):
        if op == 'transp':
            out = torch.fft.ifftshift( torch.fft.ifftn( torch.fft.fftshift( x ), norm='ortho' ) )
        else:
            out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( x ), norm='ortho' ) )
        return out
    
    def applyS(x, op='notransp'):
        """
        S should take x in image space from [Nx x Ny] -> [Nx x Ny x Ncoils]
            (or do we want to map it to [Nx x Ny*Ncoils])
        So S^* should take y in kspace from [Nx x Ny x Ncoils] -> [Nx x Ny]
        sMaps are of shape [Nx x Ny x Ncoils]
        """
        if op == 'transp':
            out = torch.sum( torch.conj(sMaps) * x, -1 )
        else:
            # out = torch.zeros(size=[nBatch, 1, *sMaps.shape], dtype=sMaps.dtype)
            out = torch.zeros(size = [*x.shape, nCoils], dtype=sMaps.dtype)
            if len(x.shape) == 4:
                sm = sMaps.unsqueeze(0)
            else:
                sm = sMaps
            for i in range(nCoils):
                out[..., i] = sm[...,i] * x
        return out

    if op == 'transp':
        """
        here x should be a column vector of size [Nx*Ny*Ncoils*p x 1]
        and out will be image space size [Nx x Ny]
        """

        # zero fill to size from mask transpose
        out = applyM(x, 'transp')

        # apply f transpose
        out = applyF(out, 'transp')

        # apply S transpose
        out = applyS(out, 'transp')

    else:
        """
        x should be something like
        [Nbatch x Nchannels x Nx x Ny x Ncoils] as complex at this point?
        x is just the image so it should be 
        [Nbatch x Nchannels x Nx x Ny]
        sMaps oughta be
        [Nx x Ny x Ncoils]
        maybe just make the sMaps
        [1 x 1 x Nx x Ny x Ncoils]
        and apply that somehow
        """
        # apply S
        out = applyS(x, sMaps)

        # apply f
        out = applyF(out)
        
        # apply mask
        out = applyM(out)


    return out

def main():
    # b = torch.zeros((256, 256), dtype=torch.float32)
    # mask = torch.zeros((256, 256), dtype=torch.bool)
    # sImg = [256, 256]

    # A = lambda x, t='notransp': newA(x, mask, smaps, t)
    # model = Unrolled_Network(A, b, mask, sImg)
    # print(summary(model, (1, 256, 256)))
    return 0

if __name__ == "__main__":
    main()