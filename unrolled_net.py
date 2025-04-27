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

    def __init__(self, sMaps, device, wavSplit):
        super(prox_block, self).__init__()
        self.device = device
        self.sMaps = sMaps
        self.nCoils = sMaps.shape[-1]
        self.wavSplit = wavSplit

    def forward(self, inputs):
        """
        forward method for prox block
        simply to do a proximal step at the very end for data consistency
        """
        x, mask, b = inputs

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

        out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( out, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )

        if len(mask.shape) < 3:
          out[..., mask, :] = b[..., mask, :]
        else:
          out[mask] = b[mask]

        out = torch.fft.fftshift( torch.fft.ifftn( torch.fft.ifftshift( out, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
        out = torch.sum( torch.conj(sm) * out, -1 ) #roemer

        #out = torch.view_as_real(out)

        return out


class unrolled_net(nn.Module):
    def __init__(self, sImg, device, sMaps=[], n=10, dc=True):
        super(unrolled_net, self).__init__()
        self.n = n
        self.device = device
        self.wavSplit = torch.tensor(math_utils.makeWavSplit(sImg))
        self.dc = dc
        mod = []
        if len(sMaps) == 0: # single coil
            assert False, 'single coil not implemented for wavelets yet'
            for i in range(n):
                mod.append(unrolled_block_sc(sImg, device))
            mod.append(prox_block_sc(device))
        else: # multicoil
            for i in range(n):
                mod.append(unrolled_block_wav(sMaps, sImg, self.wavSplit, device, dc))
            if dc:
                mod.append(prox_block(sMaps, device, self.wavSplit))
            else:
                mod.append(final_block_nodc_wav(sMaps, device, self.wavSplit))

        self.model = nn.Sequential(*mod)

    def forward(self, inputs, mask, b):
      return self.model((inputs, mask, b))