"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import gc

def complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Complex multiplication.

    This multiplies two complex tensors assuming that they are both stored as
    real arrays with the last dimension being the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == y.shape[-1] == 2:
        raise ValueError("Tensors do not have separate complex dim.")

    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]

    return torch.stack((re, im), dim=-1)


def complex_conj(x: torch.Tensor) -> torch.Tensor:
    """
    Complex conjugate.

    This applies the complex conjugate assuming that the input array has the
    last dimension as the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data**2).sum(dim=-1).sqrt()


def complex_abs_sq(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared absolute value of a complex tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Squared absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data**2).sum(dim=-1)


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    return torch.view_as_complex(data).numpy()

def np_to_complex(data: np.ndarray) -> np.ndarray:
    """
    converts a [N, M, 2] real array to [N, M] complex
    """
    return data[..., 0] + 1j*data[..., 1]

def complex_mse_loss(output, target, mask):
    om = output * mask
    tm = target * mask
    return torch.abs((0.5*(om - tm)**2).mean(dtype=torch.complex64))

def unrolled_loss(output, target, mask, sMaps):
  """
  here the output is going to be the output of the network in image space [Nbatch x Nx x Ny]
  and target is the k-space [Nbatch x Nchannel x Nx x Ny x Ncoil]
  mask will be size of k-space

  so we need to go to k-space and then do L2 loss on the masked data
  """

  nCoils = target.shape[-1]
  if target.shape[0] == 1:
    target = target.squeeze()
  out = torch.zeros(size = sMaps.shape, dtype= sMaps.dtype, device=output.device)
  if len(output.shape) == 4:
      sm = sMaps.unsqueeze(0)
  else:
      sm = sMaps
  for i in range(nCoils):
      out[..., i] = sm[...,i] * output

  del output, sm
  gc.collect()

  om = out[mask > 0]
  tm = target[mask > 0]

  out = torch.norm(om.flatten() - tm.flatten()) / torch.norm(tm.flatten())

  del om, tm
  gc.collect()

  return out  

def unrolled_loss_mixed(output, target, mask, sMaps):
    """
  here the output is going to be the output of the network in image space [Nbatch x Nx x Ny]
  and target is the k-space [Nbatch x Nchannel x Nx x Ny x Ncoil]
  mask will be size of k-space

  so we need to go to k-space and then do L2 loss on the masked data
  """

    nCoils = target.shape[-1]
    if target.shape[0] == 1:
      target = target.squeeze()
    out = torch.zeros(size = sMaps.shape, dtype= sMaps.dtype, device=output.device)
    if len(output.shape) == 4:
        sm = sMaps.unsqueeze(0)
    else:
        sm = sMaps
    for i in range(nCoils):
        out[..., i] = sm[...,i] * output

    del output, sm
    gc.collect()

    om = out[mask > 0]
    tm = target[mask > 0]

    out = torch.norm(om.flatten() - tm.flatten()) / torch.norm(tm.flatten()) + torch.norm(om.flatten() - tm.flatten(), 1) / torch.norm(tm.flatten(), 1)

    del om, tm
    gc.collect()

    return out 

def unrolled_loss_sc(output, target, mask):
  """
  here the output is going to be the output of the network in image space [Nbatch x Nx x Ny]
  and target is the k-space [Nbatch x Nchannel x Nx x Ny x Ncoil]
  mask will be size of k-space

  so we need to go to k-space and then do L2 loss on the masked data
  """

  if target.shape[0] == 1:
    target = target.squeeze()
 
  om = output[..., mask > 0]
  tm = target[..., mask > 0]

  out = 0.5*torch.norm(om.flatten() - tm.flatten())**2 # / torch.norm(tm.flatten())

  del om, tm
  gc.collect()

  return out  

def unrolled_loss_mixed_sc(output, target, mask):
    """
  here the output is going to be the output of the network in image space [Nbatch x Nx x Ny]
  and target is the k-space [Nbatch x Nchannel x Nx x Ny x Ncoil]
  mask will be size of k-space

  so we need to go to k-space and then do L2 loss on the masked data
  """

    om = output[..., mask > 0]
    tm = target[..., mask > 0]

    out = torch.norm(om.flatten() - tm.flatten()) / torch.norm(tm.flatten()) + torch.norm(om.flatten() - tm.flatten(), 1) / torch.norm(tm.flatten(), 1)

    del om, tm
    gc.collect()

    return out 

def mixed_loss(output, target, mask):
    om = output * mask
    tm = target * mask
    n = torch.norm(om - tm) / torch.norm(tm) + torch.norm(om - tm, 1) / torch.norm(tm, 1)
    return n

def kspace_to_imspace(kspace):
  
    im_space = np.fft.ifftshift( np.fft.ifftn( np.fft.fftshift( kspace, axes=(0, 1)),  axes=(0, 1) ), axes=(0,1))

    return im_space

def view_im(kspace, title=''):

    im_space = kspace_to_imspace(kspace)

    plt.imshow( np.abs( im_space ), cmap='grey')

    if len(title) > 0:
        plt.title(title)
    plt.show()

def test_adjoint(x0, f, ip = lambda x, y: np.vdot(x, y), num_test = 10):
    """
    test whether the adjoint of f is implemented correctly
    f should be a function that takes in x0 and an optional parameter of either 'transp' or 'notransp'
    ip is the inner product to use, this is really only for funky situations where you have a real scalar field etc.
    """

    fx0 = f(x0)
    ftfx0 = f(fx0, 'transp')
    rng = np.random.default_rng(20250303)

    error = 0

    dataComplex = False
    if np.any(x0.imag):
        dataComplex = True


    for _ in range(num_test):
        y1 = rng.normal(size=x0.shape)
        y2 = rng.normal(size=fx0.shape)

        if dataComplex:
            y1 = y1 + 1j * rng.normal(size=x0.shape)
            y2 = y2 + 1j * rng.normal(size=fx0.shape)

        fy1 = f(y1)
        fty2 = f(y2, 'transp')
        
        t1 = ip(y1, fty2)
        t2 = ip(fy1, y2)

        error += np.abs(t1 - t2)

    error /= num_test
    assert error < 1e-8, "adjoint test failed"

def test_adjoint_torch(x0, f, ip = lambda x, y: torch.real(torch.vdot(x.flatten(), y.flatten())), num_test = 10):
    """
    test whether the adjoint of f is implemented correctly
    f should be a function that takes in x0 and an optional parameter of either 'transp' or 'notransp'
    ip is the inner product to use, this is really only for funky situations where you have a real scalar field etc.
    """

    fx0 = f(x0)
    ftfx0 = f(fx0, 'transp')
    rng = torch.Generator()
    rng.manual_seed(20250332)
    tol = 1e-8

    error = torch.tensor([0])

    dataComplex = False
    if torch.is_complex(x0):
        dataComplex = True


    for _ in range(num_test):
        # y1 = torch.normal(mean=0, std=1, size=x0.shape, generator=rng)
        # y2 = torch.normal(mean=0, std=1, size=fx0.shape, generator=rng)

        # if dataComplex:
        #     y1 = y1 + 1j * torch.normal(mean=0, std=1, size=x0.shape, generator=rng)
        #     y2 = y2 + 1j * torch.normal(mean=0, std=1, size=fx0.shape, generator=rng)

        if dataComplex:
            y1 = torch.randn_like(x0, dtype=torch.complex128)
            y2 = torch.randn_like(fx0, dtype=torch.complex128)
        else:
            y1 = torch.randn_like(x0, dtype=torch.float64)
            y2 = torch.randn_like(fx0, dtype=torch.float64)

        fy1 = f(y1)
        fty2 = f(y2, 'transp')
        
        t1 = ip(y1, fty2)
        t2 = ip(fy1, y2)

        tmperr = torch.abs(t1 - t2)

        if torch.abs(t1) > 0.1 * tol:
            tmperr /= torch.abs(t1)

        error = torch.max(error, tmperr)

    if error > 1e-8:
        print("adjoint test failed")

    return error