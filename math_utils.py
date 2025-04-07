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

def makeWavSplit(sImg, minSplitSize=16):
    """
    make the split to be used iwht the wavelet transforms
    INPUTS:
        sImg - size of the image as a tuple or list
        (OPTIONAL) minSplitSize - minimum size of split, default 16
    """

    def findNPows(sizeDim, minSplitSize):
        binPow = np.log2(sizeDim)

        nPow = 0
        for powIdx in range(np.floor(binPow).astype(int)):
            if sizeDim % 2 == 0 and sizeDim/2 >= minSplitSize:
                nPow += 1
                sizeDim = sizeDim/2
            else:
                break
        
        return nPow

    nDims = len(sImg)
    nPows = np.zeros(shape=(1, nDims))

    if np.size(minSplitSize) == 1:
        minSplitSize = minSplitSize * np.ones(shape=(nDims, 1))

    for dimIdx in range(nDims):
        nPows[0, dimIdx] = findNPows( sImg[dimIdx], minSplitSize[dimIdx, 0])

    if nDims == 1:
        wavSplit = np.zeros(shape=(2**(nPows-1), 1))
    else:
        wavSplit = np.zeros(*np.power(2, nPows-1).astype(int))

    wavSplit[0, 0] = 1

    sWavSplit = wavSplit.shape
    wavSplit = wavSplit[0:np.min(sWavSplit), 0:np.min(sWavSplit)]

    return wavSplit

def upsample2(img, U):
    """
    only implementing the forward method for right now
    """

    sImg = img.shape
    ndims = len(sImg)
    S = torch.zeros((ndims, 1), dtype=torch.int64)

    sOut = torch.tensor(sImg) * torch.tensor(U) # this might be fragile

    yqs = S[0] + torch.arange(0, sImg[0])*U[0]
    xqs = S[1] + torch.arange(0, sImg[1])*U[1]

    size = [sOut[i].item() for i in range(sOut.numel())]
    out = torch.zeros(size, dtype=img.dtype, device=img.device)
    out[torch.meshgrid(yqs, xqs, indexing='ij')] = img

    return out

def wtDaubechies2( img, split = torch.tensor([1]) ):
    """
    applies the 2 dimensional Daubechies 4 wavelet transform
    INPUTS:
        img - 2d img
        split - (OPTIONAL) describes the way to split
    """
    sSplit = torch.tensor(split.shape)

    imgrt3 = torch.sqrt(torch.tensor(3)) * img
    img3 = 3 * img

    imgPimgrt3 = img + imgrt3
    img3Pimgrt3 = img3 + imgrt3
    imgMimgrt3 = img - imgrt3
    img3Mimgrt3 = img3 - imgrt3

    wt1 = imgMimgrt3 + torch.roll(img3Mimgrt3, -1, dims=0) \
                     + torch.roll(img3Pimgrt3, -2, dims=0) \
                     + torch.roll(imgPimgrt3, -3, dims=0)
    wt1 = wt1[::2, :]

    wt1rt3 = wt1 * torch.sqrt(torch.tensor(3))
    wt13 = wt1 * 3

    wt1Pwt1rt3 = wt1 + wt1rt3
    wt13Pwt1rt3 = wt13 + wt1rt3
    wt1Mwt1rt3 = wt1 - wt1rt3
    wt13Mwt1rt3 = wt13 - wt1rt3

    wt11 = wt1Mwt1rt3 + torch.roll(wt13Mwt1rt3, [0, -1], dims=[0, 1]) \
                      + torch.roll(wt13Pwt1rt3, [0, -2], dims=[0, 1]) \
                      + torch.roll(wt1Pwt1rt3, [0, -3], dims=[0, 1])
    
    wt11 = wt11[:, ::2]

    wt12 = -1*wt1Pwt1rt3 + torch.roll(wt13Pwt1rt3, [0, -1], dims=[0, 1]) \
                         + torch.roll(-1*wt13Mwt1rt3, [0, -2], dims=[0, 1]) \
                         + torch.roll(wt1Mwt1rt3, [0, -3], dims=[0, 1])
    wt12 = wt12[:, ::2]

    wt2 = -1*imgPimgrt3 + torch.roll(img3Pimgrt3, [-1, 0], dims=[0, 1]) \
                        + torch.roll(-1*img3Mimgrt3, [-2, 0], dims=[0, 1]) \
                        + torch.roll(imgMimgrt3, [-3, 0], dims=[0, 1])
    wt2 = wt2[::2, :]

    wt2rt3 = wt2 * torch.sqrt(torch.tensor(3))
    wt23 = wt2 * 3

    wt2Pwt2rt3 = wt2 + wt2rt3
    wt23Pwt2rt3 = wt23 + wt2rt3
    wt2Mwt2rt3 = wt2 - wt2rt3
    wt23Mwt2rt3 = wt23 - wt2rt3

    wt21 = wt2Mwt2rt3 + torch.roll(wt23Mwt2rt3, [0, -1], dims=[0, 1]) \
                      + torch.roll(wt23Pwt2rt3, [0, -2], dims=[0, 1]) \
                      + torch.roll(wt2Pwt2rt3, [0, -3], dims=[0, 1])
    wt21 = wt21[:, ::2]

    wt22 = -1*wt2Pwt2rt3 + torch.roll(wt23Pwt2rt3, [0, -1], dims=[0, 1]) \
                         + torch.roll(-1*wt23Mwt2rt3, [0, -2], dims=[0, 1]) \
                         + torch.roll(wt2Mwt2rt3, [0, -3], dims=[0, 1])
    wt22 = wt22[:, ::2]

    nSplit = split.numel()
    if nSplit > 1:
        s11 = split[0:sSplit[0]//2,0:sSplit[1]//2]
        s12 = split[0:sSplit[0]//2, sSplit[1]//2+1:]
        s21 = split[sSplit[1]//2+1:, 0:sSplit[0]//2]
        s22 = split[sSplit[1]//2+1:, sSplit[1]//2+1:]

        if s11.sum() > 0:
            if torch.any(torch.remainder(torch.tensor(wt11.shape), 2)):
                raise ValueError('wt11 is invalid shape')
            wt11 = wtDaubechies2(wt11, s11)

        if s12.sum() > 0:
            if torch.any(torch.remainder(torch.tensor(wt12.shape), 2)):
                raise ValueError('wt12 is invalid shape')
            wt12 = wtDaubechies2(wt12, s12)

        if s21.sum() > 0:
            if torch.any(torch.remainder(torch.tensor(wt21.shape), 2)):
                raise ValueError('wt21 is invalid shape')
            wt21 = wtDaubechies2(wt21, s21)

        if s22.sum() > 0:
            if torch.any(torch.remainder(torch.tensor(wt22.shape), 2)):
                raise ValueError('wt22 is invalid shape')
            wt22 = wtDaubechies2(wt22, s22)


    a1 = torch.concatenate([wt11, wt12], dim=1)
    a2 = torch.concatenate([wt21, wt22], dim=1)
    wt = torch.concatenate([a1, a2], dim=0)

    wt /= 32

    return wt

def iwtDaubechies2(wt, split = torch.tensor([1])):
    """
    inverse Daubechies wavelet transformation
    """

    sWT = wt.shape
    ## TODO check that the sizes are divisible by two?
    wt11 = wt[:sWT[0]//2, :sWT[1]//2]
    wt21 = wt[sWT[0]//2:, :sWT[1]//2]
    wt12 = wt[:sWT[0]//2, sWT[1]//2:]
    wt22 = wt[sWT[0]//2:, sWT[1]//2:]

    sSplit = torch.tensor(split.shape)
    if torch.max( torch.remainder( torch.log2( sSplit), 1) ) > 0:
        raise ValueError('something in the split is the wrong size')
    nSplit = split.numel()
    if nSplit > 1:
        s11 = split[:sSplit[0]//2, :sSplit[1]//2]
        s12 = split[:sSplit[0]//2, sSplit[1]//2:]
        s21 = split[sSplit[1]//2:, :sSplit[0]//2]
        s22 = split[sSplit[1]//2:, sSplit[1]//2:]

        if s11.sum() > 0:
            if torch.any(torch.remainder(torch.tensor(wt11.shape), 2)):
                raise ValueError('wt11 is invalid shape')
            wt11 = iwtDaubechies2(wt11, s11)
        if s12.sum() > 0:
            if torch.any(torch.remainder(torch.tensor(wt12.shape), 2)):
                raise ValueError('wt12 is invalid shape')
            wt12 = iwtDaubechies2(wt12, s12)
        if s21.sum() > 0:
            if torch.any(torch.remainder(torch.tensor(wt21.shape), 2)):
                raise ValueError('wt21 is invalid shape')
            wt21 = iwtDaubechies2(wt21, s21)
        if s22.sum() > 0:
            if torch.any(torch.remainder(torch.tensor(wt22.shape), 2)):
                raise ValueError('wt22 is invalid shape')
            wt22 = iwtDaubechies2(wt22, s22)
    
    ## todo: write upsample
    tmp = upsample2(wt11, [1, 2])

    tmp3 = 3 * tmp
    tmprt3 = torch.sqrt(torch.tensor(3)) * tmp

    wt1_1 = tmp - tmprt3 + torch.roll(tmp3 - tmprt3, [0, 1], dims = [0, 1]) \
                         + torch.roll( tmp3 + tmprt3, [0, 2], dims=[0, 1]) \
                         + torch.roll(tmp + tmprt3, [0, 3], dims = [0, 1])
    
    tmp = upsample2(wt12, [1, 2])
    tmp3 = 3 * tmp
    tmprt3 = torch.sqrt(torch.tensor(3)) * tmp

    wt1_2 = -1 * (tmp + tmprt3) + torch.roll(tmp3 + tmprt3, [0, 1], dims = [0, 1]) \
                                + torch.roll( -1 * (tmp3 - tmprt3), [0, 2], dims=[0, 1]) \
                                + torch.roll(tmp - tmprt3, [0, 3], dims = [0, 1])
    
    wt1 = upsample2( wt1_1 + wt1_2, [2, 1])
    
    tmp = upsample2(wt21, [1, 2])
    tmp3 = 3 * tmp
    tmprt3 = torch.sqrt(torch.tensor(3)) * tmp

    wt2_1 = tmp - tmprt3 + torch.roll(tmp3 - tmprt3, [0, 1], dims = [0, 1]) \
                         + torch.roll(tmp3 + tmprt3, [0, 2], dims=[0, 1]) \
                         + torch.roll(tmp + tmprt3, [0, 3], dims = [0, 1])
    
    tmp = upsample2( wt22, [1, 2])
    tmp3 = 3 * tmp
    tmprt3 = torch.sqrt(torch.tensor(3)) * tmp

    wt2_2 = -1 * (tmp + tmprt3) + torch.roll(tmp3 + tmprt3, [0, 1], dims = [0, 1]) \
                                + torch.roll(-1 * (tmp3 - tmprt3), [0, 2], dims=[0, 1]) \
                                + torch.roll(tmp - tmprt3, [0, 3], dims = [0, 1])
    
    wt2 = upsample2( wt2_1 + wt2_2, [2, 1])

    tmp = wt1
    tmp3 = 3 * tmp
    tmprt3 = torch.sqrt(torch.tensor(3)) * tmp

    sig1 = tmp - tmprt3 + torch.roll( tmp3 - tmprt3, [1, 0], dims = [0, 1]) \
                       + torch.roll( tmp3 + tmprt3, [2, 0], dims = [0, 1]) \
                       + torch.roll( tmp + tmprt3, [3, 0], dims = [0, 1])
    
    tmp = wt2
    tmp3 = 3 * tmp
    tmprt3 = torch.sqrt(torch.tensor(3)) * tmp

    sig2 = -1*(tmp + tmprt3) + torch.roll(tmp3 + tmprt3, [1, 0], dims = [0, 1]) \
                             + torch.roll( -1*( tmp3 - tmprt3), [2, 0], dims = [0, 1]) \
                             + torch.roll( tmp - tmprt3, [3, 0], dims = [0, 1])
    
    img = (sig1 + sig2) / 32

    return img

def upsample2_np(img, U):
    """
    only implementing the forward method for right now
    """

    sImg = img.shape
    ndims = len(sImg)
    S = np.zeros((ndims, 1), dtype=np.int64)

    sOut = np.array(sImg) * np.array(U) # this might be fragile

    yqs = S[0] + np.arange(0, sImg[0])*U[0]
    xqs = S[1] + np.arange(0, sImg[1])*U[1]

    out = np.zeros(sOut, dtype=img.dtype)
    out[np.ix_(yqs, xqs)] = img

    return out

def wtDaubechies2_np( img, split = np.array([1]) ):
    """
    applies the 2 dimensional Daubechies 4 wavelet transform
    INPUTS:
        img - 2d img
        split - (OPTIONAL) describes the way to split
    """
    sSplit = split.shape

    imgrt3 = np.sqrt(3) * img
    img3 = 3 * img

    imgPimgrt3 = img + imgrt3
    img3Pimgrt3 = img3 + imgrt3
    imgMimgrt3 = img - imgrt3
    img3Mimgrt3 = img3 - imgrt3

    wt1 = imgMimgrt3 + np.roll(img3Mimgrt3, -1, axis=0) \
                     + np.roll(img3Pimgrt3, -2, axis=0) \
                     + np.roll(imgPimgrt3, -3, axis=0)
    wt1 = wt1[::2, :]

    wt1rt3 = wt1 * np.sqrt(3)
    wt13 = wt1 * 3

    wt1Pwt1rt3 = wt1 + wt1rt3
    wt13Pwt1rt3 = wt13 + wt1rt3
    wt1Mwt1rt3 = wt1 - wt1rt3
    wt13Mwt1rt3 = wt13 - wt1rt3

    wt11 = wt1Mwt1rt3 + np.roll(wt13Mwt1rt3, [0, -1], axis=[0, 1]) \
                      + np.roll(wt13Pwt1rt3, [0, -2], axis=[0, 1]) \
                      + np.roll(wt1Pwt1rt3, [0, -3], axis=[0, 1])
    
    wt11 = wt11[:, ::2]

    wt12 = -1*wt1Pwt1rt3 + np.roll(wt13Pwt1rt3, [0, -1], axis=[0, 1]) \
                         + np.roll(-1*wt13Mwt1rt3, [0, -2], axis=[0, 1]) \
                         + np.roll(wt1Mwt1rt3, [0, -3], axis=[0, 1])
    wt12 = wt12[:, ::2]

    wt2 = -1*imgPimgrt3 + np.roll(img3Pimgrt3, [-1, 0], axis=[0, 1]) \
                        + np.roll(-1*img3Mimgrt3, [-2, 0], axis=[0, 1]) \
                        + np.roll(imgMimgrt3, [-3, 0], axis=[0, 1])
    wt2 = wt2[::2, :]

    wt2rt3 = wt2 * np.sqrt(3)
    wt23 = wt2 * 3

    wt2Pwt2rt3 = wt2 + wt2rt3
    wt23Pwt2rt3 = wt23 + wt2rt3
    wt2Mwt2rt3 = wt2 - wt2rt3
    wt23Mwt2rt3 = wt23 - wt2rt3

    wt21 = wt2Mwt2rt3 + np.roll(wt23Mwt2rt3, [0, -1], axis=[0, 1]) \
                      + np.roll(wt23Pwt2rt3, [0, -2], axis=[0, 1]) \
                      + np.roll(wt2Pwt2rt3, [0, -3], axis=[0, 1])
    wt21 = wt21[:, ::2]

    wt22 = -1*wt2Pwt2rt3 + np.roll(wt23Pwt2rt3, [0, -1], axis=[0, 1]) \
                         + np.roll(-1*wt23Mwt2rt3, [0, -2], axis=[0, 1]) \
                         + np.roll(wt2Mwt2rt3, [0, -3], axis=[0, 1])
    wt22 = wt22[:, ::2]

    nSplit = split.size
    if nSplit > 1:
        s11 = split[0:sSplit[0]//2,0:sSplit[1]//2]
        s12 = split[0:sSplit[0]//2, sSplit[1]//2+1:]
        s21 = split[sSplit[1]//2+1:, 0:sSplit[0]//2]
        s22 = split[sSplit[1]//2+1:, sSplit[1]//2+1:]

        if s11.sum() > 0:
            if np.any(np.mod(wt11.shape, 2)):
                raise ValueError('wt11 is invalid shape')
            wt11 = wtDaubechies2(wt11, s11)

        if s12.sum() > 0:
            if np.any(np.mod(wt12.shape, 2)):
                raise ValueError('wt12 is invalid shape')
            wt12 = wtDaubechies2(wt12, s12)

        if s21.sum() > 0:
            if np.any(np.mod(wt21.shape, 2)):
                raise ValueError('wt21 is invalid shape')
            wt21 = wtDaubechies2(wt21, s21)

        if s22.sum() > 0:
            if np.any(np.mod(wt22.shape, 2)):
                raise ValueError('wt22 is invalid shape')
            wt22 = wtDaubechies2(wt22, s22)


    a1 = np.concatenate([wt11, wt12], axis=1)
    a2 = np.concatenate([wt21, wt22], axis=1)
    wt = np.concatenate([a1, a2], axis=0)

    wt /= 32

    return wt

def iwtDaubechies2_np(wt, split = np.array([1])):
    """
    inverse Daubechies wavelet transformation
    """

    sWT = wt.shape
    ## TODO check that the sizes are divisible by two?
    wt11 = wt[:sWT[0]//2, :sWT[1]//2]
    wt21 = wt[sWT[0]//2:, :sWT[1]//2]
    wt12 = wt[:sWT[0]//2, sWT[1]//2:]
    wt22 = wt[sWT[0]//2:, sWT[1]//2:]

    sSplit = split.shape
    if np.max( np.mod( np.log2( sSplit), 1) ) > 0:
        raise ValueError('something in the split is the wrong size')
    nSplit = split.size
    if nSplit > 1:
        s11 = split[:sSplit[0]//2, :sSplit[1]//2]
        s12 = split[:sSplit[0]//2, sSplit[1]//2:]
        s21 = split[sSplit[1]//2:, :sSplit[0]//2]
        s22 = split[sSplit[1]//2:, sSplit[1]//2:]

        if s11.sum() > 0:
            if np.any(np.mod(wt11.shape, 2)):
                raise ValueError('wt11 is invalid shape')
            wt11 = iwtDaubechies2(wt11, s11)
        if s12.sum() > 0:
            if np.any(np.mod(wt12.shape, 2)):
                raise ValueError('wt12 is invalid shape')
            wt12 = iwtDaubechies2(wt12, s12)
        if s21.sum() > 0:
            if np.any(np.mod(wt21.shape, 2)):
                raise ValueError('wt21 is invalid shape')
            wt21 = iwtDaubechies2(wt21, s21)
        if s22.sum() > 0:
            if np.any(np.mod(wt22.shape, 2)):
                raise ValueError('wt22 is invalid shape')
            wt22 = iwtDaubechies2(wt22, s22)
    
    ## todo: write upsample
    tmp = upsample2_np(wt11, [1, 2])

    tmp3 = 3 * tmp
    tmprt3 = np.sqrt(3) * tmp

    wt1_1 = tmp - tmprt3 + np.roll(tmp3 - tmprt3, [0, 1], axis = [0, 1]) \
                         + np.roll( tmp3 + tmprt3, [0, 2], axis=[0, 1]) \
                         + np.roll(tmp + tmprt3, [0, 3], axis = [0, 1])
    
    tmp = upsample2_np(wt12, [1, 2])
    tmp3 = 3 * tmp
    tmprt3 = np.sqrt(3) * tmp

    wt1_2 = -1 * (tmp + tmprt3) + np.roll(tmp3 + tmprt3, [0, 1], axis = [0, 1]) \
                                + np.roll( -1 * (tmp3 - tmprt3), [0, 2], axis=[0, 1]) \
                                + np.roll(tmp - tmprt3, [0, 3], axis = [0, 1])
    
    wt1 = upsample2_np( wt1_1 + wt1_2, [2, 1])
    
    tmp = upsample2_np(wt21, [1, 2])
    tmp3 = 3 * tmp
    tmprt3 = np.sqrt(3) * tmp

    wt2_1 = tmp - tmprt3 + np.roll(tmp3 - tmprt3, [0, 1], axis = [0, 1]) \
                         + np.roll(tmp3 + tmprt3, [0, 2], axis=[0, 1]) \
                         + np.roll(tmp + tmprt3, [0, 3], axis = [0, 1])
    
    tmp = upsample2_np( wt22, [1, 2])
    tmp3 = 3 * tmp
    tmprt3 = np.sqrt(3) * tmp

    wt2_2 = -1 * (tmp + tmprt3) + np.roll(tmp3 + tmprt3, [0, 1], axis = [0, 1]) \
                                + np.roll(-1 * (tmp3 - tmprt3), [0, 2], axis=[0, 1]) \
                                + np.roll(tmp - tmprt3, [0, 3], axis = [0, 1])
    
    wt2 = upsample2_np( wt2_1 + wt2_2, [2, 1])

    tmp = wt1
    tmp3 = 3 * tmp
    tmprt3 = np.sqrt(3) * tmp

    sig1 = tmp - tmprt3 + np.roll( tmp3 - tmprt3, [1, 0], axis = [0, 1]) \
                       + np.roll( tmp3 + tmprt3, [2, 0], axis = [0, 1]) \
                       + np.roll( tmp + tmprt3, [3, 0], axis = [0, 1])
    
    tmp = wt2
    tmp3 = 3 * tmp
    tmprt3 = np.sqrt(3) * tmp

    sig2 = -1*(tmp + tmprt3) + np.roll(tmp3 + tmprt3, [1, 0], axis = [0, 1]) \
                             + np.roll( -1*( tmp3 - tmprt3), [2, 0], axis = [0, 1]) \
                             + np.roll( tmp - tmprt3, [3, 0], axis = [0, 1])
    
    img = (sig1 + sig2) / 32

    return img


if __name__ == "__main__":
    rng = np.random.default_rng(2025)
    for _ in range(20):
        x0 = rng.normal(size=(25, 25))
        x1 = np.roll(x0, [-1, 0], axis=[0,1])
        x2 = torch.roll(torch.tensor(x0), [-1, 0], dims=[0, 1])
        print(torch.norm(x2 - torch.tensor(x1)))

    w = rng.normal(size = (256, 256))
    ww = wtDaubechies2_np(w)
    wwt = iwtDaubechies2_np(ww)

    torch_ww = wtDaubechies2(torch.tensor(w))
    torch_wwt = iwtDaubechies2(torch_ww)
    print(torch.norm(torch_wwt - torch.tensor(wwt)))
    print(torch.norm(torch_ww - torch.tensor(ww)))