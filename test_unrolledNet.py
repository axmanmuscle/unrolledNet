import torch
import scipy.io as sio
from unrollednet import Unrolled_Network, newA
import numpy as np
from scipy.stats import laplace, norm
from torchinfo import summary

from zs_unrollednet import ZS_Unrolled_Network

def test_adjoint(x0, f, ip = lambda x, y: torch.vdot(x, y), num_test = 10):
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
    #if np.any(x0.imag): #numpy
    if x0.dtype == torch.complex128:
        dataComplex = True


    for _ in range(num_test):
        y1 = rng.normal(size=x0.shape)
        y2 = rng.normal(size=fx0.shape)

        if dataComplex:
            y1 = y1 + 1j * rng.normal(size=x0.shape)
            y2 = y2 + 1j * rng.normal(size=fx0.shape)

        y1 = torch.tensor(y1)
        y2 = torch.tensor(y2)
        fy1 = f(y1)
        fty2 = f(y2, 'transp')
        
        t1 = ip(y1, fty2)
        t2 = ip(fy1, y2)

        error += np.abs(t1 - t2)

    error /= num_test
    assert error < 1e-10, "adjoint test failed"


def size2imgCoordinates(n):
    """
    INPUTS:
        n - array giving number of elements in each dimension
    OUTPUTS:
        coords - if n is a scalar, a 1D array of image coordinates
                 if n is an array, then a (size(n) x 1) array of image coordinates
    """
    if type(n) == list or tuple:
        numN = len(n)
        coords = []
        for i in range(numN):
            coords.append(size2img1d(n[i]))
    else:
        coords = [size2img1d(n)]

    return coords

def size2img1d(N):
    coords = np.array([i for i in range(N)]) - np.floor(0.5*N)
    return coords.astype('int')

def vdSampleMask(smask, sigmas, numSamps, maskType = 'laplace'):
    """
    generates a vd sample mask
    INPUTS:
        smask - 1-D array corresponding to number of samples in each dimension
        sigmas - 1-D array corresponding to the standard deviation of the distribution
                 in each dimension
        numSamps - number of (total) samples

    """
    maxIters = 500
    rng = np.random.default_rng(20230911)

    coords = size2imgCoordinates(smask)

    mask = np.zeros(smask)
    nDims = len(smask) # can't pass in just an integer

    if maskType == 'laplace':
        pdf = lambda x, sig: laplace.pdf(x, loc=0, scale=np.sqrt(0.5*sig*sig))
    elif maskType == 'gaussian':
        pdf = lambda x, sig: norm.pdf(x, loc=0, scale=sig)

    for idx in range(maxIters):
        sampsLeft = int(numSamps - mask.sum(dtype=int))
        dimSamps = np.zeros((nDims, sampsLeft))
        for dimIdx in range(nDims):
            c = coords[dimIdx]
            probs = pdf(c, sigmas[dimIdx])
            probs = probs / sum(probs)
            samps = rng.choice(c, sampsLeft, p=probs)
            dimSamps[dimIdx, :] = samps - min(c)
        
        mask[tuple(dimSamps.astype(int))] = 1

        if mask.sum(dtype=int) == numSamps:
            return mask
    
    print('hit max iters vdSampleMask')
    return mask

def main():
    # data = sio.loadmat('/Users/alex/Documents/School/Research/Dwork/dataConsistency/brain_data.mat')
    data = sio.loadmat('/home/alex/Documents/research/mri/data/brain_data.mat')
    kSpace = data['d2']
    kSpace = kSpace / np.max(np.abs(kSpace))
    sMaps = data['smap']
    sMaps = sMaps / np.max(np.abs(sMaps))

    sImg = kSpace.shape[0:2]

    mask = vdSampleMask(kSpace.shape[0:2], [30, 30], np.round(np.prod(kSpace.shape[0:2]) * 0.4))
    us_kSpace = kSpace*mask[:, :, np.newaxis]

    test_nBatch = 2
    sMaps = torch.tensor(sMaps, dtype=torch.complex64)
    mask = torch.tensor(mask)
    mask = mask.to(torch.bool)
    mk2 = torch.zeros(test_nBatch, *mask.shape, dtype=torch.bool)
    kSpace2 = torch.zeros(test_nBatch, *kSpace.shape, dtype=torch.complex64)
    for i in range(test_nBatch):
        mk2[i, :, :] = mask
        kSpace2[i, :, :] = torch.tensor(kSpace)

    mk2 = mk2.unsqueeze(1)
    kSpace2 = kSpace2.unsqueeze(1)
    us_kSpace[~mask] = 0
    us_kSpace = torch.tensor(us_kSpace)

    A = lambda x, t='notransp': newA(x, mk2, sMaps, t)
    ## b = us_kSpace[mask] 
    # b = us_kSpace
    b = kSpace2
    x0 = np.random.randn(test_nBatch, *kSpace.shape[0:2]) + 1j * np.random.randn(test_nBatch,*kSpace.shape[0:2])
    x0 = torch.tensor(x0)
    # Ax0 = A(x0)
    # AtAx0 = A(Ax0, 'transp')
    # ip = lambda x, y: torch.real(torch.vdot(x.flatten(), y.flatten()))
    # test_adjoint(x0, A, ip)
    # print(f'Adjoint test passed')

    device = torch.device("cpu")

    mk2 = mk2.to(device)
    b = b.to(device)
    sMaps = sMaps.to(device)

    # model = Unrolled_Network(A, b, mk2, sMaps, sImg, 5)
    # model = model.to(device)
    # print(summary(model, (1, 256, 256), device="cpu"))
    
    # model2 = ZS_Unrolled_Network(sMaps, sImg)
    # model2 = model2.to(device)
    # print(summary(model2, [(2, 1, 256, 256), mk2.shape, b.shape], dtypes=[torch.complex64, torch.bool, torch.complex64], device="cpu"))

    model3 = ZS_Unrolled_Network(sImg, device)
    print(summary(model3, [(1, 1, 256, 256), mk2.shape, b.shape], dtypes=[torch.complex64, torch.bool, torch.complex64], device="cpu"))


    return 0

if __name__ == "__main__":
    main()