# we need stuff like a data class, etc
# need to decide do we load the data first (like from disk) and use this class for a single image?
# i think that's what we want.

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace, norm

def complex_to_channels(x: torch.Tensor) -> torch.Tensor:
    """Convert complex tensor [B, H, W] to [B, 2, H, W]"""
    return torch.stack([x.real, x.imag], dim=1)

def channels_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert real tensor [B, 2, H, W] to complex [B, H, W]"""
    return torch.complex(x[:, 0], x[:, 1])

def save_im_cube(data, fname, title=''):
    """
    the expectation here is that we're plotting data of size [Nx x Ny x C]
    and we will make C subplots
    """
    sImg = data.shape
    fig = plt.figure()
    C = sImg[-1]
    cols = int(np.ceil(C/2))
    rows = int(np.ceil(C/cols))

    for i in range(C):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(np.abs(data[:,:,i]), cmap='gray')
        if len(title) == 0:
            ax.set_title('Coil ' + str(i))

    if len(title) > 0:
        plt.suptitle(title)

    fig.savefig(fname)
    plt.close()


def view_im_cube(data, title=''):
    """
    the expectation here is that we're plotting data of size [Nx x Ny x C]
    and we will make C subplots
    """
    sImg = data.shape
    fig = plt.figure()
    C = sImg[-1]
    cols = int(np.ceil(C/2))
    rows = int(np.ceil(C/cols))

    for i in range(C):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(np.abs(data[:,:,i]), cmap='gray')
        if len(title) == 0:
            ax.set_title('Coil ' + str(i))

    if len(title) > 0:
        plt.suptitle(title)

    plt.show()


def kspace_to_imspace(kspace):

    im_space = np.fft.ifftshift( np.fft.ifftn( np.fft.fftshift( kspace, axes=(0, 1)),  axes=(0, 1) ), axes=(0,1))

    return im_space

def view_im(kspace, title=''):

    im_space = kspace_to_imspace(kspace)

    plt.imshow( np.abs( im_space ), cmap='grey')

    if len(title) > 0:
        plt.title(title)
    plt.show()
    
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

def undersample_kspace(sImg, rng, samp_frac):
    """
    here we should do the generic undersampling of k-space to go from a fully sampled image to whatever
    undersampling pattern we want?
    """

    numCols = sImg[1]

    center = numCols // 2

    fivePer = round(0.05 * numCols)
    lowerB = center - fivePer
    upperB = center + fivePer

    colRange = [i for i in range(numCols) if i < lowerB or i > upperB]

    colsChosen = rng.choice(colRange, round(samp_frac * len(colRange)), replace=False)

    mask = np.zeros(sImg)
    mask[:, colsChosen] = 1
    mask[:, lowerB:upperB] = 1

    return mask

def undersample_kspace_gaussian(sImg, rng, samp_frac):
    """
    undersampling kspace in the same way but using a gaussian instead
    i think this needs to be iterative
    """

    numCols = sImg[1]

    center = numCols // 2

    fivePer = round(0.05 * numCols)
    lowerB = center - fivePer
    upperB = center + fivePer

    colRange = [i for i in range(numCols)]
    totalNumCols = round(samp_frac * len(colRange))

    colsChosen = [i for i in range(lowerB, upperB+1)]
    numColsChosen = len(colsChosen)

    while numColsChosen < totalNumCols:
        t = rng.normal(center, center/2)
        t = round(t)
        if t > 0 and t < numCols:
            if t not in colsChosen:
                colsChosen.append(t)
                numColsChosen += 1


    mask = np.zeros(sImg)
    mask[:, colsChosen] = 1

    return mask

def mask_split(mask, rng, big_frac):
    mask_indices = np.where(mask == 1)
    num_samples = len(mask_indices[0])

    mask_rows = mask_indices[0]
    mask_cols = mask_indices[1]

    # make training mask

    big_num = round(big_frac * num_samples)
    big_chosen = rng.choice(num_samples, big_num, replace=False)

    big_mask = np.zeros(mask.shape)
    for idx in range(big_num):
        mask_idx = big_chosen[idx]
        big_mask[mask_rows[mask_idx], mask_cols[mask_idx]] = 1

    small_mask = mask - big_mask
    return big_mask.astype(np.float32), small_mask.astype(np.float32)


def training_val_split(mask, rng, train_frac):
    """
    here we'll split the undersampling mask into the training and validation portions
    in the ZS-SSL paper this is splitting omega into Gamma and Omega/Gamma
    """

    mask_indices = np.where(mask == 1)
    num_samples = len(mask_indices[0])

    mask_rows = mask_indices[0]
    mask_cols = mask_indices[1]

    # make training mask

    train_num = round(train_frac * num_samples)
    train_chosen = rng.choice(num_samples, train_num, replace=False)

    train_mask = np.zeros(mask.shape)
    for idx in range(train_num):
        mask_idx = train_chosen[idx]
        train_mask[mask_rows[mask_idx], mask_cols[mask_idx]] = 1

    val_mask = mask - train_mask
    return train_mask, val_mask


def training_loss_split(training_mask, rng, loss_frac):
    """
    Here we'll take in the training mask and split that into training data and loss data. This will probably get called several times
    (in fact it will get called k times) per training epoch
    """
    mask_indices = np.where(training_mask == 1)
    num_samples = len(mask_indices[0])

    mask_rows = mask_indices[0]
    mask_cols = mask_indices[1]

    # make training mask

    loss_num = round(loss_frac * num_samples)
    loss_chosen = rng.choice(num_samples, loss_num, replace=False)

    loss_mask = np.zeros(training_mask.shape)
    for idx in range(loss_num):
        mask_idx = loss_chosen[idx]
        loss_mask[mask_rows[mask_idx], mask_cols[mask_idx]] = 1

    train_mask = training_mask - loss_mask
    return train_mask, loss_mask

def mri_reconRoemer(coilRecons, sMaps = []):
    """
    implements the Roemer reconstruction method
    https://pubmed.ncbi.nlm.nih.gov/2266841/
    numpy version
    """

    if len(sMaps) == 0:
        ssqRecon = np.sqrt( np.sum( coilRecons * np.conj(coilRecons), 2 ))
        sMaps = coilRecons / ssqRecon
        sMaps[sMaps == np.inf] = 0
    
    recon = np.sum( coilRecons * np.conj(sMaps), 2)

    return recon

def makeFullySampledCenterRegion(smask, shape):
    """
    make a fully sampled center region mask
    INPUTS:
      smask - size of total mask
      shape - size of fully sampled region
    """

    mask = np.zeros(smask)

    center = np.array(smask) // 2 # this should be the center point
    shapeRad = np.array(shape) // 2

    topY = center[0] - shapeRad[0]
    bottomY = center[0] + shapeRad[0]

    leftX = center[1] - shapeRad[1]
    rightX = center[1] + shapeRad[1]

    mask[topY:bottomY, leftX:rightX] = 1
    return mask

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


def make_masks(sImg, rng, samp_frac, train_frac, loss_frac=0.3):
    """
    i don't actually know what I should do here
    maybe a random subset of columns?

    yeah lets do a subset of columns and some amount chosen in the middle
    take the 10% of middle columns and some fraction of the rest

    INPUTS:
      sImg - dimensions of image
      rng - numpy random number generator
      samp_frac - fraction (0 < x < 1) of columns to use as undersampling
      train_frac - fraction (0 < x < 1) of samples to use as training, > 0.85 recommended
      loss_frac - fraction (0 < x < 1) of training mask to use as the loss calculation, < 0.3 recommended

    """

    # make sampling mask

    numCols = sImg[1]

    center = numCols // 2

    fivePer = round(0.05 * numCols)
    lowerB = center - fivePer
    upperB = center + fivePer

    colRange = [i for i in range(numCols) if i < lowerB or i > upperB]

    colsChosen = rng.choice(colRange, round(samp_frac * len(colRange)), replace=False)

    mask = np.zeros(sImg)
    mask[:, colsChosen] = 1
    mask[:, lowerB:upperB] = 1

    mask_indices = np.where(mask == 1)
    num_samples = len(mask_indices[0])

    mask_rows = mask_indices[0]
    mask_cols = mask_indices[1]

    # make training mask

    train_num = round(train_frac * num_samples)
    train_chosen = rng.choice(num_samples, train_num, replace=False)

    train_mask = np.zeros(sImg)
    for idx in range(train_num):
        mask_idx = train_chosen[idx]
        train_mask[mask_rows[mask_idx], mask_cols[mask_idx]] = 1

    train_mask_indices = np.where(train_mask == 1)
    num_train_samples = len(train_mask_indices[0])

    train_mask_rows = train_mask_indices[0]
    train_mask_cols = train_mask_indices[1]
    # make loss mask??
    # this is a subset of the training mask

    loss_num = round(loss_frac * train_num)
    loss_chosen = rng.choice(train_num, loss_num, replace=False)

    loss_mask = np.zeros(sImg)
    for idx in range(loss_num):
        mask_idx = loss_chosen[idx]
        loss_mask[train_mask_rows[mask_idx], train_mask_cols[mask_idx]] = 1

    rest_training_mask = train_mask - loss_mask
    val_mask = mask - train_mask

    # leftover_num_training = np.where(rest_training_mask == 1)

    # print(f'num samples in total image: {sImg[0]} * {sImg[1]} = {np.prod(sImg)}')
    # print(f'num samples in undersampled image: {num_samples}')
    # print(f'num samples in training mask: {train_num}')
    # print(f'num samples in training mask from mask: {num_train_samples}')
    # print(f'num samples in loss mask: {loss_num}')
    # print(f'num samples leftover for training: {len(leftover_num_training[0])}')
    # print(f'num samples for validation: {np.sum(val_mask)}')

    # plt.imshow(train_mask, cmap='gray')
    # plt.title('training mask')

    # plt.show()

    # plt.imshow(loss_mask, cmap='gray')
    # plt.title('loss mask')

    # plt.show()

    # plt.imshow(rest_training_mask, cmap='gray')
    # plt.title('leftover training')

    # plt.show()

    # plt.imshow(val_mask, cmap='gray')
    # plt.title('validation mask')

    # plt.show()

    return (
        mask.astype(np.float32),
        train_mask.astype(np.float32),
        loss_mask.astype(np.float32),
    )  # convert to 32bit float to prevent uptyping


class ssl_dataset(torch.utils.data.Dataset):
    """
    this class is going to assume that you've loaded the k-space samples
    from the image you want to reconstruct (e.g. fastmri)

    this is now a data loader over those samples with a given mask
    """

    def __init__(self, kspace_samples, train_mask):
        self.train_kspace = kspace_samples
        self.train_mask = train_mask

    def __len__(self):
        return len(self.train_kspace)

    def __getitem__(self, idx):
        sample = self.train_kspace(idx)
        return sample


def test():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(2024)

    sImg = [256, 256]
    sample_frac = 0.3
    big_frac = 0.85
    small_frac = 0.15

    mask = undersample_kspace(sImg, rng, sample_frac)
    # plt.imshow(mask, cmap='gray')
    # plt.show()

    mask2 = undersample_kspace_gaussian(sImg, rng, sample_frac)
    # plt.imshow(mask2, cmap='gray')
    # plt.show()

    rng1 = np.random.default_rng(2024)
    m1, t1 = mask_split(mask, rng1, big_frac)
    rng2 = np.random.default_rng(2024)
    m2, t2 = training_val_split(mask, rng2, big_frac)
    rng3 = np.random.default_rng(2024)
    m3, t3 = training_loss_split(mask, rng3, small_frac)

    sFSR = [32, 32]
    msk = makeFullySampledCenterRegion(sImg, sFSR)
    
    plt.imsave('maskTest_fsr.png', msk, cmap='gray')

    return 0

if __name__ == "__main__":
    test()