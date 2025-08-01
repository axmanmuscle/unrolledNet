"""
supervised training for the unrolled network

here we're just starting with the UNet to reconstruct images, no unrolled net
"""

import numpy as np
import glob
import math_utils
import torch
import os
import h5py
import utils
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import argparse
import logging
from torch.utils.data import random_split
from unet import build_unet, build_unet_small
from model_supervised import supervised_net
from model_supervised_separate import unrolled_net_separate
import matplotlib.pyplot as plt
from math_utils import supervised_mse_loss, supervised_mixed_loss, supervised_sse_loss, supervised_mixed_loss_sum

# from model_unet import supervised_unet

## helper functions
def parse_args():
    parser = argparse.ArgumentParser(description="Train unrolled MRI reconstruction model")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to dataset root (should contain kspace and sens_maps)")
    parser.add_argument('--save_dir', type=str, default='./results', help="Directory to save checkpoints and logs")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--dc', action='store_true', help="Enforce Data Consistency or not")
    parser.add_argument('--grad', action='store_true', help="Do grad descent step/s or not")
    parser.add_argument('--ls', action='store_true', help="Do grad descent line search")
    parser.add_argument('--wav', action='store_true', help="wavelets")
    parser.add_argument('--share', action='store_true', help="when TRUE, this option makes the unet at all layers of the unrolled network use the same weights")
    parser.add_argument('--alpha', type=float, default=1e-3, help="(optional) grad descent default step size")
    parser.add_argument('--n', type=int, default=1, help = 'number of unrolled iters to do (default 1)')
    parser.add_argument('--checkpoint', type=str, default='false', help="checkpoint from which to resume training")
    parser.add_argument('--sf', type=float, default = 0.1, help='total sample burden to use (NOTE: this may not be exact due to presence of fully sampled center region)')
    parser.add_argument('--lambd', type=float, default = 0.5, help = 'loss function is || x-y ||_2^2 + lambda*||x - y||_1')
    parser.add_argument('--sum', action='store_true', help='if this is true/enabled, use the sum of errors instead of mean reduction for loss')
    args = parser.parse_args()
    return args

def evaluate(model, val_loader, device, mask, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for kspace, sens_maps, target in val_loader:
            # Move data to device
            kspace = kspace.to(device)  
            sens_maps = sens_maps.to(device)  
            target = target.to(device)

            # Apply undersampling mask to k-space
            kspace_undersampled = kspace.clone()
            if len(mask.shape) < 3:
                mask_exp = mask[None, None, :, :]  # (1, 1, H, W)
                kspace_undersampled = kspace * mask_exp
            else:
                kspace_undersampled[~mask] = 0

            # Initialize input for the model (e.g., zero-filled reconstruction)
            ks = torch.fft.ifftshift( torch.fft.ifftn(torch.fft.fftshift(kspace_undersampled, dim=[2, 3]), dim = [2, 3]), dim = [2, 3])

            ## estim noise
            noise_val = ks[..., :25, :25]
            eps = []
            for i in range(ks.shape[1]):
                eps.append(torch.std(noise_val[:, i, ...]))

            ## for now let's not do roemer, just SoS of the zero-filled
            ks1 = ks * torch.conj(sens_maps)
            # ks1 = ks * torch.conj(ks) # SoS
            x_init = torch.sum(ks1, dim = 1) # dim = 1 is coil dimension

            sens_maps = torch.permute(sens_maps, dims=(0,2,3,1))
            kspace_undersampled = torch.permute(kspace_undersampled, (0,2,3,1))

            # Forward pass
            output = model(x_init, mask, kspace_undersampled, sens_maps, eps)  # Output shape: (batch, H, W)

            # Compute target image (e.g., inverse Fourier transform of fully-sampled k-space)
            itarget = torch.fft.ifftshift( torch.fft.ifftn(torch.fft.fftshift(target, dim=[2, 3]), dim = [2, 3]), dim = [2, 3])
            itarget = torch.permute(itarget, dims=(0, 2, 3, 1))
            itarget1 = itarget * torch.conj(sens_maps)
            target_image = torch.sum(itarget1, dim=-1) # dim 1 is coil dim

            # Normalize both to match scale
            # output = output / output.abs().amax(dim=(-2, -1), keepdim=True)
            # target_image = target_image / target_image.abs().amax(dim=(-2, -1), keepdim=True)

            # loss = criterion(output.abs(), target_image.abs())
            loss = criterion(output, target_image)
            val_loss += loss.item()
    return val_loss / len(val_loader.dataset)

class MRIDataset(Dataset):
    def __init__(self, kspace_dir, sens_dir):
        """
        Initialize dataset with directories containing k-space and sensitivity map .h5 files.
        
        Args:
            kspace_dir (str): Directory containing k-space .h5 files.
            sens_dir (str): Directory containing sensitivity map .h5 files.
        """
        self.kspace_dir = kspace_dir
        self.sens_dir = sens_dir
        self.slice_indices = []
        
        # Get list of k-space .h5 files
        kspace_files = sorted(glob.glob(os.path.join(kspace_dir, "*.h5")))
        
        # Build slice indices
        for kspace_path in kspace_files:
            # Construct corresponding sensitivity map file path
            kspace_filename = os.path.basename(kspace_path)
            sens_filename = kspace_filename.replace(".h5", "_sens.h5")
            sens_path = os.path.join(sens_dir, sens_filename)
            
            # Verify sensitivity map file exists
            if not os.path.exists(sens_path):
                print(f"file not found for sens (skipping for now)")
                continue
            
            # Open both files to get number of slices
            with h5py.File(kspace_path, 'r') as kspace_f, h5py.File(sens_path, 'r') as sens_f:
                kspace_data = kspace_f['kspace']  # Assume dataset key is 'kspace'
                sens_data = sens_f['sens_maps']  # dataset key is 'sensitivity_maps'
                
                num_slices_kspace = kspace_data.shape[0]
                imag_data = sens_data['i']  # pull imaginary portion
                num_slices_sens = imag_data.shape[0]
                
                # Verify slice counts match
                if num_slices_kspace != num_slices_sens:
                    raise ValueError(
                        f"Mismatch in slice counts: {kspace_path} has {num_slices_kspace} slices, "
                        f"but {sens_path} has {num_slices_sens} slices"
                    )
                
                # Add slice indices for this file pair
                self.slice_indices.extend(
                    [(kspace_path, sens_path, i) for i in range(num_slices_kspace)]
                )

    def __len__(self):
        """Return total number of slices."""
        return len(self.slice_indices)

    def __getitem__(self, idx):
        """
        Load and return a single slice of k-space data and sensitivity map.
        
        Returns:
            tuple: (kspace_tensor, sens_tensor, target_tensor)
                - kspace_tensor: K-space data (e.g., shape (2, H, W) for real/imag).
                - sens_tensor: Sensitivity map (e.g., shape (num_coils, H, W, 2) for complex).
                - target_tensor: Target for reconstruction (e.g., k-space or image).
        """
        kspace_path, sens_path, slice_idx = self.slice_indices[idx]
        
        # Load k-space slice
        with h5py.File(kspace_path, 'r') as kspace_f:
            kspace = kspace_f['kspace'][slice_idx, ...]  # Shape: (C, H, W), complex-valued
            kspace = np.array(kspace, dtype=np.complex64)
        
        # Load sensitivity map slice
        with h5py.File(sens_path, 'r') as sens_f:
            sens_map = sens_f['sens_maps']
            sens_real = sens_map['r']
            sens_imag = sens_map['i']
            srr = sens_real[slice_idx, ...]  # Shape: (num_coils, H, W)
            sri = sens_imag[slice_idx, ...]
            sens_map = srr + 1j * sri
            sens_map = sens_map.astype('complex64')
        
        # Preprocess k-space (normalize, split real/imag)
        kspace = kspace / np.max(np.abs(kspace)) # Shape: (num_coils, Nx, Ny)

        # Preprocess sensitivity map (normalize, split real/imag)
        # sens_map = sens_map / np.max(np.abs(sens_map)) # shape (num_coils, Nx, Ny)
        # norm_factor = np.sqrt(np.sum(np.abs(sens_map)**2, axis=0))  # [C, H, W]
        # sens_map = sens_map / (norm_factor + 1e-8)

        # Convert to PyTorch tensors
        kspace_tensor = torch.from_numpy(kspace)
        sens_tensor = torch.from_numpy(sens_map)
        
        # Return k-space, sensitivity map, and target (adjust target as needed)
        return kspace_tensor, sens_tensor, kspace_tensor  # Example: target is k-space

def main():
    # Define directories (update these paths as needed)

    args = parse_args()
    kspace_dir = os.path.join(args.data_dir)
    sens_dir = os.path.join(args.data_dir, 'sens_maps')

    os.makedirs(args.save_dir, exist_ok=True)
    # kspace_dir = '/mnt/e/mri/fastMRI/brain'
    # sens_dir = '/mnt/e/mri/fastMRI/brain/sens_maps'

    # logging
    logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(args.save_dir, "train.log"), mode='w')
    ]
    )

    # Create dataset
    dataset = MRIDataset(kspace_dir=kspace_dir, sens_dir=sens_dir)

    # split for validation
    val_fraction = 0.1
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    
    # Set device
    # device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f"data consistency chosen: {args.dc}")
    logging.info(f"grad chosen: {args.grad}")
    logging.info(f"device chosen: {device}")
    logging.info(f"linesearch: {args.ls}")
    if args.ls:
        logging.info(f"grad step size: N/A (linesearch)")
    else:
        logging.info(f"grad step size: {args.alpha}")

    logging.info(f"wavelets: {args.wav}")
    logging.info(f"sharing weights: {args.share}")
    logging.info(f"sample fraction (requested): {args.sf}")

    # Define image size
    # sImg = [256, 256]
    for i, data in enumerate(train_loader):
      k = data[0]
      sImg = k.shape[-2:]
      break

    # Initialize model
    dataconsistency = args.dc
    torch.manual_seed(20250709)

    model = unrolled_net_separate(sImg, device, dc=dataconsistency, grad=args.grad, linesearch=args.ls, alpha=args.alpha, wavelets=args.wav, n = args.n)
    # model = supervised_net(sImg, device, dc=dataconsistency, grad=args.grad, linesearch=args.ls, alpha=args.alpha, wavelets=args.wav, n = args.n, share_weights=args.share)
    # model = supervised_unet(sImg, device)
    model = model.to(device)

    if args.checkpoint != 'false':
        logging.info(f"loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adjusted learning rate

    # Create undersampling mask
    mask = utils.vdSampleMask(sImg, [180, 95], args.sf * np.prod(sImg), maskType='laplace')
    mask = mask > 0
    sFSR = [40, 20]
    fsr = utils.makeFullySampledCenterRegion(sImg, sFSR)
    fsr = fsr > 0
    msk2 = mask | fsr
    # plt.imsave('msktest3.png', msk2, cmap='gray')
    mask = torch.tensor(msk2)
    mask = mask.to(device)

    actual_samples = torch.sum(mask)
    logging.info(f"sample fraction (actual): {float(actual_samples) / np.prod(sImg)}")

    # Define loss function
    if np.abs(args.lambd) < 1e-10:
        if args.sum:
            criterion = lambda x, y: supervised_sse_loss(x, y)
            logging.info(f"loss function: sum || x - y ||_2^2")
        else:
            criterion = lambda x, y: supervised_mse_loss(x, y)
            logging.info(f"loss function: mean || x - y ||_2^2")
    else:
        if args.sum:
            criterion = lambda x, y: supervised_mixed_loss_sum(x, y, args.lambd)
            logging.info(f"loss function: sum || x - y ||_2^2 + {args.lambd} * || x - y ||_1")
        else:
            criterion = lambda x, y: supervised_mixed_loss(x, y, args.lambd)
            logging.info(f"loss function: mean || x - y ||_2^2 + {args.lambd} * || x - y ||_1")

    # Training parameters
    num_epochs = args.epochs

    # best val loss
    best_val_loss = float('inf')

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        logging.info(f"Starting epoch {epoch+1}/{num_epochs} | Training samples: {len(train_loader.dataset)} | Validation samples: {len(val_loader.dataset)}")
        
        for batch_idx, (kspace, sens_maps, target) in enumerate(progress_bar):
            # Move data to device
            kspace = kspace.to(device)  
            sens_maps = sens_maps.to(device)  
            target = target.to(device)         

            # Apply undersampling mask to k-space

            if len(mask.shape) < 3:
                mask_exp = mask[None, None, :, :]  # (1, 1, H, W)
                kspace_undersampled = kspace * mask_exp
            else:
                mask_exp = mask[None, :, :]  # (1, 1, H, W)
                kspace_undersampled = kspace * mask_exp

            # Initialize input for the model (e.g., zero-filled reconstruction)
            ks = torch.fft.ifftshift( torch.fft.ifftn(torch.fft.fftshift(kspace_undersampled, dim=[2, 3]), norm='ortho', dim = [2, 3]), dim = [2, 3])
            zf = torch.sqrt((ks.real ** 2 + ks.imag ** 2).sum(dim=1, keepdim=True))  # SoS

            ## estim noise
            noise_val = ks[..., :25, :25]
            eps = []
            # for i in range(ks.shape[1]):
            #     eps.append(torch.std(noise_val[:, i, ...]))

            ## roemer input
            ks1 = ks * torch.conj(sens_maps)
            x_init = torch.sum(ks1, dim = 1) # dim = 1 is coil dimension

            sens_maps = torch.permute(sens_maps, dims=(0,2,3,1))
            kspace_undersampled = torch.permute(kspace_undersampled, (0,2,3,1))

            # Forward pass
            output = model(x_init, mask, kspace_undersampled, sens_maps, eps)  # Output shape: (batch, H, W)

            # Compute target image (e.g., inverse Fourier transform of fully-sampled k-space)
            itarget = torch.fft.ifftshift( torch.fft.ifftn(torch.fft.fftshift(target, dim=[2, 3]), norm='ortho', dim = [2, 3]), dim = [2, 3])
            itarget = torch.permute(itarget, dims=(0, 2, 3, 1))
            itarget1 = itarget * torch.conj(sens_maps)
            target_image = torch.sum(itarget1, dim=-1) # dim -1 is coil dim

            # target_image = target_image / torch.max(torch.abs(target_image))

            # Compute loss (MSE between reconstructed and target images)
            loss = criterion(output, target_image)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update running loss
            train_loss += loss.item() * kspace.size(0)

            # Update progress bar
            progress_bar.set_postfix({'batch_loss': loss.item()})

        # Compute average epoch loss
        epoch_loss = train_loss / len(train_loader.dataset)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss:.6f}")

        # compute validation loss
        val_loss = evaluate(model, val_loader, device, mask, criterion)
        logging.info(f"Validation Loss: {val_loss:.6f}")

        # save for best val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            tstr = f"best_model.pth"
            checkpoint_path = os.path.join(args.save_dir, tstr)
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Saved best checkpoint at epoch {epoch} to {checkpoint_path}")

        # Optionally save model checkpoint
        if (epoch + 1) % 100 == 0:
            if dataconsistency:
                tstr = f"dc_checkpoint_epoch_{epoch+1}.pth"
            else:
                tstr = f"checkpoint_epoch_{epoch+1}.pth"
            checkpoint_path = os.path.join(args.save_dir, tstr)
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")

    return 0

def test_dataset():
    dirname = '/mnt/e/mri/fastMRI/brain'
    dataset = MRIDataset(kspace_dir=dirname, sens_dir=dirname + '/sens_maps')

    dataloader = DataLoader(
        dataset,
        batch_size=1,          # Adjust based on GPU memory
        shuffle=True,          # Shuffle for training
        num_workers=1,         # Parallel loading (adjust based on system)
        pin_memory=True,       # Faster GPU training
        drop_last=True         # Drop incomplete batch
    )

    for i, data in enumerate(dataloader):
        print(f'data idx {i}')
        kspace = data[0]
        print(f'{kspace.shape}')

    return 0

if __name__ == "__main__":
    # test_dataset()
    main()
