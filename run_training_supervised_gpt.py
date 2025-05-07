"""
supervised training for the unrolled network
"""

import numpy as np
import glob
import math_utils
import torch
import os
import h5py
import utils
from tqdm import tqdm
from unrolled_net import unrolled_net
from torch.utils.data import Dataset, DataLoader


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
        sens_map = sens_map / np.max(np.abs(sens_map)) # shape (num_coils, Nx, Ny)

        # Convert to PyTorch tensors
        kspace_tensor = torch.from_numpy(kspace)
        sens_tensor = torch.from_numpy(sens_map)
        
        # Return k-space, sensitivity map, and target (adjust target as needed)
        return kspace_tensor, sens_tensor, kspace_tensor  # Example: target is k-space

def main():
    # Define directories (update these paths as needed)
    kspace_dir = '/mnt/e/mri/fastMRI/brain'
    sens_dir = '/mnt/e/mri/fastMRI/brain/sens_maps'

    # Create dataset
    dataset = MRIDataset(kspace_dir=kspace_dir, sens_dir=sens_dir)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=1,          # Adjust based on GPU memory
        shuffle=False,          # Shuffle for training TODO CHange
        num_workers=1,         # Parallel loading (adjust based on system)
        pin_memory=True,       # Faster GPU transfer
        drop_last=True         # Drop incomplete batch
    )

    # Set device
    device = torch.device('cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define image size
    # sImg = [256, 256]
    for i, data in enumerate(dataloader):
      k = data[0]
      sImg = k.shape[-2:]
      break

    # Initialize model
    wavSplit = torch.tensor(math_utils.makeWavSplit(sImg))
    dataconsistency = True
    multicoil = True
    model = unrolled_net(sImg, device, 2, dataconsistency, multicoil)
    model = model.to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adjusted learning rate

    # Create undersampling mask
    mask = utils.vdSampleMask(sImg, [50, 30], 0.05 * np.prod(sImg), maskType='laplace')
    mask = mask > 0
    mask = torch.tensor(mask)
    mask = mask.to(device)

    # Define loss function
    criterion = torch.nn.MSELoss()

    # Training parameters
    num_epochs = 100

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (kspace, sens_maps, target) in enumerate(progress_bar):
            # Move data to device
            kspace = kspace.to(device)  
            sens_maps = sens_maps.to(device)  
            target = target.to(device)         

            # Apply undersampling mask to k-space
            kspace_undersampled = kspace.clone()
            if len(mask.shape) < 3:
                kspace_undersampled[:, :, ~mask] = 0
            else:
                kspace_undersampled[~mask] = 0

            # Initialize input for the model (e.g., zero-filled reconstruction)
            ks = torch.fft.ifftshift( torch.fft.ifftn(torch.fft.fftshift(kspace_undersampled, dim=[2, 3]), dim = [2, 3]), dim = [2, 3])
            ks1 = ks * torch.conj(sens_maps)
            x_init = torch.sum(ks1, dim = 1) # dim = 1 is coil dimension
            x_init = x_init.unsqueeze(1)

            wx_init = torch.zeros_like(x_init)
            wx_init[..., :, :] = math_utils.wtDaubechies2(torch.squeeze(x_init), wavSplit)

            sens_maps = torch.permute(sens_maps, dims=(0,2,3,1))

            kspace_undersampled = torch.permute(kspace_undersampled, (0,2,3,1))
            kspace_undersampled = kspace_undersampled.unsqueeze(1)

            # Forward pass
            output = model(wx_init, mask, kspace_undersampled, sens_maps)  # Output shape: (batch, H, W)

            # Convert output to image space if needed (assuming output is in wavelet domain)

            # Compute target image (e.g., inverse Fourier transform of fully-sampled k-space)
            itarget = torch.fft.ifftshift( torch.fft.ifftn(torch.fft.fftshift(target, dim=[2, 3]), dim = [2, 3]), dim = [2, 3])
            itarget = torch.permute(itarget, dims=(0, 2, 3, 1))
            itarget1 = itarget * torch.conj(sens_maps)
            target_image = torch.sum(itarget1, dim=-1) # dim 1 is coil dim
            target_image = target_image.unsqueeze(1)

            # Compute loss (MSE between reconstructed and target images)
            loss = criterion(output.abs(), target_image.abs())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update running loss
            train_loss += loss.item() * kspace.size(0)

            # Update progress bar
            progress_bar.set_postfix({'batch_loss': loss.item()})

        # Compute average epoch loss
        epoch_loss = train_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss:.6f}")

        # Optionally save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")

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