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
                # raise FileNotFoundError(f"Sensitivity map file not found: {sens_path}")
            
            # Open both files to get number of slices
            with h5py.File(kspace_path, 'r') as kspace_f, h5py.File(sens_path, 'r') as sens_f:
                kspace_data = kspace_f['kspace']  # Assume dataset key is 'kspace'
                sens_data = sens_f['sens_maps']  # dataset key is 'sensitivity_maps'
                
                num_slices_kspace = kspace_data.shape[0]

                imag_data = sens_data['i'] # pull imaginary portion
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
            srr = sens_real[slice_idx, ...]  # Shape: (num_coils, H, W), complex-valued
            sri = sens_imag[slice_idx, ...]
            sens_map = srr + 1j * sri
        
        # Preprocess k-space (normalize, split real/imag)
        kspace = (kspace - kspace.mean()) / kspace.std()
        kspace = np.stack([kspace.real, kspace.imag], axis=0)  # Shape: (2, H, W)
        
        # Preprocess sensitivity map (normalize, split real/imag)
        sens_map = (sens_map - sens_map.mean()) / sens_map.std()
        sens_map = np.stack([sens_map.real, sens_map.imag], axis=-1)  # Shape: (num_coils, H, W, 2)
        
        # Convert to PyTorch tensors
        kspace_tensor = torch.from_numpy(kspace).float()
        sens_tensor = torch.from_numpy(sens_map).float()
        
        # Return k-space, sensitivity map, and target (adjust target as needed)
        return kspace_tensor, sens_tensor, kspace_tensor  # Example: target is k-space

def main():

  ## load data
  ## maybe get the stuff from training the fastMRI data ? data loader or something

  # create dataset from the class above
  dataset = 0
  # Create DataLoader
  dataloader = DataLoader(
      dataset,
      batch_size=4,          # Adjust based on GPU memory
      shuffle=True,          # Shuffle for training
      num_workers=4,         # Parallel loading (adjust based on system)
      pin_memory=True,       # Faster GPU transfer
      drop_last=True         # Drop incomplete batch
  )

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  ## make model
  sImg = [256, 256] # get the data size here

  dataconsistency = True
  multicoil = True
  model = unrolled_net(sImg, device, dataconsistency, multicoil)

  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

  ## make undersampling mask
  mask = utils.vdSampleMask(sImg, [30, 50], 0.25*np.prod(sImg), maskType = 'laplace')

  ## apply to data

  ## get sensitivity maps

  ## train
  model.train()

  num_epochs = 100

  for i in range(num_epochs):
    ## do training
    train_loss = 0 

  #### batch here
  return 0

def test_dataset():
    dirname = '/mnt/e/mri/fastMRI/brain'
    dataset = MRIDataset(kspace_dir = dirname, sens_dir = dirname + '/sens_maps')

    dataloader = DataLoader(
      dataset,
      batch_size=1,          # Adjust based on GPU memory
      shuffle=True,          # Shuffle for training
      num_workers=1,         # Parallel loading (adjust based on system)
      pin_memory=True,       # Faster GPU transfer
      drop_last=True         # Drop incomplete batch
    )

    for i, data in enumerate(dataloader):
        print(f'data idx {i}')
        kspace = data[0]



    return 0

if __name__ == "__main__":
    test_dataset()
  # main()