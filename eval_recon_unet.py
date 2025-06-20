import os
import torch
import numpy as np
import argparse
import math_utils
import utils
import torchvision
import matplotlib.pyplot as plt
from unet import build_unet
from run_training_supervised_gpt import MRIDataset
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from tqdm import tqdm
from model_supervised import supervised_net

def save_image(tensor, path, cmap='gray'):
    img = tensor.squeeze().cpu().numpy()
    img = np.abs(img)
    img = img / img.max()
    plt.imsave(path, img, cmap=cmap)

def compute_metrics(recon, gt):
    # recon, gt: torch tensors with shape (1, H, W) or (H, W)
    recon_np = recon.squeeze().cpu().numpy()
    gt_np = gt.squeeze().cpu().numpy()
    recon_np /= recon_np.max()
    gt_np /= gt_np.max()

    mse = np.mean((recon_np - gt_np) ** 2)
    psnr = compare_psnr(gt_np, recon_np, data_range=1.0)
    ssim = compare_ssim(gt_np, recon_np, data_range=1.0)
    return mse, psnr, ssim

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./eval_results')
    parser.add_argument('--dc', action='store_true', help="Enforce Data Consistency or not")
    parser.add_argument('--grad', action='store_true', help="Grad descent steps or not")
    parser.add_argument('--ls', action='store_true', help="Do grad descent line search")
    parser.add_argument('--wav', action='store_true', help="wavelets")
    parser.add_argument('--alpha', type=float, default=1e-3, help="(optional) grad descent default step size")
    parser.add_argument('--n', type=int, default=1, help = 'number of unrolled iters to do (default 1)')
    parser.add_argument('--share', action='store_true', help="when TRUE, this option makes the unet at all layers of the unrolled network use the same weights")

    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MRIDataset(kspace_dir=args.data_dir, sens_dir=os.path.join(args.data_dir, '../subset_sensmaps/sens_maps'))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    sImg = dataset[0][0].shape[-2:]
    print(sImg)
    wavSplit = torch.tensor(math_utils.makeWavSplit(sImg))
    model = supervised_net(sImg, device, args.dc, args.grad, linesearch=args.ls, alpha=args.alpha, wavelets=args.wav, n=args.n, share_weights=args.share)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    model.to(device)

    mask = utils.vdSampleMask(sImg, [180, 95], 0.075 * np.prod(sImg), maskType='laplace')
    mask = (torch.tensor(mask) > 0).to(device)

    all_mse, all_psnr, all_ssim = [], [], []

    with torch.no_grad():
        for idx, (kspace, sens_maps, _) in enumerate(tqdm(loader)):
            kspace = kspace.to(device)
            sens_maps = sens_maps.to(device)

            # Create zero-filled recon
            if len(mask.shape) < 3:
                mask_exp = mask[None, None, :, :]
                kspace_undersampled = kspace * mask_exp
            else:
                kspace_undersampled = kspace.clone()
                kspace_undersampled[~mask] = 0


            # Initialize input for the model (e.g., zero-filled reconstruction)
            ks = torch.fft.ifftshift( torch.fft.ifftn(torch.fft.fftshift(kspace_undersampled, dim=[2, 3]), dim = [2, 3]), dim = [2, 3])
            zf = torch.sqrt((ks.real ** 2 + ks.imag ** 2).sum(dim=1, keepdim=True))  # SoS

            ## roemer input
            ks1 = ks * torch.conj(sens_maps)
            x_init = torch.sum(ks1, dim = 1) # dim = 1 is coil dimension

            sens_maps = torch.permute(sens_maps, dims=(0,2,3,1))
            kspace_undersampled = torch.permute(kspace_undersampled, (0,2,3,1))

            # Forward pass
            output = model(x_init, mask, kspace_undersampled, sens_maps)  # Output shape: (batch, H, W)

            # Ground truth image
            gt = torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftshift(kspace, dim=[2, 3]), dim=[2, 3]), dim=[2, 3])
            gt = torch.permute(gt, dims=(0, 2, 3, 1))
            gt = gt * torch.conj(sens_maps)
            gt = torch.sum(gt, dim=-1, keepdim=True)
            gt = gt / gt.abs().amax(dim=(-3, -2), keepdim=True)
            gt = torch.abs(gt)

            # Input to network
            # ks1 = ks * torch.conj(sens_maps)
            ks1 = ks * torch.conj(ks) # SoS
            x_init = torch.sum(ks1, dim=1, keepdim=True)

            # Network output
            output = output / output.abs().amax(dim=(-2, -1), keepdim=True)
            output = torch.abs(output)

            # Save reconstructions
            save_image(gt, os.path.join(args.save_dir, f"{idx:04d}_gt.png"))
            save_image(x_init.abs(), os.path.join(args.save_dir, f"{idx:04d}_input.png"))
            save_image(zf, os.path.join(args.save_dir, f"{idx:04d}_zf.png"))
            save_image(output, os.path.join(args.save_dir, f"{idx:04d}_recon.png"))

            # Compute metrics
            mse, psnr, ssim = compute_metrics(output, gt)
            all_mse.append(mse)
            all_psnr.append(psnr)
            all_ssim.append(ssim)

    print(f"\nEvaluation Results on {len(dataset)} slices:")
    print(f"  Avg MSE  : {np.mean(all_mse):.6f}")
    print(f"  Avg PSNR : {np.mean(all_psnr):.2f} dB")
    print(f"  Avg SSIM : {np.mean(all_ssim):.4f}")

if __name__ == "__main__":
    main()
