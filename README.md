## Unrolled Network
Implementation of unrolled network for MRI reconstruction

## To Do
Need to get the hooks on so this can train zero shot self supervised style

 - Get sens maps for fast MRI data or just test on brain
 - write training loop for unrolled net
 - debug all the zero shot nonsense


 I think we may need the unrolled network to just take the sensemaps and the mask and just make M, F, S on the fly? Rather than fixing A to begin with. For zero-shot training we'll need to change the sampling mask

 ## Debugging
 Something is clearly going wrong with it. We need to see if the steps of gradient descent and the prox without the NN are doing the correct thing.

 Before that, why don't we double check that SENSE reconstruction is lining up with Nick's implementation.

 ## 320
 Change to VD sampling mask

 ## 41
 This seems to be working with the gradient line search - although now we need to add wavelets (yayyyyyy)

 ## 4/28
 writing the non-zero shot version of this now. needs to take in the sensitivity maps and the initial image (or wavelet coeffs?)

 ### TO DO TODAY:
  - [ ] write this
  - [ ] write paper
  - [ ] work on fetal data for a little

#### Changes for supervised
 - SMaps at ``forward`` time not construction
 - same with masks i guess


#### For Fetal data
since the data isnt centered in the space domain we need to be careful about doing the fftshift *after* the ffts during the prox blocks, etc.

#### For Supervised
Have the dataset written now it's time to test a simple training loop

if that works then we can just throw it on the supercomputer although we'll need to get the sensmaps over there too
 ezpz do that sometime tomorrow



## Supervised Network

## 🕐 **2-Hour Task Breakdown**

### 🔹 **Hour 1: Debug + Refactor Core Training Pipeline**

**⏱ 0:00–0:10 — Fix Dataset Issues**

* [ ] Fix `__getitem__` signature (`def __getitem__(self, idx):`)
* [ ] Ensure returned `kspace_tensor` and `sens_tensor` are properly normalized
* [ ] Make sure `kspace_tensor` remains complex (no real/imag stack unless your network expects it)

**⏱ 0:10–0:25 — Fix DataLoader and Batch Construction**

* [ ] Confirm that `mask` is broadcastable to k-space shape: `[1, H, W]`
* [ ] Apply it cleanly: `kspace_undersampled = kspace * mask[None, None, :, :]`

**⏱ 0:25–0:45 — Fix Forward Pass**

* [ ] Call `model(x_init, mask, kspace_undersampled, sens_maps)` correctly
* [ ] Make sure `x_init` has correct shape `[B, 1, H, W]`
* [ ] Print output shape from model and verify it is what you expect (image, wavelet coeffs, etc.)

**⏱ 0:45–1:00 — Swap to Image-Domain Target**

* [ ] Compute `target_image = ifft2c(kspace)` (use `math_utils` or define it)
* [ ] Apply Roemer combination: `target_image = coil_combine(target_image, sens_maps)`
* [ ] Compare to `output_image = iwtDaubechies2(output, model.wavSplit)`
* [ ] Use `MSELoss(output_image.abs(), target_image.abs())`

---

### 🔹 **Hour 2: Utilities + Evaluation Clean-up**

**⏱ 1:00–1:20 — Add Utility for Coil Combination**

* [ ] Add `coil_combine(image, sMaps)` using Roemer equation:

  ```python
  def coil_combine(image, sens_maps):
      return torch.sum(torch.conj(sens_maps) * image, dim=1)
  ```
* [ ] Check output type: shape `[B, H, W]`, complex

**⏱ 1:20–1:40 — Normalize and Visualize**

* [ ] Normalize image magnitudes to `[0, 1]` or their own maximums
* [ ] Optionally plot a few slices (e.g. `matplotlib.imshow(output_image[0].abs().cpu())`) for sanity

**⏱ 1:40–2:00 — Clean Logging + Loss Tracking**

* [ ] Print loss values and confirm they're not NaN
* [ ] Save a few reconstructions + targets to disk every N epochs for visual debugging
* [ ] Optionally wrap model in `torch.nn.DataParallel` if you plan to scale up later

---

## ✅ **What You'll Have at the End**

* Clean dataset and dataloader
* Proper model input/output wiring
* Image-domain loss with Roemer coil combination
* Valid reconstructions with monitored training loss
* A stable, interpretable training loop to begin fine-tuning

Would you like me to generate the updated versions of `__getitem__`, the training loop, and a `coil_combine()` function to get you started?


## TODO
Supervised training is almost debugged - you have the initial image now. Need to double check whether the initial input into the network should be the image or the wavelet coeffs
then convert the target into the image as well, make sure loss is computed correctly, then train!

## List of Items to do
 - better commenting - make sure you know what permutations, etc. are happening
 - logging?
 - validation set
 - mixed precision

## update
wrote a bunch of the above except for the comments. run the script as
```
python run_training_supervised_gpt.py --data_dir /mnt/e/mri/fastMRI/brain_small/ --save_dir /home/mcmanus/code/unrolledNet/results/ --epochs 1 
```

Running on supercomputer
now generate results using other networks

## Zero Shot Unrolled
The paper uses a much smaller network (~500k parameters) and shares the network across unrolled blocks. worth trying if its quick

## Zero Shot Unet
Need to update saving loss plots and images