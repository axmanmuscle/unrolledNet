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