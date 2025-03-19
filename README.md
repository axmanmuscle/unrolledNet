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