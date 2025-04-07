"""
new training script for zero shot
want the ability to loop over lots of parameters and let it run overnight
"""

import numpy as np
import glob
import math_utils
import torch
import os
import h5py
import utils
from tqdm import tqdm
from zs_unrollednet import ZS_Unrolled_Network, ZS_Unrolled_Network_gd, ZS_Unrolled_Network_wavelets
import matplotlib.pyplot as plt
import scipy.io as sio
import gc


def training_loop(training_data, val_data, val_mask, tl_masks,
                  model, loss_fun, sMaps, optimizer, data_consistency, 
                  val_stop_training = 15, num_epochs = 50, device = torch.device('cpu'),
                  directory=os.getcwd()):
  """
  todo:
    - the data in is undersampled k-space. split into train and validation (oh this should be done earlier)
    - the training data is split out and run through the model. compare loss
  """
  if len(sMaps) == 0:
    sc = True
  else:
    sc = False
  model.train()

  training_data = training_data.to(device)
  val_data = val_data.to(device)
  val_mask = val_mask.to(device)

  ## TODO these lines are broken
  if data_consistency:
    training_mask = torch.abs(training_data) > 0
    all_tdata_consistency = training_data
    all_tdata_consistency[~training_mask] = 0

    all_data = training_data + val_data
    alldata_mask = torch.abs(all_data) > 0
    alldata_consistency = all_data
    alldata_consistency[~alldata_mask] = 0

  vl_min = 10000

  tl_ar = []
  vl_ar = []
  ep = 0
  val_loss_tracker = 0

  model_fname = f"best_{len(tl_masks)}.pth"
  if data_consistency:
    model_fname = "dc_"+model_fname

  while ep < num_epochs and val_loss_tracker < val_stop_training:
    avg_train_loss = 0.0
    for jdx, tl_mask in tqdm(enumerate(tl_masks)):
      #print(f'subiter {jdx}')
      tmask = tl_mask[0]
      lmask = tl_mask[1]

      tmask = torch.tensor(tmask)
      lmask = torch.tensor(lmask)
      tmask = tmask.to(device)
      lmask = lmask.to(device)

      if sc:
        tdata = training_data * tmask
      else:
        tdata = training_data * tmask.unsqueeze(2) # TODO unsqueeze tmask
      if data_consistency:
        #tdata_consistency = tdata[:, :, tmask > 0]
        tdata_consistency = tdata
        if sc:
          im = torch.fft.ifftshift( torch.fft.ifftn( torch.fft.fftshift( tdata, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
        else:
          im = torch.sum( torch.fft.ifftshift( torch.fft.ifftn( torch.fft.fftshift( tdata, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) ), -1)
        
        ## for wavelets
        wavSplit = torch.tensor(math_utils.makeWavSplit(torch.squeeze(im).shape))
        im_in = torch.zeros_like(im)
        im_in[..., :] = math_utils.wtDaubechies2(torch.squeeze(im), wavSplit)
        im_in = im_in.to(device)
        out = model(im_in, tmask>0, tdata_consistency)
      else:
        out = model(tdata) 
      #if jdx < 1:
      if sc:
        train_loss = loss_fun(out, training_data, lmask)
      else:
        train_loss = loss_fun(out, training_data, lmask, sMaps) # gt needs to come from the data loader?
      avg_train_loss += train_loss.cpu().data
      optimizer.zero_grad()
      train_loss.backward()
      optimizer.step()

    if data_consistency:
      if sc:
        val_im = torch.fft.ifftshift( torch.fft.ifftn( torch.fft.fftshift( training_data, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
      else:
        val_im = torch.sum( torch.fft.ifftshift( torch.fft.ifftn( torch.fft.fftshift( training_data, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) ), -1)
      val_out = model(val_im, training_mask, all_tdata_consistency)
    else:
      val_out = model(training_data)

    if sc:
      val_loss = loss_fun(val_out, val_data, val_mask)
    else:
      val_loss = loss_fun(val_out, val_data, val_mask, sMaps)
    vl_data = val_loss.cpu().data

    tl_ar.append(avg_train_loss / (jdx+1))
    vl_ar.append(vl_data)

    checkpoint = {
      "epoch": ep,
      "val_loss_min": vl_data,
      "model_state": model.state_dict(),
      "optim_state": optimizer.state_dict()
    }

    if vl_data <= vl_min:
      vl_min = vl_data
      torch.save(checkpoint, os.path.join(directory, model_fname))
      val_loss_tracker = 0
    else:
      val_loss_tracker += 1

    ep += 1

    # optimizer.zero_grad()
    # train_loss.backward()
    # optimizer.step()

    print('on step {} of {} with tl {:.2E} and vl {:.2E}'.format(ep, num_epochs, float(tl_ar[-1]), float(val_loss.data)))

    ## optional
    # save images at each epoch
    if sc:
      all_im = torch.fft.ifftshift( torch.fft.ifftn( torch.fft.fftshift( all_data, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
    else:
      all_im = torch.sum( torch.fft.ifftshift( torch.fft.ifftn( torch.fft.fftshift( all_data, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) ), -1)
    out = model(all_im, alldata_mask, alldata_consistency)
    tstr = f'output_epoch{ep}.png'

    out = out.cpu()
    oc = np.squeeze(out.detach().numpy())
    img_dir = os.path.join(directory, 'imgs/')

    if not os.path.isdir(img_dir):
      os.mkdir(img_dir)
    plt.imsave(os.path.join(img_dir, tstr), np.abs( oc ), cmap='grey')

    plt.clf()
    del all_im, out, oc
    gc.collect()
    

  plt.plot(tl_ar)
  plt.plot(vl_ar)
  plt.legend(['training loss', 'val loss'])
  if data_consistency:
    loss_str = 'dc_loss_fig.png'
  else:
    loss_str = 'nodc_loss_fig.png'
  plt.savefig(os.path.join(directory, loss_str))

  best_checkpoint = torch.load(os.path.join(directory, model_fname))
  model.load_state_dict(best_checkpoint['model_state'])
  all_data = training_data+val_data

  if data_consistency:
    if sc:
      all_im = torch.fft.ifftshift( torch.fft.ifftn( torch.fft.fftshift( all_data, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
    else:
      all_im = torch.sum( torch.fft.ifftshift( torch.fft.ifftn( torch.fft.fftshift( all_data, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) ), -1)
      
    out = model(all_im, alldata_mask, alldata_consistency)
    tstr = 'dc_output.png'
  else:
    out = model(all_data)
    tstr = 'nodc_output.png'
  out = out.cpu()

  oc = np.squeeze(out.detach().numpy())

  #im = math_utils.kspace_to_imspace(oc)
  #plt.imshow( np.abs( im ), cmap='grey')
  plt.imsave(os.path.join(directory, tstr), np.abs( oc ), cmap='grey')

  plt.clf()

def test_only_gd(ks, model, directory, mask):

  # x0 = torch.rand(ks.shape) + 1j * torch.rand(ks.shape)
  # x0 = x0.to(model.device)

  ## multicoil only
  x0 = torch.sum( torch.fft.ifftshift( torch.fft.ifftn( torch.fft.fftshift( ks, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) ), -1)

  ks = ks.to(model.device)
  mask = mask.to(model.device)
  out = model(x0, mask, ks)
  out = out.cpu()
  oc = np.squeeze(out.detach().numpy())

  tstr = 'dc_output.png'
  plt.imsave(os.path.join(directory, tstr), np.abs( oc ), cmap='grey')

  plt.clf()

def run_training(ks, sImg, sMask, sMaps, rng, samp_frac, train_frac, 
                 train_loss_split_frac, k, dc, results_dir,
                 val_stop_training, num_epochs=100):
                   
  usMask = utils.vdSampleMask(sMask, [30, 30], np.round(samp_frac * np.prod(sImg)))
  # usMask = utils.undersample_kspace_gaussian(sMask, rng, samp_frac)
  #undersample_mask = np.zeros(sImg)
  #undersample_mask[:, left_idx:right_idx] = usMask
  train_mask, val_mask = utils.mask_split(usMask, rng, train_frac)

  # usMask = torch.tensor(usMask)
  # train_mask = torch.tensor(train_mask)
  # val_mask = torch.tensor(val_mask)

  if len(sMaps) == 0:
    sc = True
  else:
    sc = False

  if sc:
    sub_kspace = torch.tensor( usMask ) * ks
    training_kspace = torch.tensor( train_mask ) * ks
    val_kspace = torch.tensor( val_mask ) * ks
  else:
    sub_kspace = torch.tensor( usMask[:, :, np.newaxis] ) * ks
    training_kspace = torch.tensor( train_mask[:, :, np.newaxis] ) * ks
    val_kspace = torch.tensor( val_mask[:, :, np.newaxis] ) * ks

  sub_kspace = torch.tensor(sub_kspace)
  training_kspace = torch.tensor(training_kspace)
  val_kspace = torch.tensor(val_kspace)
  val_mask = torch.tensor(val_mask)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  wavelets = True

  if sc:
    print('single coil')
    model = ZS_Unrolled_Network_gd(sImg, device, n=40)
  else:
    print('multi coil')
    sMaps = sMaps.to(device)
    if wavelets:
      model = ZS_Unrolled_Network_wavelets(sImg, device, sMaps, 2)
    # model = ZS_Unrolled_Network_gd(sImg, device,sMaps, 20) 
    # model = ZS_Unrolled_Network(sImg, device,sMaps, 3)    

  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

  # training_kspace = training_kspace[None, None, :, :]
  # val_kspace = val_kspace[None, None, :, :]

  tl_masks = []
  for idx in range(k):
    tm, lm = utils.mask_split(train_mask, rng, train_loss_split_frac)
    tl_masks.append((tm, lm))

  directory = f'vd_sf{int(samp_frac*100)}p_tf{int(train_frac*100)}p_k{k}_vst{val_stop_training}'
  if dc:
    directory = "dc_" + directory
  directory = os.path.join(results_dir, directory)

  if not os.path.isdir(directory):
    os.mkdir(directory)

  if sc:
    oc = np.fft.ifftshift( np.fft.ifftn( np.fft.fftshift( sub_kspace, axes=(-1,-2)),  axes=(-1,-2) ), axes=(-1,-2))
    plt.imsave(os.path.join(directory, 'zero_filled.png'), np.abs( np.squeeze( oc ) ), cmap='grey')
    oc = np.fft.ifftshift( np.fft.ifftn( np.fft.fftshift( ks, axes=(-1,-2)),  axes=(-1,-2) ), axes=(-1,-2))
    plt.imsave(os.path.join(directory, 'gt.png'), np.abs( np.squeeze( oc ) ), cmap='grey')

  else:
    oc = np.sum(np.fft.ifftshift( np.fft.ifftn( np.fft.fftshift( sub_kspace, axes=(2,3)),  axes=(2,3) ), axes=(2,3)), axis=-1)
    plt.imsave(os.path.join(directory, 'zero_filled.png'), np.abs( np.squeeze( oc ) ), cmap='grey')
    oc = np.fft.ifftshift( np.fft.ifftn( np.fft.fftshift( ks, axes=(2,3)),  axes=(2,3) ), axes=(2,3))
    im = utils.mri_reconRoemer(np.squeeze(oc), sMaps.cpu().numpy())
    plt.imsave(os.path.join(directory, 'gt.png'), np.abs( np.squeeze( im ) ), cmap='grey')

  # test_only_gd(sub_kspace, model, directory, torch.tensor(usMask > 0))
  training_loop(training_kspace, val_kspace, val_mask, tl_masks,
              model, math_utils.unrolled_loss_mixed, sMaps, optimizer, dc, 
              val_stop_training, num_epochs, device,
              directory)

def test_mc_gd():
  rng = np.random.default_rng(20250313)
  data = sio.loadmat('/home/alex/Documents/research/mri/data/brain_data.mat')
  # data = sio.loadmat('/Users/alex/Documents/School/Research/Dwork/dataConsistency/brain_data.mat')
  kSpace = data['d2']
  kSpace = kSpace / np.max(np.abs(kSpace))
  sMaps = data['smap']
  sMaps = sMaps / np.max(np.abs(sMaps))

  sImg = kSpace.shape[0:2]

  # results_dir = '/Users/alex/Documents/School/Research/Dwork/dataConsistency/results/mc_41'
  results_dir = '/home/alex/Documents/research/mri/results/426_wavtests'
  sMaps = torch.tensor(sMaps, dtype=torch.complex64)

  kSpace2 = torch.zeros(1, *kSpace.shape, dtype=torch.complex64)
  kSpace2[0, :, :] = torch.tensor(kSpace)
  kSpace2 = kSpace2.unsqueeze(1)
  

  samp_fracs = [0.4, 0.25, 0.15, 0.1, 0.05]
  train_fracs = [0.8]
  train_loss_split_frac = 0.8
  k_s = [50]
  dcs = [True]
  val_stop_trainings = [15]

  for sf in samp_fracs:
    for tf in train_fracs:
      for k in k_s:
        for vst in val_stop_trainings:
          for dc in dcs:

            run_training(kSpace2, sImg, sImg, sMaps, rng, 
                      sf, tf, train_loss_split_frac, 
                      k, dc, results_dir, vst, 50)
  return 0


def test_sc():
  """
  read in data and decide what to iterate over
  """
  rng = np.random.default_rng(20250313)
  # data = sio.loadmat('/home/alex/Documents/research/mri/data/brain_data.mat')
  data = sio.loadmat('/Users/alex/Documents/School/Research/Dwork/dataConsistency/brain_data.mat')
  kSpace = data['d2']
  kSpace = kSpace / np.max(np.abs(kSpace))
  sMaps = data['smap']
  sMaps = sMaps / np.max(np.abs(sMaps))

  # results_dir = '/home/alex/Documents/research/mri/results/sc_only_gd_test_330'
  results_dir = '/Users/alex/Documents/School/Research/Dwork/dataConsistency/results/sc_only_gd'
  im2 = np.fft.fftshift( np.fft.ifftn( np.fft.ifftshift( kSpace, axes=(0,1)), axes=(0,1)), axes=(0,1))
  recon = utils.mri_reconRoemer(im2, sMaps)

  ks_sc = np.fft.fftshift( np.fft.fftn( np.fft.ifftshift( recon, axes=(0,1)), axes=(0,1)), axes=(0,1))

  ks_sc = torch.tensor(ks_sc, dtype=torch.complex64)

  sImg = ks_sc.shape
  kSpace = ks_sc.unsqueeze(0)
  kSpace = kSpace.unsqueeze(0)

  samp_fracs = [0.4, 0.2, 0.15]
  train_fracs = [0.9]
  train_loss_split_frac = 0.85
  k_s = [50]
  dcs = [True]
  val_stop_trainings = [15]

  sMaps = []

  for sf in samp_fracs:
    for tf in train_fracs:
      for k in k_s:
        for vst in val_stop_trainings:
          for dc in dcs:

            run_training(kSpace, sImg, sImg, sMaps, rng, 
                      sf, tf, train_loss_split_frac, 
                      k, dc, results_dir, vst, 100)

def main():
  """
  read in data and decide what to iterate over
  """
  rng = np.random.default_rng(20250313)
  # data = sio.loadmat('/home/alex/Documents/research/mri/data/brain_data.mat')
  data = sio.loadmat('/Users/alex/Documents/School/Research/Dwork/dataConsistency/brain_data.mat')
  kSpace = data['d2']
  kSpace = kSpace / np.max(np.abs(kSpace))
  sMaps = data['smap']
  sMaps = sMaps / np.max(np.abs(sMaps))

  sImg = kSpace.shape[0:2]

  # results_dir = '/home/alex/Documents/research/mri/results/426_wavtests'
  results_dir = '/Users/alex/Documents/School/Research/Dwork/dataConsistency/results/407_wavtests'

  # mask = vdSampleMask(kSpace.shape[0:2], [30, 30], np.round(np.prod(kSpace.shape[0:2]) * 0.4))
  # us_kSpace = kSpace*mask[:, :, np.newaxis]

  test_nBatch = 1
  sMaps = torch.tensor(sMaps, dtype=torch.complex64)
  # mask = torch.tensor(mask)
  # mask = mask.to(torch.bool)
  # mk2 = torch.zeros(test_nBatch, *sImg, dtype=torch.bool)
  kSpace2 = torch.zeros(test_nBatch, *kSpace.shape, dtype=torch.complex64)
  for i in range(test_nBatch):
      # mk2[i, :, :] = mask
      kSpace2[i, :, :] = torch.tensor(kSpace)

  # mk2 = mk2.unsqueeze(1)
  kSpace2 = kSpace2.unsqueeze(1)
  # us_kSpace[~mask] = 0
  # ks = torch.tensor(us_kSpace)

  samp_fracs = [0.25]
  train_fracs = [0.9]
  train_loss_split_frac = 0.8
  k_s = [4]
  dcs = [True]
  val_stop_trainings = [3]

  for sf in samp_fracs:
    for tf in train_fracs:
      for k in k_s:
        for vst in val_stop_trainings:
          for dc in dcs:

            run_training(kSpace2, sImg, sImg, sMaps, rng, 
                      sf, tf, train_loss_split_frac, 
                      k, dc, results_dir, vst, 1)
  return 0
  
if __name__ == "__main__":
  main()
  # test_mc_gd()
  # test_sc()
