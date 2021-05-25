import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms

import model.model_utils as mu

import os
import numpy as np
import lpips

import model.p_space as p_space

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ITERATIONS = 1300
SAVE_STEP = 100

# OPTIMIZER
LEARNING_RATE = 0.01
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-8

PATH_DIR = "stuff/data/input/"
EXPECTED_RESULTS = "stuff/data/Peihao_result/"
SAVING_DIR = 'stuff/results/improved_embedding_v3/'

#defining function to calculate loss
def calculate_loss(synth_img, reference_img, w_opt, perceptual_net, MSE_Loss, affine_PN, condition_function = None, downsampling_mode = 'bicubic', lambda_v = 0.001):
  
  # get the synth img to [0, 1] to measure the perceptual loss
  synth_img = (synth_img + 1) / 2
  # synth_img = (synth_img-torch.min(synth_img))/(torch.max(synth_img)-torch.min(synth_img))

  # transfor according to condition function
  if condition_function is not None:
    synth_img = condition_function(synth_img)

  # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
  tmp_synth_img = F.interpolate(synth_img, size=(256, 256), mode=downsampling_mode)
  tmp_reference_img = F.interpolate(reference_img, size=(256, 256), mode=downsampling_mode)

  transform = transforms.Compose([
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
  ])

  # Normalize to [-1,1]
  # tmp_synth_img = 2*tmp_synth_img - 1
  # tmp_reference_img = 2*tmp_reference_img - 1
  tmp_synth_img = transform(tmp_synth_img)
  tmp_reference_img = transform(tmp_reference_img)

  # calculate LPIPS Perceptual Loss
  # Features for synth images.
  perceptual_loss = perceptual_net.forward(tmp_reference_img, tmp_synth_img)

  # normalize to [0,1] to measure the MSE loss
  # synth_img = synth_img / 255.0

  # normalize to [-1,1]
  synth_img = 2*synth_img - 1
  reference_img = 2*reference_img - 1
  # synth_img = transform(synth_img)
  # reference_img = transform(reference_img)

  # calculate MSE Loss
  mse_loss = MSE_Loss(synth_img,reference_img) 

  # adding the regulizer part
  regularizer = lambda_v * (torch.linalg.norm(affine_PN(w_opt)) ** 2)

  return mse_loss, perceptual_loss, regularizer

# Loading the pretrained model
G = mu.load_pretrained_model(file_name="ffhq.pkl", space='w',device = DEVICE)

# Pixel-Wise MSE Loss
MSE_Loss = nn.MSELoss(reduction="mean")

# Load VGG16 feature detector. # StyleGANv2 version of metric
perceptual_vgg16 = lpips.LPIPS(net='vgg',version='0.0').to(DEVICE)

# affine transformation to P_N+
C, E, mean, S = p_space.get_PCA_results(q = 512, load = True, device = DEVICE)
affine_PN = p_space.mapping_P_N(C, S, mean)

def run_optimization(data, id, init, 
                    condition_function = None, 
                    sub_fix ="", 
                    downsampling_mode = 'bicubic', 
                    save_loss = False, 
                    lambda_v = 0.001):

  # get the image sample
  basename = data[id]['name'].split(".")[0] + sub_fix
  img = torch.tensor(data[id]['img'].copy(), device = DEVICE, dtype = torch.float32)
  img = img.permute(2, 0, 1).unsqueeze(0)

  img = img / 255.0
  
  # convert according to confition_function
  if condition_function is not None:
    img = condition_function(img)

  # define the init latent
  w_opt = mu.get_initial_latent(init, DEVICE)

  optimizer = optim.Adam({w_opt},lr=LEARNING_RATE,betas=(BETA_1,BETA_2),eps=EPSILON)

  print("Starting Embedding: id: {} name: {}".format(id,basename))
  loss_list=[]
  loss_mse=[]
  loss_perceptual=[]
  latent_list = {}
  for i in range(0,ITERATIONS):
      # reset the gradients
      optimizer.zero_grad()

      # get the synthetic image
      synth_img = G.synthesis(w_opt, noise_mode='const')
      
      # get the loss and backpropagate the gradients
      mse_loss, perceptual_loss, regularizer_term = calculate_loss(synth_img,
                                                img,
                                                # target_features,
                                                w_opt,
                                                perceptual_vgg16, 
                                                MSE_Loss, 
                                                affine_PN,
                                                condition_function,
                                                downsampling_mode,
                                                lambda_v)
      loss = mse_loss + perceptual_loss + regularizer_term
      loss.backward()

      optimizer.step()

      # store the losses metrics
      loss_list.append(loss.item())
      loss_mse.append(mse_loss.item())
      loss_perceptual.append(perceptual_loss.item())

      # every SAVE_STEP, I store the current latent
      if (i +1) % SAVE_STEP == 0:
          print('iter[%d]:\t loss: %.4f\t mse_loss: %.4f\tpercep_loss: %.4f\tregularizer: %.4f' % (i+1,  
                                                                                                   loss.item(), 
                                                                                                   mse_loss.item(), 
                                                                                                   perceptual_loss.item(),
                                                                                                   regularizer_term.item()))
          latent_list[str(i+1)] = w_opt.detach().cpu().numpy()

  # store all the embeddings create during optimization in .npz
  path_embedding_latent = os.path.join(SAVING_DIR, 
                                      "latents/{}_latents_iters_{}_step_{}_{}.npz".format(
                                        basename,
                                        str(ITERATIONS).zfill(6),
                                        str(SAVE_STEP).zfill(4), 
                                        init))
  print("Saving: {}".format(path_embedding_latent))
  np.savez(path_embedding_latent, **latent_list)

  if save_loss:
    loss_file = "loss_plots/{}_loss_iters_{}_step_{}_{}.npy".format(
                                        basename,
                                        str(ITERATIONS).zfill(6),
                                        str(SAVE_STEP).zfill(4), 
                                        init)
    path_loss = os.path.join(SAVING_DIR, loss_file)
    print("Saving Loss: {}".format(path_loss))
    np.save(path_loss, np.array(loss_list))
  return loss_list