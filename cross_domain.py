import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms

import model.model_utils as mu
import model.p_space as p_space
import utils.data as u_data
import utils.images as u_images

import os
import numpy as np
import lpips


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ITERATIONS = 2000
SAVE_STEP = 100

# OPTIMIZER
LEARNING_RATE = 0.01
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-8

PATH_DIR = "stuff/data/input/"
EXPECTED_RESULTS = "stuff/data/Peihao_result/"
SAVING_DIR = 'stuff/results/cross_domain/'

# Loading the pretrained model
G = mu.load_pretrained_model(file_name="anime_gan.pkl", space='w+',device = DEVICE)

# Pixel-Wise MSE Loss
MSE_Loss = nn.MSELoss(reduction="mean").to(DEVICE)

# Load VGG16 feature detector. # StyleGANv2 version of metric
perceptual_vgg16 = lpips.LPIPS(net='vgg').to(DEVICE)

# affine transformation to P_N+
C, E, mean, S = p_space.get_PCA_results(G, DEVICE, load=True)
map2PN = p_space.mapping_P_N(C, S, mean)

def run_optimization(data, id, init, 
                    sub_fix ="", 
                    save_loss = False, 
                    lambda_v = 0.001):

  # get the image sample
  basename = data[id]['name'].split(".")[0] + sub_fix

  big_image = transforms.ToTensor()(data[id]['img']).unsqueeze(0).to(DEVICE)
  big_image = u_images.lanczos_transform(data[id]['img'], DEVICE, size = 512)
  small_image = u_images.lanczos_transform(data[id]['img'], DEVICE, size = 256)
  
  big_image = transforms.Normalize(
      mean = [0.5, 0.5, 0.5], 
      std = [0.5, 0.5, 0.5])(big_image)
  small_image = transforms.Normalize(
      mean = [0.5, 0.5, 0.5], 
      std = [0.5, 0.5, 0.5])(small_image)

  #print("small_image: ", small_image.size(), small_image.max(), small_image.min())

  # define the init latent
  w_opt = mu.get_initial_latent(init, G, DEVICE, high_gpu_memory = True)
  
  optimizer = optim.Adam({w_opt},lr=LEARNING_RATE,betas=(BETA_1,BETA_2),eps=EPSILON)

  synth_img = G.synthesis(w_opt, noise_mode='const')

  print("Starting Embedding: id: {} name: {}".format(id,basename))
  loss_list=[]
  loss_mse=[]
  loss_perceptual=[]
  latent_list = {}
  for i in range(0,ITERATIONS):
      # reset the gradients
      optimizer.zero_grad()

      # get the generated image
      big_gen_img = G.synthesis(w_opt, noise_mode='const')

      # downsampling the generated image
      small_gen_img = u_images.BicubicDownSample(
          factor= 512 // 256, 
          device = DEVICE)(big_gen_img)

      # get mse loss
      mse_loss = MSE_Loss(big_gen_img,big_image) 

      # get LPIPS Perceptual Loss
      perceptual_loss = perceptual_vgg16.forward(small_image, small_gen_img)

      # get the regulizer term
      regularizer = lambda_v *((map2PN(w_opt)) ** 2).mean()

      loss = mse_loss + perceptual_loss + regularizer
      loss.backward()

      optimizer.step()

      # store the losses metrics
      loss_list.append(loss.item())
      loss_mse.append(mse_loss.item())
      loss_perceptual.append(perceptual_loss.item())

      # every SAVE_STEP, I store the current latent
      if (i +1) % SAVE_STEP == 0:
          print('iter[%d]:\t loss: %.4f\t mse_loss: %.4f\tpercep_loss: %.4f\tregularizer: %.6f' % (i+1,  
                                                                                                   loss.item(), 
                                                                                                   mse_loss.item(), 
                                                                                                   perceptual_loss.item(),
                                                                                                   regularizer.item()))
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

# load images from directory
data = u_data.load_data(PATH_DIR)

# testing downsampling
test_name = 'cross_domain'
options_lambdas = [0.001] #, 0.005, 0.01]

for i in range(len(data)):
  for lambda_v in options_lambdas:
    loss_list = run_optimization(data, id = i, 
                                init = 'w_mean',
                                sub_fix=f"_{test_name}_lambda_{lambda_v}",
                                save_loss = False, 
                                lambda_v=lambda_v)
        