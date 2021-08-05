import os 
import pickle
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

PRETRAINED_MODEL_DIR = "stuff/pretrained_models/"

# function to load pretrained model
def load_pretrained_model(file_name = "ffhq.pkl", space = 'w+', device = 'cuda:0'):
  pretrained_mode = os.path.join(PRETRAINED_MODEL_DIR,file_name)

  with open(pretrained_mode, 'rb') as f:
      G = pickle.load(f)['G_ema'].to(device)  # torch.nn.Module

      if space == 'w':
        G.mapping.num_ws = 1 # setting to the space w not w+ that was defined by deafault

  return G

#@title Calculate mean latent to initialize
# 
def get_mean_latent( G, device, n_samples = 5e5):
  
  z = torch.randn((int(n_samples), 512), device=device)
  batch_size = int(1e5)

  w_mean = torch.zeros((1,18,512),requires_grad=True,device=device)
  for i in range(int(n_samples/batch_size)):
    w = G.mapping(z[i*batch_size:(i+1)*batch_size,:], None)
    w = torch.sum(w, dim = 0).unsqueeze(0)
    w_mean = w_mean + w

  w_mean = w_mean / n_samples

  return w_mean.clone().detach().requires_grad_(True)

def get_initial_latent(init, G, device):
  if init == "w_mean":
    w_opt = get_mean_latent(G, device)
  elif init == "w_zeros":
    w_opt = torch.zeros((1,18,512),requires_grad=True,device=device)
  elif init == "w_random":
    w_opt = torch.cuda.FloatTensor(1,18,512).uniform_(-1,1).requires_grad_()

  return w_opt

def convert2grayscale(input_img):
  transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3)                        
  ])

  return transform(input_img)

def convert2grayscale_tensor(input_img):
  # result = 0.2989 * image[:,0,:,:] + 0.5870 * image[:,1,:,:] + 0.1140 * image[:,2,:,:]
  result = (input_img[:,0,:,:] + input_img[:,1,:,:] + input_img[:,2,:,:]) / 3
  return torch.stack([result,result,result], dim=1)

# def mask_function(input_img):
#   n, c, h, w = input_img.size()
#   result = transforms.functional.erase(input_img, 
#                                           int(w/2), h, 
#                                           h, int(w/2), 
#                                           -1)
#   return result

def mask_function(input_img):
  n, c, h, w = input_img.size()  
  result = input_img.clone()
  result[:,:,:,-int(w/2):] = -1.

  return result

def mask_function_PIL(input_img):
  img_ref = input_img.copy()
  w, h = img_ref.size
  rect_size = (int(w/2), h)
  rect_pos = (int(w/2), 0)

  rect = Image.new("RGB", rect_size, (0, 0, 0))
  img_ref.paste(rect, rect_pos)
  return img_ref