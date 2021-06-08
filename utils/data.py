import numpy as np 
import cv2
import matplotlib.pyplot as plt
import glob
import os
import torch

def load_image(path):
    if(path[-3:]=='bmp' or path[-3:]=='jpg' or path[-3:]=='png'):
        return cv2.imread(path)[:,:,::-1]
    else:
        img = (255*plt.imread(path)[:,:,:3]).astype('uint8')

    return img

def load_data(directory, interval = None):

  if interval is None:
    files = sorted(glob.glob(os.path.join(directory + "*.png")))
  else:
    files = sorted(np.array(glob.glob(os.path.join(directory + "*.png")))[interval])

  data = {}
  for idx, file in enumerate(files):
    img_dir = {
        "img": load_image(file),
        "name": os.path.basename(os.path.normpath(file))
    }
    data[idx] = img_dir
  return data

#@title Function to display the data
def display_data(data, ncols=4):

    inches = 4
    num_img = len(data)
    nrows = np.ceil(num_img/ncols).astype(int)
    fig, axs = plt.subplots(nrows,ncols,figsize=(inches * ncols , inches * nrows))

    idx = 0
    for i in range(nrows):
      for j in range(ncols):
        if idx < num_img:
          axs[i,j].imshow(data[idx]['img'])
          axs[i,j].set_title("Image ID: {}".format(idx))
          axs[i,j].axis('off')
          idx += 1

#Load image from a folder to a pytorch tensor
def build_tensor_results(path_generated, path_references, device):
  """Load generated and reference images from folder to a tensor.
     The tensor has the shape: [n_images, 2, C, H, W]

     The second dimension corresponds to the reference and generated image
     in that order.

  Args:
      path_generated ([str]): path where the generated images are stored
      path_references ([str]): path where the reference images are stored
      device ([torch device]): device to perform the metric calculation

  Returns:
      [tensor]: tensor with generated and reference images shape: [n_images, 2, C, H, W]
  """
  input_data = load_data(path_references)

  generated_imgs = load_data(path_generated)

  full_batches = []

  for i in range(12):
    references = input_data[i]['img']
    synthetics = generated_imgs[i]['img']

    batch_data = np.stack([synthetics, references], axis = 0)

    full_batches.append(batch_data)

  full_batches = np.array(full_batches)
  print("full_batches numpy: ", full_batches.shape)

  # convert to pytorch tensor
  full_batches = torch.tensor(full_batches, device = device, dtype = torch.float32)
  full_batches = full_batches.permute(0, 1, 4, 2, 3)
  print("full_batches tensor: ", full_batches.size())

  return full_batches