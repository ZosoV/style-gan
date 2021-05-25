import numpy as np 
import cv2
import matplotlib.pyplot as plt
import glob
import os

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