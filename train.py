from __future__ import print_function
#%matplotlib inline
import logging
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.animation as animation
import torch.optim as optim

# Self Functions
from utils.data import load_data
from model.utils import format_batch
from model.model import GAN
from utils.visualization import plot_samples
# from IPython.display import HTML

logging.getLogger().setLevel(logging.DEBUG)


# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)

random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot = "stuff/data/"

# Parameter to load data 
workers = 2         # Number of workers for dataloader
image_size = 64     # Spatial size of training images. 
                    # All images will be resized to this
                    #   size using a transformer.
nc = 3              # Number of channels in the training images. 
                    # For color images this is 3

# Architectural Parameters

nz = 100        # Size of z latent vector (i.e. size of generator input)
ngf = 64        # Size of feature maps in generator
ndf = 64        # Size of feature maps in discriminator


# Training Parameters
batch_size = 128            # Batch size during training
num_epochs = 5              # Number of training epochs
lr = 0.0002                 # Learning rate for optimizers
beta1 = 0.5                 # Beta1 hyperparam for Adam optimizers


# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Load dataset
dataloader = load_data(dataroot, image_size, batch_size, workers)

# Visualize the data samples
# plot_samples(dataloader, device)

# Create a gan
gan = GAN(ngpu, nz, ngf, ndf, nc, device)

# Initialize BCELoss function
loss_function = nn.BCELoss()

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(gan.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimezerG = optim.Adam(gan.generator.parameters(), lr=lr, betas=(beta1, 0.999))

gan.define_loss_and_opt(loss_function, optimizerD, optimezerG)

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator # size [64, nz, 1, 1]
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish some training parameters
# convention for real and fake labels during training
real_label = 1.
fake_label = 0.

gan.define_training_params(batch_size, real_label, fake_label)

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################       
        # Format batch
        b_data, labels = format_batch(data, real_label, device)

        D_x, D_G_z1, errD, b_fake_data = gan.discriminator_step(b_data, labels)
        D_G_z2, errG = gan.generator_step(b_fake_data, labels)

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = gan.generator(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1