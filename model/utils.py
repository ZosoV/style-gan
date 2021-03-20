import torch.nn as nn
import torch 
import math



# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def format_batch(data, real_label, device):
    #data zero to get just the images, not the labels
    b_data = data[0].to(device)
    b_size = b_data.size(0)
    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

    return b_data, label

# Normalization on every element of input latent vector
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

# Define a hook to perform he initialization every step
def hook_he_uniform(module, input):

    def init_weights(m):
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')

    module.apply(init_weights)

# Uniformly set the hyperparameters of Linears
# "We initialize all weights of the convolutional, fully-connected, 
# and affine transform layers using N(0, 1)"
# 5/13: Apply scaled weights
class SLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.linear = nn.Linear(dim_in, dim_out)
        self.linear.weight.data.normal_()
        self.linear.bias.data.zero_()

        self.linear.register_forward_pre_hook(hook_he_uniform)
        
    def forward(self, x):
        return self.linear(x)