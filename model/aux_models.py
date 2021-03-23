import logging
from torch.nn.modules import upsampling
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.linear import Linear
from model.utils import hook_he_uniform
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

# Normalization on every element of input latent vector
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

# "learned affine transform" A
class FC_A(nn.Module):
    '''
    Learned affine transform A, this module is used to transform
    intermediate vector w into a style vector
    '''
    def __init__(self, dim_latent, n_channel):
        super().__init__()
        self.transform = nn.Linear(dim_latent, n_channel)

        # the biases associated with the style vector intialize to 1
        self.transform.bias.data[:] = 1
        # self.transform.bias.data[n_channel:] = 0

    def forward(self, w):
        # Gain scale factor and bias with:
        # add two more dimension to right of the tensor
        # [n, n_channel, 1, 1]
        style = self.transform(w).unsqueeze(2).unsqueeze(3)
        return style

# "learned per-channel scaling factors" B
# 5/13: Debug - tensor -> nn.Parameter
class Scale_B(nn.Module):
    '''
    Learned per-channel scale factor, used to scale the noise
    '''
    def __init__(self, n_channel):
        super().__init__()
        logging.info("n_channel Scale_B: {}".format(n_channel))
        self.weight = nn.Parameter(torch.zeros((1, n_channel, 1, 1)))
    
    def forward(self, noise):
        #noise shape: [1,fmp_height,fmp_width] # a face of the current fmap
        #self.weight: [1,n_channel, 1, 1] # has a weight per each channel
        result = noise * self.weight
        return result 

class ModDemodConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def modulate(self, weights, style):
        # style = [input_channel, 1, 1]
        # style = style.reshape((-1,1,1))
        style = style.expand_as(weights)

        return nn.Parameter(weights * style)

    def demodulate(self, weights, epsilon = 0.001):
        sdv = torch.sqrt(torch.sum(torch.square(weights), dim = (2,3)) + epsilon)
        sdv = sdv.unsqueeze(-1).unsqueeze(-1)
        sdv = sdv.expand_as(weights)

        return nn.Parameter(weights * sdv)

    def forward(self, input_tensor, style, upsampling = False):
        # Style
        if upsampling:
            result = F.interpolate(input_tensor,scale_factor=2, mode='bilinear', align_corners=False)
            
        # Before to forward the input_tensor to convolution layer
        # we need to modulate and demodulate the weights
        self.convolution.weight = self.modulate(self.convolution.weight.data,style)
        self.convolution.weight = self.demodulate(self.convolution.weight.data)

        if upsampling:
            result = self.convolution(result)
        else:
            result = self.convolution(input_tensor)
        
        return result

class EarlyStyleBlock(nn.Module):
    """Used to feed the first convolutional layer with an style and an after noise

    Args:
        nn ([type]): [description]
    """
    def __init__(self, in_channel, out_channel, nz):
        super().__init__()

        self.conv = ModDemodConvLayer(in_channel, out_channel)
        self.scaled_noise = Scale_B(out_channel)
        self.style = FC_A(nz, in_channel)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, input_tensor, laten_w, noise_vector):
        # Adding style vector to the convolutional layer
        result = self.conv(input_tensor,self.style(laten_w))
        # Adding noise to the result
        result = result + self.scaled_noise(noise_vector)
        # Passing leaky relu
        result = self.lrelu(result)

        return result

class StyleBlock(nn.Module):
    """Used to feed the two convolutional layer 
        per resolution with their respective
        styles and noises
    Args:
        nn ([type]): [description]
    """
    def __init__(self, in_channel, out_channel, nz):
        super().__init__()

        self.conv1 = ModDemodConvLayer(in_channel, out_channel)
        self.scaled_noise1 = Scale_B(out_channel)
        self.style1 = FC_A(nz, in_channel)

        self.conv2 = ModDemodConvLayer(out_channel, out_channel)
        self.scaled_noise2 = Scale_B(out_channel)
        # Note the input channels now is denoted by out channel
        self.style2 = FC_A(nz, out_channel)

        # Leaky rely for both layers
        self.lrelu    = nn.LeakyReLU(0.2) 

    def forward(self, input_tensor, latent_w, noise_vector):
        # Adding style vector 1 to the convolutional layer 1
        result = self.conv1(input_tensor,self.style1(latent_w), upsampling = True)
        # Adding noise 1 to the result
        result = result + self.scaled_noise1(noise_vector)
        # Passing leaky relu
        result = self.lrelu(result)
        # Adding style vector 2 to the convolutional layer 2
        result = self.conv2(result,self.style2(latent_w))
        # Adding noise 2 to the result
        result = result + self.scaled_noise2(noise_vector)
        # Passing leaky relu
        result = self.lrelu(result)

        return result

class ResidualConvBlock(nn.Module):
    '''
    Used to construct progressive discriminator
    '''
    def __init__(self, in_channel, out_channel, size_kernel1, padding1, 
                 size_kernel2 = None, padding2 = None):
        super().__init__()
        
        if size_kernel2 == None:
            size_kernel2 = size_kernel1
        if padding2 == None:
            padding2 = padding1
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, size_kernel1, padding=padding1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channel, out_channel, size_kernel2, stride = 2, padding=padding2),
            nn.LeakyReLU(0.2)
        )

        self.adjust_fmp = nn.Conv2d(in_channel, out_channel, 1, padding=0)
    
    def forward(self, input_tensor):
        # Conv
        result = self.conv(input_tensor)

        downsampling = F.interpolate(input_tensor, scale_factor=0.5, mode="bilinear", align_corners=False)
        downsampling = self.adjust_fmp(downsampling)

        result = result + downsampling

        return result