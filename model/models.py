import torch.nn as nn
import torch
from torch.nn.modules.linear import Linear
from model.aux_models import PixelNorm, Scale_B, StyleBlock, EarlyStyleBlock, ResidualConvBlock
import logging

class MappingNetwork(nn.Module):
    '''
    A mapping consists of multiple fully connected layers.
    Used to map the input to an intermediate latent space W.
    '''
    def __init__(self, n_fc, n_z, p_space = False):

        super(MappingNetwork, self).__init__()
        layers = [PixelNorm()] # Normalize the latent_z
        for i in range(n_fc):
            layers.append(nn.Linear(n_z, n_z))
            
            # Transform to p-space
            if i == n_fc - 1 and p_space:
                layers.append(nn.LeakyReLU(5))
            else:
                layers.append(nn.LeakyReLU(0.2))
            
        self.mapping = nn.Sequential(*layers)
    
    def forward(self, latent_z):
        latent_w = self.mapping(latent_z)
        
        return latent_w    

# Generator Code
class Generator(nn.Module):
    def __init__(self, nz, nfc, nc_resolution, p_space = False):
        super(Generator, self).__init__()
        # Setting the constant input [1, channel, 4, 4 ]
        self.constant = nn.Parameter(torch.randn(1, 512, 4, 4))

        # Setting the mapping layer
        self.mapping_net = MappingNetwork(nfc, nz, p_space)

        # Adding the first noise inmediately after the constant
        self.first_noise = Scale_B(nc_resolution[0][1])

        # Define the style block for the early step with just one layer
        conv_layers = [EarlyStyleBlock(nc_resolution[0][0], nc_resolution[0][1], nz)]
        # # Define style blocks per resolution
        for nc_in, nc_out in nc_resolution[1:]:
            conv_layers.append(StyleBlock(nc_in, nc_out, nz))
        self.conv_layers = nn.ModuleList(conv_layers)

        # Layer to transform to rgb
        self.to_rgb = nn.Conv2d(nc_resolution[-1][1], 3, 3, padding=1)

    def forward(self, latent_z, vector_noises):

        latent_w = self.mapping_net(latent_z)

        #apply the first noise inmediately after the constant
        result = self.constant + self.first_noise(vector_noises[0])
        
        #apply the sequence of style, conv, noise
        for i, conv in enumerate(self.conv_layers):
            result = conv(result, latent_w,vector_noises[i])           

        result = self.to_rgb(result)

        return result

class Discriminator(nn.Module):
    def __init__(self, nc_resolution, final_resolution):
        super(Discriminator, self).__init__()
        self.from_rgb = nn.Conv2d(3, nc_resolution[0][0], 1)

        convs = []
        for nc_in, nc_out in nc_resolution:
            convs.append(ResidualConvBlock(nc_in, nc_out, 3, 1))
        self.convs  = nn.ModuleList(convs)

        self.fcs = nn.Sequential(
            Linear(nc_resolution[-1][1] * final_resolution, nc_resolution[-1][1]),
            Linear(nc_resolution[-1][1], 1)
        )
        
    def forward(self, input_tensor):
        
        result = self.from_rgb(input_tensor)

        for conv in self.convs:
            result = conv(result)

        result = torch.flatten(result)

        result = self.fcs(result)
        return result