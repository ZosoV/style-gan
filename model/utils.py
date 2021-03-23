import torch.nn as nn
import torch 
import math
import logging

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

# Define a hook to perform he initialization every step
def hook_he_uniform(module, input):
    
    logging.info("Executing hook_he_uniform")

    def scale(m):
        classname = m.__class__.__name__
        
        if classname.find('Linear') != -1:
            fan_in = m.weight.data.size(1) * m.weight.data[0][0].numel()
            
            m.weights = m.weight * math.sqrt(2 / fan_in)

    def init_weights(m):
        classname = m.__class__.__name__
        
        if classname.find('Linear') != -1:
            logging.info("Class Name: {}".format(classname))
            nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')

    module.apply(scale)

