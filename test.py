import logging
from model.models import Discriminator
log_format = ' (line %(lineno)d %(filename)s) | [%(levelname)s]: %(message)s'
logging.basicConfig(format= log_format, level=logging.INFO)

from model.models import Generator, Discriminator
import torch

nc_per_resolution = {
    4    :   (512, 512),     # 4x4 first resolution just one layer
    8    :   (512, 512),     # 8x8
    16   :   (512, 512),     # 16x16
    32   :   (512, 512),     # 32x32
    64   :   (512, 256),     # 64x64
    128  :   (256, 128),     # 128x128
    256  :   (128, 64),      # 256x256
    512  :   (64, 32),       # 512x512
    1024 :   (32, 16)        # 1024x1024
}

nc_resolutions = list(nc_per_resolution.values())
resolutions = list(nc_per_resolution.keys())
nz = 512

latent_z = torch.randn( (1, nz))
# Adding one noise per resolution
vector_noises = [torch.randn((1, 1, size, size)) for size in resolutions]

# Testing the generator
# generator = Generator(nz = 512, nfc = 8, nc_resolution = nc_resolutions)
# print(generator)
# result = generator(latent_z, vector_noises)
# print(result.size())

# Testing the discriminator
real_sample = torch.randn(1,3, 1024, 1024, dtype= torch.float)
print(real_sample.size())

nc_resolutions_D = [(j,i) for i, j in nc_resolutions[::-1]]
discriminator = Discriminator(nc_resolutions_D, resolutions[0])

print(discriminator)
label = discriminator(real_sample)

print(label.size())
