from model.sub_models import MappingNetwork
import torch
import numpy as np

mapNet = MappingNetwork(4,4, 10)

tensor = torch.tensor(np.arange(0,20).reshape(2,10),dtype=torch.float)
print(tensor)
mapNet(tensor)

print(mapNet)