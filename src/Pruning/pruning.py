import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torchvision import models
import os

from models.pretrained_models import PretrainedModel

#get n directories before

def get_nth_parent(n):
    path = os.getcwd()
    par_dir = os.path.abspath(os.path.join(path, os.pardir))
    for i in range(n):
        path = par_dir
        par_dir = os.path.abspath(os.path.join(path, os.pardir))
    return par_dir



#read in model from .pth file

# Step 1: Define your model architecture
# For example, let's use a pre-trained ResNet model
model = models.densenet121(pretrained=False)

model_path = 'Baseline_2024-01-26_19-38-22/Baseline_2024-01-26_19-38-22_2_best.pth'


# Step 2: Load the pre-trained weights from the .pth file
file_path = f'{get_nth_parent(1)}/experiments/{model_path}'


checkpoint = torch.load(file_path, map_location=torch.device('cpu'))

##print(list(checkpoint.keys()))
#print(checkpoint['model'])
#print(checkpoint['model'])
# If the .pth file contains the state_dict() only, use the following line:
#model.load_state_dict(checkpoint)

# If the .pth file contains more than just the state_dict (e.g., optimizer state, epoch information), use the following line:
#model.load_state_dict(checkpoint['modelParams'])

pm = PretrainedModel(checkpoint['modelParams'])

print('loaded model successfully')
