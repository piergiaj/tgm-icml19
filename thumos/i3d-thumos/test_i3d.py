import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-load_model', type=str)
parser.add_argument('-gpu', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms


import numpy as np

from pytorch_i3d import InceptionI3d



def run(mode='rgb', batch_size=4, load_model=''):
    device = torch.device('cuda')
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2, spatial_size=112)
    else:
        i3d = InceptionI3d(400, in_channels=3, spatial_size=112)
    sd = torch.load(load_model)
    i3d.load_state_dict(sd)
    
    i3d.to(device)

    data = torch.rand((4,3,32,112,112)).to(device)
    print(i3d(data))


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, load_model=args.load_model)
