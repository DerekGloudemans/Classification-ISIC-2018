# torch and specific torch packages for convenience
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import data
from torch import multiprocessing

# for convenient data loading, image representation and dataset management
from torchvision import models, transforms
from PIL import Image, ImageFile, ImageStat
ImageFile.LOAD_TRUNCATED_IMAGES = True
from scipy.ndimage import affine_transform
import cv2

# always good to have
import time
import os
import numpy as np    
import _pickle as pickle
import random
import copy
import matplotlib.pyplot as plt
import math
import argparse



def load_model(checkpoint_file):
    """
    Reloads a checkpoint, loading the model and optimizer state_dicts and 
    setting the start epoch
    """
    checkpoint = torch.load(checkpoint_file)
    all_losses = checkpoint['losses']
    all_accs = checkpoint['accs']

    return all_losses,all_accs

parser = argparse.ArgumentParser(description='Get input file.')
parser.add_argument("input",help='<Required> string - input checkpoint file path',type = str)
args = parser.parse_args()

checkpoint = args.input

all_losses,all_accs = load_model(checkpoint)

plt.figure()
plt.plot(all_accs['train'])
plt.plot(all_accs['val'])
plt.legend(["Train acc","Val acc"])
plt.show()