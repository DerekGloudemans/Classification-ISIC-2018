"""
Run from command line. Takes an iamge file path as input and outputs its class.
"""


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
import matplotlib.colors as mcolors
import math
import argparse

from sklearn.tree import DecisionTreeClassifier
from _utils import get_metrics


# define a super simple ResNet model 
class Multi_Net(nn.Module):
    def __init__(self):
        super(Multi_Net, self).__init__()
         
        self.features = models.resnet50(pretrained = True)
        self.features.fc = nn.Linear(2048,128)
        
        self.batchnorm = nn.BatchNorm1d(128)
        self.drop = nn.Dropout()
        self.fc1 = nn.Linear(128,32)
        self.fc2 = nn.Linear(32,7)
        

    def forward(self, x):
        x = self.features(x)
        x = self.batchnorm(x)

        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = torch.sigmoid(self.fc2(x))
        #x = torch.softmax(x,dim = 0)
                
        return x


# define a super simple ResNet model
class Single_Net(nn.Module):
    def __init__(self):
        super(Single_Net, self).__init__()
         
        self.features = models.resnet50(pretrained = True)
        self.features.fc = nn.Linear(2048,128)
        
        self.batchnorm = nn.BatchNorm1d(128)
        self.drop = nn.Dropout()
        self.fc1 = nn.Linear(128,32)
        self.fc2 = nn.Linear(32,1)
        

    def forward(self, x):
        x = self.features(x)
        x = self.batchnorm(x)

        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = torch.sigmoid(self.fc2(x))
        #x = torch.softmax(x,dim = 0)
                
        return x
    
    
def generate_im_features(model_list,image_path,device):
  """
  Takes a list of dicts, each specifying a model
  Predicts the output of each model for image
  image - file path string
  model_list - list of dicts
  device - torch.device specifying cuda or cpu
  """
  im_mean = np.array([194.69792021/255, 139.26262747/255, 145.48524136/255])
  im_stddev = np.array([22.85509458/255, 30.16841156/255, 33.90319049/255])
  
  width = sum([item['outputs'] for item in model_list])
  features = np.zeros([1,width])
  
  # preprocess image
  im = Image.open(os.path.join(image_path))
  tf = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize(im_mean,im_stddev,inplace = True)                                                
     ])      

  x = tf(im)
  x = x.to(device).unsqueeze(0)

  # evaluate with all models
  current_column = 0
  for j, item in enumerate(model_list):
    if item["outputs"] == 1:
          model = Single_Net()
          checkpoint = torch.load(item['checkpoint'])
          try:
              model.to(device)
              model.load_state_dict(checkpoint['model_state_dict'])
          except:
              model = nn.DataParallel(model)
              model = model.to(device)
              model.load_state_dict(checkpoint['model_state_dict'])

    else:
      model = Multi_Net()
      model.to(device)
      checkpoint = torch.load(item['checkpoint'])
      model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    output = model(x).data.cpu().numpy()
    features[0,current_column:current_column+item['outputs']] = output
    
    current_column += item['outputs']
    print("Finished generating outputs with model {}.".format(item['name']))
    del model

  return features



if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Get input image file path.')
        parser.add_argument("inp",help='<Required> string, file path of image',type = str)
        args = parser.parse_args()
        
        # parse args
        inp =  args.inp
        
        start_time = time.time()
    
        try:
            torch.multiprocessing.set_start_method('spawn')    
        except:
            pass
        
        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        torch.cuda.empty_cache()   
        
        
        model_list = [
            {"name":"6vA", "outputs":1,                 "checkpoint":"final_checkpoints/final_6vA.pt"},
            {"name":"Weighted Multiclass", "outputs":7, "checkpoint":"final_checkpoints/final_weighted_multiclass.pt" },
            {"name":"Balanced Multiclass", "outputs":7, "checkpoint":"final_checkpoints//final_balanced_multiclass.pt" },
            {"name":"4v3", "outputs":1,                 "checkpoint":"final_checkpoints/final_5v0.pt" },
            {"name":"5v0", "outputs":1,                 "checkpoint":"final_checkpoints/final_4v3.pt"},
            {"name":"4v2", "outputs":1,                 "checkpoint":"final_checkpoints/final_4v2.pt"},
            {"name":"2vA", "outputs":1,                 "checkpoint":"final_checkpoints/final_2vA.pt"},
            {"name":"3vA", "outputs":1,                 "checkpoint":"final_checkpoints/final_3vA.pt"},
            {"name":"4vA", "outputs":1,                 "checkpoint":"final_checkpoints/final_4vA.pt"},
            {"name":"5vA", "outputs":1,                 "checkpoint":"final_checkpoints/final_5vA.pt"}
              ]
        
        test_features = generate_im_features(model_list,inp,device)
        print("Generated feature vector from models:")
        print(test_features)
        with open("final_checkpoints/trained_decision_tree.cpkl","rb") as f:
            tree = pickle.load(f)
        outputs  = tree.predict(test_features)
        print(outputs)
        class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

        cls = int(outputs[0])
        print("Predicted class: {} ({})".format(cls,class_names[cls]))
        print("Inference took {} seconds.".format(time.time()-start_time))