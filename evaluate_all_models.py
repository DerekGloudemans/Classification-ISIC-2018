# imports

# this seems to be a popular thing to do so I've done it here
#from __future__ import print_function, division


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

from sklearn.tree import DecisionTreeClassifier
from utils import get_metrics

class Im_Dataset(data.Dataset):
    """
    Defines a custom dataset for loading images from the ISIC 2018 lesion 
    classification challenge. Images are divided into a training and validation
    partition with equal class distribution in each, and numerous transforms 
    are applied if in training mode
    """
    
    def __init__(self, mode = "train",class_balance = False):
        """ 
        mode = train or validation
        """
        self.mode = mode
        self.class_balance = class_balance
        self.im_mean = np.array([194.69792021/255, 139.26262747/255, 145.48524136/255])
        self.im_stddev = np.array([22.85509458/255, 30.16841156/255, 33.90319049/255])

        self.label_names = []
        self.labels = {}
        self.im_dir = "/home/worklab/Desktop/ISIC2018_Task3_Training_Input"
        self.im_list = []
        self.train_indices = []
        self.val_indices = []
        self.all_train_indices = []
        self.all_val_indices = []

        ## load files in image directory
        im_list = [item for item in os.listdir(self.im_dir)]
        #get images only
        for item in im_list:
          if item.endswith(".jpg"):
            self.im_list.append(item.split(".")[0])

        self.im_list.sort()
        
        ## load labels
        self.label_dir = "/home/worklab/Desktop/ISIC2018_Task3_Training_GroundTruth"
        
        # read label csv file
        f = open(os.path.join(self.label_dir,"ISIC2018_Task3_Training_GroundTruth.csv"),'r')
        label_text = f.readlines()
        
        # parse each line
        self.label_names = label_text[0].split(',')[1:]
        for item in label_text[1:]:
            splits = item.split(",")
            name = splits[0]
            splits = splits[1:]
            data = []
            for val in splits:
                data.append(np.round(float(val.rstrip())))
            arr = np.array(data)

            # flatten 7-d binary label into 1-d integer label
            #arr = arr.nonzero()[0]

            # convert to torch
            label = torch.from_numpy(arr).float()
            self.labels[name] = label

        # split data by class
        self.class_indices = []
        for i in range(0,7):
          indices = []
          for j, item in enumerate(self.im_list):
            if self.labels[item][i] == 1:
              indices.append(j)
          self.class_indices.append(indices)

        for indices in self.class_indices:
            self.all_train_indices.append(indices[:int(len(indices)*0.85)])
            self.all_val_indices.append(indices[int(len(indices)*0.85):])
        
        # flatten val_indices
        for cls in self.all_val_indices:
            for idx in cls:
                self.val_indices.append(idx)

        if class_balance: # balances positive and negative examples in training data
            self.shuffle_balance()
        else:
          for cls in self.all_train_indices:
            for idx in cls:
                self.train_indices.append(idx)



        # define transforms
        self.transforms_train = transforms.Compose([
                transforms.ColorJitter(brightness = 0.2,contrast = 0.2,saturation = 0.1),
                transforms.RandomAffine(15,scale = (1.1,1.2),shear = 10,resample = Image.BILINEAR,fillcolor = (194,139,145)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.im_mean,self.im_stddev,inplace = True)                             
        ])

        self.transforms_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.im_mean,self.im_stddev,inplace = True)                                                
        ])

    def shuffle_balance(self):
      """
      Returns a list of positive indices and negative indices for training mode
      such that the number of positives and negatives is the same
      """

      # get min class size
      lengths = []
      for cls in self.all_train_indices:
          lengths.append(len(cls))
      min_length = min(lengths)  

      # get random subset of each class with min_length of each class
      self.train_indices = []
      for cls in self.all_train_indices:
          random.shuffle(cls)
          for idx in cls[:min_length]:
              self.train_indices.append(idx)





    def __getitem__(self,idx):
        """
        Note: index gives the index of either self.train_indices or 
        self.val_indices. The value at that index is itself an index to 
        self.im_list, which contains a string name of a file. This is done to
        keep training and validation sets separate but from the same underlying 
        data for correct class distribution
        """

        # get image name
        if self.mode == "train":
          im_idx = self.train_indices[idx]
        else:
          im_idx = self.val_indices[idx]
        
        im_name = self.im_list[im_idx]

        y = self.labels[im_name]

        # load image file
        im = Image.open(os.path.join(self.im_dir,im_name +'.jpg'))
      
        
        # apply transforms to image
        if self.mode == "train":
          x = self.transforms_train(im)
        else:
          x = self.transforms_val(im)

        return x, y

    def __len__(self):
      if self.mode == "train":
        return len(self.train_indices)
      else:
        return len(self.val_indices)


    def show(self,idx):
        im,label = self[idx]
        if self.mode == "train":
          label = "Image: {} || ".format(self.im_list[self.train_indices[idx]]) \
           + "Label: " + self.label_names[np.where(label == 1)[0][0]]
        else:
          label = "Image: {} || ".format(self.im_list[self.val_indices[idx]]) \
           + "Label: " + self.label_names[np.where(label == 1)[0][0]]
        # shift axes and convert RGB to GBR for plotting
        im = im.data.numpy().transpose(1, 2, 0)
         

        # unnormalize
        im = self.im_stddev * im + self.im_mean
        im = np.clip(im, 0, 1)
        im = im[:, :, ::-1]
        im = im.copy()
        cv2.putText(im,label,(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        #plot label
        im = im.copy()*255
        cv2.imshow("frame",im)
        #cv2.imwrite("im{}-{}.png".format(idx,np.random.randint(0,1000)),im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
# define a super simple ResNet model to see how it does
class Multi_Net(nn.Module):
    def __init__(self):
        super(Multi_Net, self).__init__()
         
        #self.features = models.resnet18(pretrained = True)
        self.features = models.resnet50(pretrained = True)
        self.features.fc = nn.Linear(2048,128)
        
        self.batchnorm = nn.BatchNorm1d(128)
        self.drop = nn.Dropout()
        self.fc1 = nn.Linear(128,32)
        self.fc2 = nn.Linear(32,7)
        

    def forward(self, x):
        x = self.features(x)
        x = F.relu(self.batchnorm(x))

        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = torch.sigmoid(self.fc2(x))
        #x = torch.softmax(x,dim = 0)
                
        return x


# define a super simple ResNet model to see how it does
class Single_Net(nn.Module):
    def __init__(self):
        super(Single_Net, self).__init__()
         
        #self.features = models.resnet18(pretrained = True)
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

class Test_Dataset():
    def __init__(self, mode = "train",class_balance = False):
        """ 
        mode = train or validation
        """
    
        self.im_mean = np.array([194.69792021/255, 139.26262747/255, 145.48524136/255])
        self.im_stddev = np.array([22.85509458/255, 30.16841156/255, 33.90319049/255])

        self.labels = {}
        self.im_dir = "/home/worklab/Desktop/test"
        self.im_list = []
        self.transforms_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.im_mean,self.im_stddev,inplace = True)                                                
            ])

        ## load files in image directory
        im_list = [item for item in os.listdir(self.im_dir)]
        #get images only
        for item in im_list:
          if item.endswith(".jpg"):
            self.im_list.append(item.split(".")[0])

        self.im_list.sort()
        
        ## load labels
        self.label_dir = "/home/worklab/Desktop/labels"
        
        # read label csv file
        f = open(os.path.join(self.label_dir,"Test_labels.csv"),'r')
        label_text = f.readlines()
        
        # parse each line
        self.label_names = label_text[0].split(',')[1:]
        for item in label_text[1:]:
            splits = item.split(",")
            name = splits[0]
            splits = splits[1:]
            data = []
            for val in splits:
                data.append(np.round(float(val.rstrip())))
            arr = np.array(data)
            
            # convert to torch
            label = torch.from_numpy(arr).float()
            self.labels[name] = label

    def __len__(self):
        return len(self.im_list)
     
    def __getitem__(self,idx):
        """
        Note: index gives the index of either self.train_indices or 
        self.val_indices. The value at that index is itself an index to 
        self.im_list, which contains a string name of a file. This is done to
        keep training and validation sets separate but from the same underlying 
        data for correct class distribution
        """
        
        im_name = self.im_list[idx]
        y = self.labels[im_name]

        # load image file
        im = Image.open(os.path.join(self.im_dir,im_name +'.jpg'))      
        
        # apply transforms to image
        x = self.transforms_val(im)

        return x, y   
    
def generate_features(model_list,dataset,device):
  """
  Takes a list of dicts, each specifying a model
  Predicts the output of each model for each item in the dataset, and returns this vector
  """

  length = len(dataset)
  width = sum([item['outputs'] for item in model_list])

  features = np.zeros([length,width])
  labels = np.zeros([length,7])
  
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
    for i in range(len(dataset)):
      x,label = dataset[i]
      label = label.data.numpy()
      labels[i,:] = label
      x = x.to(device).unsqueeze(0)
      output = model(x).data.cpu().numpy()
      features[i,current_column:current_column+item['outputs']] = output
    
    current_column += item['outputs']
    print("Finished generating outputs for model {}.".format(item['name']))
    del model

  return features, labels

def plot_all_losses(model_list):
    """
    Takes in a bunch of checkpoints and names and plots all losses together
    """
    plt.style.use('ggplot')
    print ("plotting losses")
    colors = list(mcolors.TABLEAU_COLORS)
    all_labels = []
    plt.figure()
    for i,model in enumerate(model_list):
        checkpoint = model['checkpoint']
        checkpoint = torch.load(checkpoint)
        train_loss = checkpoint['losses']['train']
        val_loss = checkpoint['losses']['val']
        label = model['name']
        plt.plot(train_loss,color = colors[i])
        plt.plot(val_loss, '--', color = colors[i])
        all_labels.append(label+" train")
        all_labels.append(label+" val")
        
    plt.xlabel("Epoch",fontsize = 20)
    plt.ylabel("Total Loss per Epoch",fontsize = 20)
    plt.title("Total Loss per Epoch",fontsize = 20)
    plt.legend(all_labels,fontsize = 16)
    plt.show()

        
if __name__ == "__main__":
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
        
        #dataset = Im_Dataset(mode = "train")
        #train_features, train_labels = generate_features(model_list,dataset,device)
        dataset = Im_Dataset(mode = "val")
        try:
            val_features
        except:
            val_features, val_labels = generate_features(model_list,dataset,device)
        
        condensed_labels = np.argmax(val_labels,axis = 1)
        tree = DecisionTreeClassifier(min_samples_leaf = 5)
        tree.fit(val_features,condensed_labels)
        
        # evaluate Test_Dataset()
        dataset = Test_Dataset()
        try:
            test_features
        except:
            test_features, test_labels = generate_features(model_list,dataset,device)
            
        outputs  = tree.predict(test_features)
        condensed_labels = np.argmax(test_labels,axis = 1)

        # row is pred, col is true
        confusion_matrix = np.zeros([7,7])
        for i in range(len(test_labels)):
            label = condensed_labels[i]
            output = outputs[i]
            confusion_matrix[output,label]+=1
            
        
        #a,r,p = get_metrics(confusion_matrix)
        print("Model test accuracy: {}%".format(a*100))
        print("Model test recall (per class): {}".format(recall))
        print("Model test precision (per class): {}".format(precision))
    
    
    
        model_list = [
            {"name":"6vA", "outputs":1,                 "checkpoint":"final_checkpoints/all_6vA.pt"},
            {"name":"Weighted Multiclass", "outputs":7, "checkpoint":"final_checkpoints/final_weighted_multiclass.pt" },
            {"name":"Balanced Multiclass", "outputs":7, "checkpoint":"final_checkpoints//final_balanced_multiclass.pt" },
            {"name":"4v3", "outputs":1,                 "checkpoint":"final_checkpoints/final_5v0.pt" },
            {"name":"5v0", "outputs":1,                 "checkpoint":"final_checkpoints/final_4v3.pt"},
            {"name":"4v2", "outputs":1,                 "checkpoint":"final_checkpoints/final_4v2.pt"},
            {"name":"2vA", "outputs":1,                 "checkpoint":"final_checkpoints/all_2vA.pt"},
            {"name":"3vA", "outputs":1,                 "checkpoint":"final_checkpoints/all_3vA.pt"},
            {"name":"4vA", "outputs":1,                 "checkpoint":"final_checkpoints/all_4vA.pt"},
            {"name":"5vA", "outputs":1,                 "checkpoint":"final_checkpoints/final_5vA.pt"}
              ]
        plot_all_losses(model_list)
