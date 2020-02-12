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

def get_dataset_mean(directory):
  """
  Returns mean and standard deviation for image dataset
  """
  im_list = [os.path.join(directory,item) for item in os.listdir(directory)]
  ims = []
  for item in im_list:
    if item.endswith(".jpg"):
      ims.append(item)

  running_mean = np.zeros(3)
  running_std = np.zeros(3)
  for item in ims:
    # load image file
    im = Image.open(item)
    stats = ImageStat.Stat(im)
    mean = np.array([stats.mean[0],stats.mean[1],stats.mean[2]])
    stddev = np.array([stats.stddev[0],stats.stddev[1],stats.stddev[2]])
    running_mean += mean
    running_std += stddev

  mean = running_mean / len(ims)
  stddev = running_std   / len(ims)

  return mean, stddev

class Im_Dataset(data.Dataset):
    """
    Defines a custom dataset for loading images from the ISIC 2018 lesion 
    classification challenge. Images are divided into a training and validation
    partition with equal class distribution in each, and numerous transforms 
    are applied if in training mode
    """
    
    def __init__(self,class_num = 0, mode = "train",class_balance = False):
        """ 
        mode = train or validation
        """
        self.mode = mode
        self.class_num = class_num
        self.class_balance = class_balance
        self.im_mean = np.array([194.69792021/255, 139.26262747/255, 145.48524136/255])
        self.im_stddev = np.array([22.85509458/255, 30.16841156/255, 33.90319049/255])

        self.label_names = []
        self.labels = {}
        self.im_dir = "/home/worklab/Desktop/ISIC2018_Task3_Training_Input"
        self.im_list = []
        self.all_train_indices = []
        self.all_val_indices = []
        self.train_indices = []
        self.val_indices = []

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
            arr = arr.nonzero()[0]

            # convert to torch
            label = torch.from_numpy(arr)#.float()
            self.labels[name] = label

        # split data by class
        self.class_indices = []
        for i in range(0,7):
          indices = []
          for j, item in enumerate(self.im_list):
            if self.labels[item] == i:
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

      # get number of positives
      pos = int(len(self.all_train_indices[self.class_num]))
      neg = len(self.im_list) - pos

      # get random positive indices ordering
      pos_indices = self.all_train_indices[self.class_num].copy()
      random.shuffle(pos_indices)

      # get random negative indices ordering
      neg_indices = []
      for cls in range(len(self.all_train_indices)):
        if cls != self.class_num:
          for idx in self.all_train_indices[cls]:
            neg_indices.append(idx)
      random.shuffle(neg_indices)
      
      if pos > neg:
        # get random subsample of positive indices
        pos_indices = pos_indices[:neg]
        

      elif pos < neg:
        # get random subsample of negative indices
        # note: perhaps should balance classes here
        neg_indices = neg_indices[:pos]

      assert len(neg_indices) == len(pos_indices), "Unequal pos and neg lengths: {} {}".format(len(neg_indices),len(pos_indices))
      self.train_indices = neg_indices + pos_indices



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
           + "Label: " + self.label_names[label]
        else:
          label = "Image: {} || ".format(self.im_list[self.val_indices[idx]]) \
           + "Label: " + self.label_names[label]
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
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
# define a super simple ResNet model to see how it does
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
         
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

def eval_accuracy(pred,actual):
  """
  Returns the accuracy of the predictions (hard instead of softmax loss)
  """
  pred = torch.round(pred)
  diff = torch.where((actual - pred) != 0)[0]
  accuracy = 1 - len(diff)/len(actual)
  return accuracy

def load_model(checkpoint_file,model,optimizer):
    """
    Reloads a checkpoint, loading the model and optimizer state_dicts and 
    setting the start epoch
    """
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    all_losses = checkpoint['losses']
    all_accs = checkpoint['accs']

    return model,optimizer,epoch,all_losses,all_accs


def get_weights(targets,device):
  """
  Weights the importance of each example by the relative frequency in the dataset
  """

  weights = np.array([1/1113,1/6705,1/514,1/327,1/1099,1/115,1/142])
  weights = torch.from_numpy(weights/np.sum(weights)).float().unsqueeze(0).to(device)
  weights = weights.repeat(targets.shape[0],1)

  weights = torch.mul(weights,targets)
  weights = torch.sum(weights, dim = 1)
  print (weights)
  return weights.to(device)

class weightedBCELoss(nn.Module):
    def __init__(self,batch_size = 16):
        super(weightedBCELoss,self).__init__()
        weights =   weights = np.array([1/1113,1/6705,1/514,1/327,1/1099,1/115,1/142])
        weights = torch.from_numpy(weights/np.sum(weights)).float().unsqueeze(0).to(device)
        self.weights = weights.repeat(batch_size,1).to(device)

    def forward(self,preds,target):
        """ Compute the prism corner coords and calculate 
            MSE loss compared to target corner coords"""

        # bce should be [Batch_size x num_classes]
        #bce = torch.mul(target,preds.log()) # + torch.mul(1-target,1-preds.log()) # supress negative labels to prevent negative overwhelming
        weights = torch.mul(target,self.weights)
        weights_norm = (weights/torch.sum(weights))*target.shape[0]
        #weighted = torch.mul(bce,weights_norm)

        bce = torch.mul(weights_norm,preds.log())  + 0.1667*torch.mul(1-target,(1-preds).log()) # supress negative labels to prevent negative overwhelming
        return -torch.sum(bce) /(preds.shape[0]*preds.shape[1])



def binary_confusion_vectors(counts,cls):
    """
    Plots binary confusion matrix for binary classifier
    counts - 2 x num_classes numpy array of raw counts (correct incorrect)
    cls - int, for title of plots
    """
    class_labels = [0,1,2,3,4,5,6]
    sums = np.sum(counts,axis= 0)
    percentages = np.round(counts/sums * 100)
   
    fig, ax = plt.subplots(2,1,figsize = (10,3.3))
    # plot correct items
    ax0_data = percentages[0,np.newaxis]
    im = ax[0].imshow(ax0_data,cmap = "YlGn", aspect = "auto")
    ax[0].set_yticks(np.arange(1))
    ax[0].set_yticklabels(["Correct"],fontsize = 20)
    # Loop over data dimensions and create text annotations.
    for j in range(len(class_labels)):
        text = ax[0].text(j, 0, counts[0, j],
                   ha="center", va="bottom", color="k",fontsize = 20)
        text = ax[0].text(j, 0, str(percentages[0, j])+"%",
                 ha="center", va="top", color="k",fontsize = 14)
   
    # plot incorrect items
    ax1_data = percentages[1,np.newaxis]
    im = ax[1].imshow(ax1_data,cmap = "YlOrRd", aspect = "auto")
    ax[1].set_yticks(np.arange(1))
    ax[1].set_yticklabels(["Incorrect"],fontsize = 20)
    plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
   
    # Loop over data dimensions and create text annotations.
    for j in range(len(class_labels)):
        text = ax[1].text(j, 0, counts[1, j],
                       ha="center", va="bottom", color="k",fontsize = 20)
        text = ax[1].text(j, 0, str(percentages[1, j])+"%",
                       ha="center", va="top", color="k",fontsize = 14)
           
    ax[0].set_title("Class {}".format(cls),fontsize = 20)
    #ax[1].set_xlabel("Class",fontsize = 20)
    fig.tight_layout(h_pad = -2)
    plt.show()
   

         