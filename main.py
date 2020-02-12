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

from utils import Net, Im_Dataset, eval_accuracy, load_model, get_weights, weightedBCELoss

def train_model(model, optimizer, scheduler,loss_function,
                    datasets,positive_class,device, num_epochs=5, start_epoch = 0,
                    all_losses = None,all_accs = None):
        """
        Alternates between a training step and a validation step at each epoch. 
        Validation results are reported but don't impact model weights
        """
        # for storing all metrics
        if all_losses == None:
          all_losses = {
                  'train':[],
                  'val':[]
          }
          all_accs = {
                  "train":[],
                  "val":[]
                  }
        avg_acc = 0

        # create testloader for val_dataset (trainloader is made each epoch)
        params = {'batch_size': 12,
              'shuffle': True,
              'num_workers': 0,
              'drop_last' : True
              }
        testloader = data.DataLoader(datasets["val"],**params)

        for epoch in range(start_epoch,num_epochs):
            # reshuffle dataset to get new set of positives or negatives
            if datasets["train"].class_balance:
                datasets["train"].shuffle_balance()
            trainloader = data.DataLoader(datasets["train"],**params)
            dataloaders = {"train":trainloader,"val":testloader}
            
            print("")

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                    if epoch > 0:
                      scheduler.step(avg_acc)
                else:
                    model.eval()   # Set model to evaluate mode
                #print("Epoch {}: learning rate {}".format(epoch,optimizer.param_groups[lr]))

                # Iterate over data.
                count = 0
                total_loss = 0
                total_acc = 0
                for inputs, targets in dataloaders[phase]:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        
                        # shift targets to 1 0 positive negative labels
                        targets = 1- torch.ceil(torch.abs((targets.float()-positive_class)/7))

                        loss = loss_function(outputs,targets)
                        acc = eval_accuracy(outputs,targets)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
        
                    # verbose update
                    count += 1
                    total_acc += acc
                    total_loss += loss.item()
                    if count % 10 == 0:
                      print("{} epoch {} batch {} -- Loss: {:03f} -- Accuracy {:02f}".format(phase,epoch,count,loss.item(),acc))
                
                avg_acc = total_acc/count
                avg_loss = total_loss/count
                if epoch % 1 == 0:
                  print("Epoch {} avg {} loss: {:05f}  acc: {}".format(epoch, phase,avg_loss,avg_acc))
                  all_losses[phase].append(total_loss)
                  all_accs[phase].append(avg_acc)

                  # plot metrics
#                  plt.figure()
#                  plt.plot(all_losses[phase])
#                  plt.plot(all_accs[phase])

                  # save a checkpoint
                  PATH = "class{}_epoch{}.pt".format(positive_class,epoch)
                  torch.save({
                      'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      "losses":all_losses,
                      "accs":all_accs
                      }, PATH)
                  
                torch.cuda.empty_cache()

                
        return model , all_losses,all_accs


###############################################################################
###############################################################################
###############################################################################

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')    
    except:
        pass
    
    for positive_class in range(0,6):
    #positive_class = 0

        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        if torch.cuda.device_count() > 1 and positive_class > 0:
            print("Using multiple GPUs")
            MULTI = True
        else:
            MULTI = False
        torch.cuda.empty_cache()   
        
    
        #%% Create Model
        try:
            model
        except:
            model = Net()
            if MULTI:
                model = nn.DataParallel(model)
            model = model.to(device)
            print("Loaded model.")
    
    
        #%% Create datasets
        try:
            train_dataset
        except:
            train_dataset  = Im_Dataset(mode = "train",class_num = positive_class, class_balance = True)
            val_dataset = Im_Dataset(mode = "val")
            datasets = {"train":train_dataset, "val": val_dataset}
            print("Loaded datasets.")
        
        start_epoch = -1
        num_epochs = 50
    
        #loss = torch.nn.MSELoss()
        loss = nn.BCELoss()
        #loss = weightedBCELoss()
    
        optimizer = optim.SGD(model.parameters(), lr= 0.03,momentum = 0.1)    
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor = 0.3, mode = "max", patience = 2,verbose=True)
    
        checkpoint = None
        if positive_class == 0:
            checkpoint = "/home/worklab/Documents/Derek/Classification-ISIC-2018/class0_epoch18.pt"
#        elif positive_class == 1:
#            checkpoint = "/home/worklab/Documents/Derek/Classification-ISIC-2018/class1_epoch0.pt"
        if checkpoint:
          model,optimizer,start_epoch,all_losses,all_accs = load_model(checkpoint,model, optimizer)
          print("Reloaded checkpoint {}.".format(checkpoint))
        if True:    
        # train model
            print("Beginning training on {}.".format(device))
            model,all_losses,all_accs = train_model(model,  optimizer, scheduler,
                                loss, datasets,positive_class,device,
                                num_epochs, start_epoch+1,all_losses= None,all_accs= None)
        del model
        del train_dataset
        del val_dataset
        torch.cuda.empty_cache()
        
        # if things go badly awry, get rid of weighting and replace softmax in network