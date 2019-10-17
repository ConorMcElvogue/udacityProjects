import argparse
import torch
import torchvision
import json
import torch.nn.functional as nFunc
import matplotlib.pyplot as plt
import numpy as np 
from torch import nn
from torch import optim
from PIL import Image
from torchvision import datasets, transforms, models
from collections import OrderedDict
from matplotlib.ticker import FormatStrFormatter
from funcs_utils import data_loader, model_setup, train_val_model, save_model


parse = argparse.ArgumentParser(description='Train Network') 


parse.add_argument('data_dir',action = 'store',default="./flowers/",help='Please provide the path to training data')
parse.add_argument('--save_dir',action = 'store',dest='save_dir',
                   default='/home/workspace/ImageClassifier/checkpoint.pth')

parse.add_argument('--epochs',action = 'store',dest ='epochs',type=int, default=4)
parse.add_argument('--l_rate', action="store", dest='l_rate',type=float, default=0.001)
parse.add_argument('--gpu',action='store',dest='gpu', default='gpu')
parse.add_argument('--arch', action='store', dest='architecture', default='vgg16')

parse.add_argument('--h_layers',action='store',dest="h_layers",type=int,default=4096)


#ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)

res = parse.parse_args()

data_dir     = res.data_dir
save_dir     = res.save_dir
l_rate       = res.l_rate
h_layers     = res.h_layers
epochs       = res.epochs 
device       = res.gpu
architecture = res.architecture

train_loader, test_loader, val_loader,train_data = data_loader(data_dir) 

model, optimizer, criterion = model_setup(architecture,h_layers,l_rate,train_data) 

model = train_val_model(model,criterion,optimizer,epochs,device,train_loader, val_loader)

save_model(save_dir,model,architecture,train_data) 
