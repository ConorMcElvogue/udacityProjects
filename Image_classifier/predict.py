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
from funcs_utils import load_model, predict,model_setup

parse = argparse.ArgumentParser(description='Predict Image')

#check that the folder exists for this input name 
parse.add_argument('--image_loc',action ='store',default = "flowers/train/1/image_06734.jpg")

parse.add_argument('--checkpoint',action='store',dest='check_loc',default = 'checkpoint.pth')

parse.add_argument('--top_k',action='store',dest ="top_k",type=int,default=1) 

parse.add_argument('--cat_name',action='store',dest='cat_name_dir',default ='cat_to_name.json') 

parse.add_argument('--gpu',action='store',dest='gpu', default='cuda',help='cpu used by default, to use GPU please use --gpu "cuda"') 

res = parse.parse_args()

image_path = res.image_loc
checkpoint = res.check_loc 
top_k      = res.top_k
device     = res.gpu
cat_name   = res.cat_name_dir

with open(cat_name, 'r') as f:
    cat_to_name = json.load(f)

model = load_model(checkpoint,model,"cpu")

probs, flower_list = predict(image_path,model,top_k,device)

flower_names = [] 

for flower in flower_list:
    flower_names.append(cat_to_name[flower])
 
for i in range(0,len(flower_names)):
    print(flower_names[i])
    print(probs[i])

    
