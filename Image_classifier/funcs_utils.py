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


def data_loader(data_dir): 

    '''
    Args   : Filepath to Images 
    Return : Train, validation, test datasets 
    Desc   : Transforms the images found in the specified directories 
             using various transform functionality (resize,crop) etc 
             to help generalize the network/improve preformance
    '''
    
    #data_dir  = './flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir  = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir,transform = train_transforms)
    val_data   = datasets.ImageFolder(test_dir,transform = test_transforms) 
    test_data  = datasets.ImageFolder(test_dir,transform = test_transforms) 
    
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
    val_loader   = torch.utils.data.DataLoader(train_data,batch_size=32,shuffle=True)
    test_loader  = torch.utils.data.DataLoader(train_data,batch_size=32,shuffle=True)
    
    return train_loader, val_loader,test_loader,train_data
    
def model_setup(architecture,h_layer_1,l_rate,train_data):
    
    '''
    Args   : Pretrained network achitecture, hyperparameters:1st Hidden layer value, learning rate, GPU/CPU device to run on
    Return : Model initalization, criterion and optimizer to be used for training 
    Desc   : Load a pretrianed newtork then 
             Defining a new, untrained feed-forward network as a classifier
    '''
    
    arch = {"vgg16":25088, "densenet121": 1000} 
    
    if architecture =='vgg16':
        model = models.vgg16(pretrained=True)
    elif architecture =='densenet121':
        model = models.densenet121(pretrained=True) 
    else:
        print("This pre-trained model is not avaliable, please select either vgg16 or densenet121")
    
    for param in model.parameters(): 
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                           ('fc1',nn.Linear(arch[architecture],h_layer_1)),
                           ('reLu1',nn.ReLU()),
                           ('dropout1',nn.Dropout(p=0.5)),
                           ('fc2',nn.Linear(h_layer_1,len(train_data.class_to_idx))),
                           ('output',nn.LogSoftmax(dim=1))
                           ]))
                
    model.classifier = classifier
    
    criterion = nn.NLLLoss() 
    optimizer = optim.Adam(model.classifier.parameters(),l_rate)
    return model, optimizer, criterion 
    
    
def train_val_model(model, criterion,optimizer, epochs,device,train_loader,val_loader):

    '''
    Args   : model,criterion,optimizer,no. of Epochs, device to train on (GPU/CPU)
    Return : The trained Model
    Desc   : Training the classifier layers using backpropagation, then using the pretrained network to get features
    '''
    
    if device == 'cuda' and torch.cuda.is_available(): 
        model.to('cuda')
    else: 
        model.to('cpu')
        print("GPU not avaliable - using CPU")
    
    print_every = 40
    steps       = 0 

    for ep in range (epochs):
        running_loss = 0
    
        #Iterating through data to carry out training step
        for ii,(inputs, labels) in enumerate(train_loader):
            steps += 1 
            inputs,labels = inputs.to(device),labels.to(device)
        
            #setting the gradients back to 0 
            optimizer.zero_grad()
        
            op = model.forward(inputs)
            loss = criterion(op,labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval() 
                with torch.no_grad(): 
                    val_loss = 0
                    val_accuracy = 0 
                    for ii, (val_inputs,val_labels) in enumerate(val_loader):
                        val_inputs,val_labels = val_inputs.to(device),val_labels.to(device)
                        outputs = model.forward(val_inputs)
                        val_loss += criterion(outputs,val_labels).item()
                        prob    = torch.exp(outputs)
                        equality = (val_labels.data == prob.max(dim=1)[1])
                        val_accuracy += equality.type(torch.FloatTensor).mean()
                     
                print("Epoch: {}/{}... ".format(ep+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss {:.4f}".format(val_loss/len(val_loader)),
                      "Accuracy: {:.4f}".format(val_accuracy/len(val_loader)))
                      
                running_loss = 0
                model.train()
                
    print("training model complete")        
    return model 
  
def save_model(path,model,architecture,train_data):
    
    '''
    Args   : directory of location to save model, the trained model
    Return : N/A
    Desc   : Saving the trained model for future use
    '''
    
    model.class_to_idx = train_data.class_to_idx

    torch.save({'arch'        :architecture,
                'classifier'  :model.classifier,
                'state_dict'  :model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)
                
def load_model(filepath,device):

    '''
    Args   : the location of the saved model
    Return : The trained Model
    Desc   : Loading the trained model for future use
    '''
    
    if device == 'cuda' and torch.cuda.is_available(): 
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
        
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print("Model not supported")
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model 
     
def process_image(image): 
    
    '''
    Args   : The image to be processed
    Return : Processed image (Tensor)
    Desc   : Applies transformations required to predict image
    '''
    
    pil_img = Image.open(image)
    process_img = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]) 
       
    pil_img = process_img(pil_img)
    
    return pil_img
    
def predict(img_path,model,topk=5,device="cuda"):
    
    '''
    Args   : image to be predicted, model to use, number of top predictions to return
    Return : probabilities and labels of top N predictions
    Desc   : Used the saved model to predict the probability of a given image, returning the N most likely results
    '''
    
    model.eval()
    if device == 'cuda' and torch.cuda.is_available(): 
        model.to('cuda')
    else:
        model.to('cpu')
    
    image       = process_image(img_path)
    image       = image.unsqueeze_(0)
    image       = image.float()
    idx         = model.class_to_idx
    flower_list = []
    
    with torch.no_grad():
        if device == 'cuda' and torch.cuda.is_available():  
            output = model.forward(image.cuda())
        else:
            output = model.forward(image)
    
    probability = nFunc.softmax(output.data,dim=1)
    probs,labs = probability.topk(topk)
  
    probs = probs.tolist()
    labs  = labs.tolist()   
    
    idx_to_class = {v: k for k, v in idx.items()}
    for vals in labs[0]:
        flower_list.append(idx_to_class[vals])

    return probs[0], flower_list
  
