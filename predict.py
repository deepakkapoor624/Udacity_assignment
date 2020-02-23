import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data
import pandas as pd
from collections import OrderedDict
from PIL import Image
import argparse
import json

parser = argparse.ArgumentParser(description="Train a Neural Network (NN) using transfer learning")
# 1. The directory to the image files
parser.add_argument ('image_dir', help = 'Provide path to image. Mandatory argument', type = str)
parser.add_argument ('load_dir', help = 'Provide path to checkpoint. Mandatory argument', type = str)
parser.add_argument ('--top_k', help = 'Top K most likely classes. Optional', type = int)
parser.add_argument ('--category_names', help = 'Mapping of categories to real names. JSON file name to be provided. Optional', type = str)
# 5. Choose the GPU for training
parser.add_argument('--gpu', action='store_true',
                    help="If you would like to use the GPU for training. Default is False. if you want to use the gpu enter True or gpu or cuda")  

#preparing class_names using mapping with cat_to_name
class_names = [cat_to_name [item] for item in classes]

def load_model(file_path):
    checkpoint = torch.load (file_path) #loading checkpoint from a file
    if checkpoint ['arch'] == 'alexnet':
        model = models.alexnet (pretrained = True)
    else: #vgg13 as only 2 options available
        model = models.vgg13 (pretrained = True)
    model.classifier = checkpoint ['classifier']
    model.load_state_dict (checkpoint ['state_dict'])
    model.class_to_idx = checkpoint ['mapping']

    for param in model.parameters():
        param.requires_grad = False #turning off tuning of the model

    return model

# function to process a PIL image for use in a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open (image) #loading image

    # smallest part: width or height should be kept not more than 256
    im.thumbnail((256,256))
    
    width, height = im.size #new size of im
    #crop 224x224 in the center
    crop = 224
    left = (width - crop)/2
    top = (height - crop)/2
    right = left + 224
    bottom = top + 224
    im = im.crop((left, top, right, bottom))

    #preparing numpy array
    np_image = np.array (im)/255 #to make values from 0 to 1
    np_image -= np.array ([0.485, 0.456, 0.406])
    np_image /= np.array ([0.229, 0.224, 0.225])

    #PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array.
    #The color channel needs to be first and retain the order of the other two dimensions.
    np_image= np_image.transpose ((2,0,1))
    return np_image

#defining prediction function
def predict(image_path, model, topkl, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file
    image = process_image (image_path) #loading image and processing it using above defined function

    #we cannot pass image to model.forward 'as is' as it is expecting tensor, not numpy array
    #converting to tensor
    if device == 'cuda':
        im = torch.from_numpy (image).type (torch.cuda.FloatTensor)
    else:
        im = torch.from_numpy (image).type (torch.FloatTensor)

    im = im.unsqueeze (dim = 0) #used to make size of torch as expected. as forward method is working with batches,
    #doing that we will have batch size = 1

    #enabling GPU/CPU
    model.to (device)
    im.to (device)

    with torch.no_grad ():
        output = model.forward (im)
    output_prob = torch.exp (output) #converting into a probability

    probs, indeces = output_prob.topk (topkl)
    probs = probs.cpu ()
    indeces = indeces.cpu ()
    probs = probs.numpy () #converting both to numpy array
    indeces = indeces.numpy ()

    probs = probs.tolist () [0] #converting both to list
    indeces = indeces.tolist () [0]

    mapping = {val: key for key, val in
                model.class_to_idx.items()
                }

    classes = [mapping [item] for item in indeces]
    classes = np.array (classes) #converting to Numpy array

    return probs, classes

#setting values data loading
args = parser.parse_args ()
file_path = args.image_dir

#defining device: either cuda or cpu
if args.gpu == 'gpu' and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

#loading JSON file if provided, else load default file name
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

#loading model from checkpoint provided
model = load_model(args.load_dir)

#defining number of classes to be predicted. Default = 1
nm_cl = args.top_k

#calculating probabilities and classes
probs, classes = predict (file_path, model, nm_cl, device)


for i in range (nm_cl):
     print("Number: {}/{} ".format(i+1, nm_cl),
            "Class name: {} ".format(class_names [i]),
            "Probability: {}% ".format(probs [i]*100),
            )
