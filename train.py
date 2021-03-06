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
parser.add_argument('data_dir',
                    help="The relative path to the image files to train on. It include three folders: 'train', 'test' and 'valid' for training.")
# 2. The path where shoud save the model or the checkpoint
parser.add_argument('--save_dir', default='./',
                    help="The relative path to save the neural network checkpoint")             
# 3. Choose the architecture
parser.add_argument('--arch', default="vgg16",
                    help="The model architecture supported here are  vgg16 or alexnet")
# 4. Set the hyperparameters: Learning Rate, Hidden Units, Epochs.
parser.add_argument('--lr', type=float, default="0.0005",
                    help="The learning rate for the model. Should be very small")

parser.add_argument('--hidden_units', type=int, default=256,
                    help="The number of units in the hidden layer")

parser.add_argument('--output_class', type=int, default=102,
                    help="The number of Output Class")

parser.add_argument('--epochs', type=int, default=2,
                    help="The number of epochs you want to use")

# 5. Choose the GPU for training
parser.add_argument('--gpu', action='store_true',
                    help="If you would like to use the GPU for training. Default is False. Give args --gpu")

args = parser.parse_args()
data_dir = args.data_dir

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
save_directory = args.save_dir
arch = args.arch
lr = args.lr
output_class = args.output_class
num_hidden_units = args.hidden_units
epochs = args.epochs

if args.gpu == True and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

if data_dir:
    # Define your transforms for the training, validation, and testing sets
    train_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                                transforms.Resize((224,224)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])

    valid_data_transforms = transforms.Compose([transforms.Resize((224,224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])

    test_data_transforms = transforms.Compose([transforms.Resize((224,224)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                               ])
    
    
    # Load the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder(train_dir, transform = train_data_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform = valid_data_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_image_datasets, batch_size = 32, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_image_datasets, batch_size = 32, shuffle = True)
  

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def load_model (arch, hidden_units=2048):
    #setting model based on vgg16
    if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        
       
        classifier = nn.Sequential  (OrderedDict ([
                    ('fc1', nn.Linear(25088, 4096)),
                    ('relu1', nn.ReLU()),
                    ('dropout1', nn.Dropout(p = 0.3)),
                    ('fc2', nn.Linear(4096, hidden_units)),
                    ('relu2', nn.ReLU()),
                    ('dropout2', nn.Dropout(p = 0.3)),
                    ('fc3', nn.Linear(hidden_units, output_class)),
                    ('output', nn.LogSoftmax(dim =1))
                    ]))
    #setting model based on default Alexnet ModuleList
    else:
        arch = 'alexnet'
        model = models.alexnet(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
       
        classifier = nn.Sequential(OrderedDict ([
                    ('fc1', nn.Linear(25088, 4096)),
                    ('relu1', nn.ReLU()),
                    ('dropout1', nn.Dropout(p = 0.3)),
                    ('fc2', nn.Linear(4096, hidden_units)),
                    ('relu2', nn.ReLU()),
                    ('dropout2', nn.Dropout (p = 0.3)),
                    ('fc3', nn.Linear(hidden_units, output_class)),
                    ('output', nn.LogSoftmax(dim =1))
                    ]))
    
    model.classifier = classifier
    return model, arch

# Defining validation Function. will be used during training
def validation(model, valid_loader, criterion):
    model.to(device)

    valid_loss = 0
    accuracy = 0
    for inputs, labels in valid_loader:

        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += equals.type(torch.FloatTensor).mean()

    return valid_loss, accuracy

#loading model using above defined functiion
model, arch = load_model (arch, num_hidden_units)

#Actual training of the model
#initializing criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = lr)

#device can be either cuda or cpu
model.to(device)
#setting number of epochs to be run
print_every = 50
steps = 0
model.train()
#runing through epochs
for e in range (epochs):
    running_loss = 0
    for inputs, labels in train_loader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        #where optimizer is working on classifier paramters only
        optimizer.zero_grad()

        # Forward and backward passes
        #calculating output
        outputs = model.forward(inputs)
        #calculating loss (cost function)
        loss = criterion(outputs, labels)
        loss.backward()
        #performs single optimization step
        optimizer.step()
        # loss.item () returns value of Loss function
        running_loss += loss.item()

        if steps % print_every == 0:
            #switching to evaluation mode so that dropout is turned off
            model.eval()
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                valid_loss, accuracy = validation(model, valid_loader, criterion)

            print("Epoch: {}/{}".format(e+1, epochs),
                  "Training Loss: {} ".format(running_loss/print_every),
                  "Valid Loss: {}".format(valid_loss/len(valid_loader)),
                  "Valid Accuracy: {}%".format(accuracy/len(valid_loader)*100))

            running_loss = 0
            # Make sure training is back on
            model.train()

#saving trained Model
model.to ('cpu')
# Save the checkpoint
#saving mapping between predicted class and class name,
model.class_to_idx = train_image_datasets.class_to_idx
#second variable is a class name in numeric

#creating dictionary for model saving
checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'arch': arch,
              'mapping':    model.class_to_idx
             }

torch.save (checkpoint, args.save_dir + '/checkpoint.pth')
