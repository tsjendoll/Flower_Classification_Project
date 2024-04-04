#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                            
# PROGRAMMER: Jen Berenguel                                                   
# DATE CREATED: 03/23/2024                                  
# REVISED DATE:                                 
# PURPOSE:

from datetime import datetime

import torch
from torch import nn, optim
from torchvision import models

import numpy as np

import tkinter as tk
from tkinter import filedialog

from utils import prettify, process_image

# pretrained networks
resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
alexnet = models.alexnet(weights='AlexNet_Weights.DEFAULT')
vgg16 = models.vgg16(weights='VGG16_Weights.DEFAULT')
densenet= models.densenet121(weights='DenseNet121_Weights.DEFAULT')

""" 
a dict containing dicts of each pretrained network's model, 
number of input features required for a custom classifier module,
and the name of the classifier itself 
"""
archs = {'resnet': {'model': resnet50, 
                    'in_features': 2048, 
                    'classifier': 'fc'}, 
        'alexnet': {'model': alexnet, 
                      'in_features': 9216, 
                      'classifier': 'classifier'},
        'vgg':      {'model': vgg16, 
                     'in_features': 25088, 
                     'classifier': 'classifier'},
        'densenet': {'model': densenet, 
                     'in_features': 1024, 
                     'classifier': 'classifier'}}

def build_network(arch: str, hidden_units: list, n_classes: int, dropout_rate: float): 
    """
    Constructs a customized neural network based on a selected pretrained model. The architecture, 
    number of layers, neuron count per layer, number of class labels, and dropout rate are all 
    configurable via the function's parameters.

    Parameters:
    - arch (str): The architecture of the model. Supported architectures include ['vgg', 'alexnet', 'densenet', 'resnet'].
    - hidden_units (list): A list of integers specifying the number of neurons for each hidden layer.
    - n_classes (int): The number of labels the network will use for classification.
    - dropout_rate (float): The dropout rate to be applied to the network.

    Returns:
    - model: The selected pretrained model with a customized classifier appended.
    """

    assert isinstance(arch, str), "arch must be a string"
    assert isinstance(hidden_units, list), "hidden_units must be a list"
    assert all(isinstance(x, int) for x in hidden_units), f"All elements of hidden_units must be integers. {hidden_units}"
    assert isinstance(n_classes, int), "n_classes must be an integer"
    assert isinstance(dropout_rate, float), "dropout_rate must be a float"

    # Load pretrained model based on arch
    model = archs[arch]['model']

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Build custom classifier class
    classifier = nn.Sequential()

    classifier.add_module('layer1', nn.Linear(archs[arch]['in_features'], hidden_units[0]))
    classifier.add_module('relu1', nn.ReLU())
    classifier.add_module('dropout1', nn.Dropout(p=dropout_rate))

    for i in range(len(hidden_units)):
        print(i)
        if i == len(hidden_units)-1:
            classifier.add_module('layer'+ str(i+2), nn.Linear(hidden_units[-1], n_classes))
            break
        else:
            classifier.add_module('layer'+ str(i+2), nn.Linear(hidden_units[i], hidden_units[i+1]))
            classifier.add_module('reul' + str(i+2), nn.ReLU())
            classifier.add_module('dropout' + str(i+2), nn.Dropout(p=dropout_rate))

    classifier.add_module('softmax', nn.LogSoftmax(dim=1))

    if arch == 'resnet':
        model.fc = classifier
    else:
        model.classifier = classifier

    print('*****************Custom Classifer*****************')
    print(classifier)

    return model

def train_network(model, trainloader, validloader, epochs: int, lr: float, device_choice: str, dropout: float, count: int, hidden_layers, train_data):
    # TODO docstring for train_network
    """_summary_

    Args:
        model (_type_): _description_
        trainloader (_type_): _description_
        validloader (_type_): _description_
        epochs (int): _description_
        lr (float): _description_
        device_choice (str): _description_
        dropout (float): _description_
        count (int): _description_
        hidden_layers (_type_): _description_

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """

    cpu = torch.device('cpu')
    gpu = torch.device('cuda')

    # Set device according to user choice.  If GPU is selected 
    # but cuda is not available, default to CPU
    if device_choice == 'cpu':
        device = cpu
        print("Device set to 'cpu'")
    elif device_choice == 'gpu':
        # Set device to CUDA if available,
        if torch.cuda.is_available():
            device = gpu
            print("Device set to 'gpu'")
        else:
            device = cpu
            print("'gpu' was selected but cuda device is not available.  \
                    Setting device to 'cpu' instead.")
    else:
        raise Exception("device must be 'cpu' or 'gpu'")
    
    model.to(device);

    steps = 0
    running_loss = 0
    print_every = 20

    
    criterion = nn.NLLLoss()
    
    arch = type(model).__name__.lower()
    if arch == 'resnet':
        optimizer = optim.Adam(model.fc.parameters(), lr=lr)    
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    train_losses, test_losses, accuracy_list = [], [], []

    prettify('train')

    # Do validation on the test set
    for e in range(epochs):
        for images, labels in trainloader:
            steps += 1
            optimizer.zero_grad()
            
            # Move image and label tensors to the default device
            images, labels = images.to(device), labels.to(device)

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            running_loss += loss.item()
            
            loss.backward()
            optimizer.step()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        _, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


                train_loss  = running_loss/print_every
                test_loss = test_loss/len(validloader)
                accuracy = accuracy/len(validloader)

                train_losses.append(train_loss)
                test_losses.append(test_loss)
                accuracy_list.append(accuracy)

                print(f"Epoch {e+1}/{epochs}.. "
                    f"Train loss: {train_loss:.3f}.. "
                    f"Test loss: {test_loss:.3f}.. "
                    f"Test accuracy: {accuracy * 100:.2f}%")
                
                running_loss = 0
                model.train() #Put back in training mode for next pass

    model.class_idx_mapping = train_data.class_to_idx

    checkpoint = {
        'model': archs[arch]['model'],
        'input_size': archs[arch]['in_features'],  
        'output_size': count,
        'hidden_layers': hidden_layers,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs,
        'dropout': dropout,
        'lr': lr,
        'class_idx_mapping': model.class_idx_mapping
    }

    return checkpoint, train_losses, test_losses

def save_checkpoint(checkpoint):
    # TODO save_checkpoint docstring
    """_summary_

    Args:
        model_state (_type_): _description_
        optimizer_state (_type_): _description_
        arch (_type_): _description_
        count (_type_): _description_
        hidden_layers (_type_): _description_
        epochs (_type_): _description_
        dropout (_type_): _description_
    """

    # Create a Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Ensure the dialog box appears in front of other apps

    # Get the current date and time
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Set the default filename
    default_filename = f"checkpoint_{current_datetime}.pth"

    # Open the "Save As" dialog box
    file_path = filedialog.asksaveasfilename(defaultextension=".pth",
                                            initialfile=default_filename,
                                            filetypes=[("PyTorch Checkpoint", "*.pth"), ("All Files", "*.*")])

    # Check if a file was selected
    if file_path:
        # Save the checkpoint to the selected file
        torch.save(checkpoint, file_path)
        print(f"Checkpoint saved to {file_path}")
    else:
        print("Save operation was cancelled.")

# loads a checkpoint and rebuilds the model
def load_model(filepath):
    # Load the checkpoint
    if torch.cuda.is_available():
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    
    # Reconstructing the model
    model = checkpoint['model']

    arch = type(model).__name__.lower()
    if arch == 'resnet':
        optimizer = optim.Adam(model.fc.parameters(), lr=checkpoint['lr'])    
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['lr'])

    for param in model.parameters():
        param.requires_grad = False
    
    # Build custom classifier class
    classifier = nn.Sequential()

    classifier.add_module('layer1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'][0]))
    classifier.add_module('relu1', nn.ReLU())
    classifier.add_module('dropout1', nn.Dropout(p=checkpoint['dropout']))

    for i in range(len(checkpoint['hidden_layers'])):
        print(i)
        if i == len(checkpoint['hidden_layers'])-1:
            classifier.add_module('layer'+ str(i+2), nn.Linear(checkpoint['hidden_layers'][-1], checkpoint['output_size']))
            break
        else:
            classifier.add_module('layer'+ str(i+2), nn.Linear(checkpoint['hidden_layers'][i], checkpoint['hidden_layers'][i+1]))
            classifier.add_module('reul' + str(i+2), nn.ReLU())
            classifier.add_module('dropout' + str(i+2), nn.Dropout(p=checkpoint['dropout']))

    classifier.add_module('softmax', nn.LogSoftmax(dim=1))

    
    model.classifier = classifier
    
    # Load the model's state_dict
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    class_idx_mapping = checkpoint['class_idx_mapping']
    idx_class_mapping = {v: k for k, v in class_idx_mapping.items()}

    return model, idx_class_mapping

def predict(image_path, model, labels, idx_class_mapping, topk=5):

    # # No need for GPU
    device = torch.device('cpu')
    model.to(device)
    
    model.eval()
     
    img = process_image(image_path)
    img = np.expand_dims(img, axis=0)
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor).to(device)
    
    with torch.no_grad():
        logps = model.forward(img_tensor)
    
    probabilities = torch.exp(logps)
    probs, indices = probabilities.topk(topk)
    
    probs = probs.numpy().squeeze()
    indices = indices.numpy().squeeze()
    indices = np.atleast_1d(indices)
    probs = np.atleast_1d(probs)
    classes = [idx_class_mapping[index] for index in indices]
    classes = [labels[x] for x in classes]

    return probs, classes