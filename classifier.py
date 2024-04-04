#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                            
# PROGRAMMER: Jen Berenguel                                                   
# DATE CREATED: 03/23/2024                                  
# REVISED DATE: 04/03/2024                       
# PURPOSE:

# Standard Library Imports
from datetime import datetime

# Third-Party Library Imports
import numpy as np
import torch
from torch import nn, optim
from torchvision import models
import tkinter as tk
from tkinter import filedialog

# Local Imports
from utils import prettify, process_image

# Pretrained networks
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

    # Add hidden layers to model
    for i in range(len(hidden_units)):
        if i == len(hidden_units)-1:
            classifier.add_module('layer'+ str(i+2), nn.Linear(hidden_units[-1], n_classes))
            break
        else:
            classifier.add_module('layer'+ str(i+2), nn.Linear(hidden_units[i], hidden_units[i+1]))
            classifier.add_module('reul' + str(i+2), nn.ReLU())
            classifier.add_module('dropout' + str(i+2), nn.Dropout(p=dropout_rate))

    classifier.add_module('softmax', nn.LogSoftmax(dim=1))

    # Define the classifier based on the model architecture
    if arch == 'resnet':
        model.fc = classifier
    else:
        model.classifier = classifier

    print('*****************Custom Classifer*****************')
    print(classifier)

    return model

def train_network(model, trainloader, validloader, epochs: int, lr: float, device_choice: str, dropout: float, count: int, hidden_layers, train_data):
    """
    Train a neural network model.

    Args:
    - model (nn.Module): The neural network model to be trained.
    - trainloader (DataLoader): The data loader for the training dataset.
    - validloader (DataLoader): The data loader for the validation dataset.
    - epochs (int): The number of training epochs.
    - lr (float): The learning rate for the optimizer.
    - device_choice (str): The device to be used for training ('cpu' or 'gpu').
    - dropout (float): The dropout rate.
    - count (int): The number of output classes.
    - hidden_layers (list): A list containing the sizes of the hidden layers.
    - train_data: The training data used for training.

    Returns:
    - checkpoint: The checkpoint dictionary
    - training losses: a list containg the train loss for each step
    - test losses:: a list containing the test loss for each step
    """

    cpu = torch.device('cpu')
    gpu = torch.device('cuda')

    # Set device according to user choice. If GPU is selected 
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

    # Define the criterion for the loss function
    criterion = nn.NLLLoss()
    
    # Determine the architecture of the model and set the optimizer accordingly
    arch = type(model).__name__.lower()
    if arch == 'resnet':
        optimizer = optim.Adam(model.fc.parameters(), lr=lr)    
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    train_losses, test_losses = [], []

    # Print a pretty message indicating the start of training
    prettify('train')

    # Loop through each epoch
    for e in range(epochs):
        # Loop through each batch in the training data
        for images, labels in trainloader:
            steps += 1
            optimizer.zero_grad()
            
            # Move image and label tensors to the selected device
            images, labels = images.to(device), labels.to(device)

            # Forward pass to get the log probabilities
            log_ps = model(images)
            # Calculate the loss
            loss = criterion(log_ps, labels)
            running_loss += loss.item()
            
            # Backward pass to update the weights
            loss.backward()
            optimizer.step()

            # Print training loss and validation metrics every 'print_every' steps
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    # Loop through each batch in the validation data
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        _, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                # Calculate average training and validation losses and accuracy
                train_loss  = running_loss/print_every
                test_loss = test_loss/len(validloader)
                accuracy = accuracy/len(validloader)

                # Append the losses to the lists for plotting
                train_losses.append(train_loss)
                test_losses.append(test_loss)

                # Print the epoch, training loss, validation loss, and accuracy
                print(f"Epoch {e+1}/{epochs}.. "
                    f"Train loss: {train_loss:.3f}.. "
                    f"Test loss: {test_loss:.3f}.. "
                    f"Test accuracy: {accuracy * 100:.2f}%")
                
                running_loss = 0
                model.train()  # Put back in training mode for next pass

    # Save necessary information in the checkpoint dictionary
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
    """
    Save a PyTorch checkpoint to a specified file using a Tkinter file dialog.

    Args:
    - checkpoint (dict): The checkpoint dictionary to be saved.

    Returns:
    None
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

def load_model(filepath):
    """
    Load a PyTorch model from a specified checkpoint file.

    Args:
    - filepath (str): The path to the checkpoint file.

    Returns:
    - model: the reconstructed model 
    - class_idx_mapping: class index mapping for label prediction
    """

    # Load the checkpoint
    if torch.cuda.is_available():
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    
    # Reconstructing the model
    model = checkpoint['model']

    # Determine the model architecture
    arch = type(model).__name__.lower()
    
    # Define the optimizer based on the model architecture
    if arch == 'resnet':
        optimizer = optim.Adam(model.fc.parameters(), lr=checkpoint['lr'])    
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['lr'])

    # Set all model parameters to not require gradients
    for param in model.parameters():
        param.requires_grad = False
    
    # Build custom classifier
    classifier = nn.Sequential()

    # Add input layer to classifier
    classifier.add_module('layer1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'][0]))
    classifier.add_module('relu1', nn.ReLU())
    classifier.add_module('dropout1', nn.Dropout(p=checkpoint['dropout']))

    # Add hidden layers to classifier
    for i in range(len(checkpoint['hidden_layers'])):
        if i == len(checkpoint['hidden_layers'])-1:
            classifier.add_module('layer'+ str(i+2), nn.Linear(checkpoint['hidden_layers'][-1], checkpoint['output_size']))
            break
        else:
            classifier.add_module('layer'+ str(i+2), nn.Linear(checkpoint['hidden_layers'][i], checkpoint['hidden_layers'][i+1]))
            classifier.add_module('relu' + str(i+2), nn.ReLU())
            classifier.add_module('dropout' + str(i+2), nn.Dropout(p=checkpoint['dropout']))

    # Add softmax activation to classifier
    classifier.add_module('softmax', nn.LogSoftmax(dim=1))

    # Replace the model's classifier with the custom classifier based on the model architecture
    if arch == 'resnet':
        model.fc = classifier
    else:
        model.classifier = classifier
    
    # Load the model's state_dict
    model.load_state_dict(checkpoint['state_dict'])
    
    # Load the optimizer's state_dict
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get the class index mapping from the checkpoint
    class_idx_mapping = checkpoint['class_idx_mapping']
    
    # Create a dictionary for index to class mapping
    idx_class_mapping = {v: k for k, v in class_idx_mapping.items()}

    return model, idx_class_mapping

def predict(image_path, model, labels, idx_class_mapping, device_choice, topk=5):
    """
    Predict the top classes and their probabilities for a given image.

    Args:
    - image_path (str): The path to the image file.
    - model (nn.Module): The trained neural network model.
    - labels (list): List of label names.
    - idx_class_mapping (dict): Dictionary mapping class indices to class names.
    - device_choice (str): The device to be used for training ('cpu' or 'gpu').
    - topk (int, optional): The number of top classes to return. Default is 5.

    Returns:
    - probs: the probabilities for the 'topk' classes
    - classes: the label name for the 'topk' classes
    """

    if device_choice == 'gpu':
        # Set device to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
     
    # Process the image
    img = process_image(image_path)
    img = np.expand_dims(img, axis=0)
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor).to(device)
    
    # Forward pass to get the log probabilities
    with torch.no_grad():
        logps = model.forward(img_tensor)
    
    # Calculate the probabilities
    probabilities = torch.exp(logps)
    probs, indices = probabilities.topk(topk)
    
    # Convert tensors to numpy arrays
    probs = probs.cpu().numpy().squeeze()
    indices = indices.cpu().numpy().squeeze()
    indices = np.atleast_1d(indices)
    probs = np.atleast_1d(probs)
    
    # Get the class names from indices
    classes = [idx_class_mapping[index] for index in indices]
    classes = [labels[x] for x in classes]

    return probs, classes