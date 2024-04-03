# Udacity AIPND - Project 2
# python3
# -*- coding: utf-8 -*-
#                                                                            
# PROGRAMMER: Jen Berenguel                                                   
# DATE CREATED: 03/23/2024                                  
# REVISED DATE:                            
# PURPOSE:  Defines utility functions necessary for the command line applications  
#           to run.

import os
import re
import time
import json
import argparse
from datetime import datetime

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog

from PIL import Image

def get_input_args(command: str):
    """
    Create a function that retrieves the following 3 command line inputs 
    from the user using the Argparse Python module. If the user fails to 
    provide some or all of the 3 inputs, then the default values are
    used for the missing inputs.
     Command Line Arguments:
        train:
            1. Image Folder as --dir with default value 'flowers/'
            2. CNN Model Architecture as --arch with default value 'densenet'
            3. JSON or Text file with the label mapping as --labels with default value 'cat_to_name.json'
            4. Learning rate as --lr with default value 0.003
            5. Dropout rate as --dropout with default value 0.2
            6. Hidden layers as --hidden_layers. A list of comma-seprated integers With default value '[512]'
            7. Epochs as --epochs with default value 3
            8. Device as --device with default value 'gpu'
        predict:
            1. Input data as --input_data.  Required
            2. Model to be loaded as --model_path with default value 'checkpoint.pth'
            3. JSON or Text file with the label mapping as --labels with default value 'cat_to_name.json'
            5. top K classes to display as --topk with default value 5
    Parameters:
    - command (str): Selects mode 'train' or 'predict'.

    Returns:
    - args: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()

    def list_of_ints(arg):
        try:
            numbers = re.findall(r'\d+', arg) 
            return [int(num) for num in numbers]
        
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid list of integers. Please enter comma-separated integers.")
                                         
    if command == 'train':
        parser.add_argument('--dir', type = str, default = 'flowers/',
                            help = 'path to the folder containing subfolders \
                                    \'test\', \'train\', and \'valid\' each organized \
                                    by category') 
        parser.add_argument('--arch', type = str, default = 'densenet',
                            choices = ['vgg', 'alexnet', 'densenet', 'resnet'],
                            help = 'The CNN model architecture: \
                                    \'resnet\', \'alexnet\', \'densenet\', or \'vgg\'')
        parser.add_argument('--labels', type = str, default = 'cat_to_name.json',
                            help = 'path to file that contains a .json or .txt file to \
                                    be convereted to a dictionatry of labels.')
        parser.add_argument('--lr', type = float, default = 0.003,
                            help = 'the learning rate of the model.')
        parser.add_argument('--dropout', type = float, default= 0.2,
                            help = 'The dropout rate for each layer of neurons')
        parser.add_argument('--hidden_layers', type = list_of_ints, default = [512],
                           help = 'A list of integers specifying the number of neurons for each hidden layer \
                            in the custom classifier.  Does not include input_features, or final output_features.')
        parser.add_argument('--epochs', type = int, default = 1,
                            help = 'The number of epochs to train the data on')
        parser.add_argument('--device', type = str, default='gpu', choices= ['cpu','gpu'],
                            help = 'cpu\' or \'gpu\'.  If \'gpu\' is selected but no cuda \
                                    device is available, will default to \'cpu\'')   
    
    elif command == 'predict':
        parser.add_argument('--input_data', type=str,
                            help='Path to the input data for prediction.')
        parser.add_argument('--model_path', type=str, default='checkpoint.pth', 
                            help='Path to .pth checkpoint of model.')
        parser.add_argument('--labels', type = str, default = 'cat_to_name.json',
                            help = 'path to file that contains a .json or .txt file to \
                                    be convereted to a dictionatry of labels.')
        parser.add_argument('--topk', type = int, default=5,
                            help = 'choose how many of the top predictions to display')
        
    args = parser.parse_args()
    return args 

def check_subfolders(path):
    # TODO docstring check_subfolders
    """_summary_

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """
    # List all subfolders in the main folder
    subfolders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    
    expected_subfolders = ['test', 'train', 'valid']
    
    if len(subfolders) != 3 or sorted(subfolders) != expected_subfolders:
        print("Error: The main folder should contain exactly three subfolders named 'train', 'valid', and 'test'.")
        return False, 0

    # Iterate over each subfolder
    final_subfolder_names = {}
    final_subfolder_counts = {}
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(path, subfolder)
        
        # List all sub-subfolders
        sub_subfolders = [f for f in os.listdir(subfolder_path) if os.path.isdir(os.path.join(subfolder_path, f))]
        
        if not sub_subfolders:
            print(f"Error: Subfolder {subfolder} does not contain any subfolders.")
            return False, 0
        
        # Get the class label folders name and count
        final_subfolder_name = os.path.basename(os.path.normpath(sub_subfolders[0]))
        final_subfolder_count = len(sub_subfolders)
        
        final_subfolder_names[subfolder] = final_subfolder_name
        final_subfolder_counts[subfolder] = final_subfolder_count

    # Check if all class label folders names are the same
    if len(set(final_subfolder_names.values())) == 1:
        print(f"All class label folders have the same name: {final_subfolder_names[subfolder]}")
        
        # Check if all class label folders counts are the same
        if len(set(final_subfolder_counts.values())) == 1:
            print(f"Number of class label folders: {final_subfolder_counts[subfolder]}")
            return True, final_subfolder_counts[subfolder]
        else:
            print("Error: class label folders do not have the same count in all three main subfolders.")
            return False, 0
    else:
        print("Error: class label folders do not have the same name in all three main subfolders.")
        return False, 0
    
def setup_data(path: str):
    """
    Sets up data loaders for the neural network.

    Parameters:
        path (str): The path to the data folder containing subfolders 'test', 'train', and 'valid', each organized by category.

    Returns:
        trainloader (torch.utils.data.DataLoader): DataLoader for the train dataset.
        testloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        validloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    """
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)


    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)    
    
    return trainloader, testloader, validloader

def label_mapping(path: str):
    """
    Fucntion that creates a dict with the label mapping used for classification

    Args:
        path (str): path to .json or .txt file containing the label mapping

    Returns:
        label_dict: a dict of indices as keys and category names as values
    """
    # Check if the file path ends with '.json'
    if path.endswith('.json'):
        try:
            with open(path, 'r') as file:
                label_dict = json.load(file)
            return label_dict
        except json.JSONDecodeError:
            print(f"Error decoding JSON file at {path}")
            return None
    

    # Check if the file path ends with '.txt'
    elif path.endswith('.txt'):
        try:
            with open(path, 'r') as file:
                lines = file.readlines()
            # Create a dictionary from the lines in the text file
            data = {}
            for line in lines:
                key, value = line.strip().replace("'", "").replace("{", "").replace("}", "").split(': ')
                data[int(key)] = value
            return data
        except Exception as e:
            print(f"Error reading TXT file at {path}: {e}")
            return None
    
    else:
        print(f"Unsupported file format at {path}. Please provide a JSON or TXT file.")
        return None
    
def wait(seconds):
    # TODO docstring for wait
    """_summary_

    Args:
        seconds (_type_): _description_
    """
    print("\n")
    for i in range(seconds):
        print('.', end='')
        time.sleep(1)
    print("\n")

def prettify(text):
    # TODO docstring for prettify
    """_summary_

    Args:
        text (_type_): _description_
    """
    if text != 'title':
        wait(5)

    switcher = {
        'train': '\n\
    ▀█▀ █▀█ ▄▀█ █ █▄░█ █ █▄░█ █▀▀\n\
    ░█░ █▀▄ █▀█ █ █░▀█ █ █░▀█ █▄█',
        
        'classifier': None,
        
        'title': '\n\
      █▀█ █▀█ █▀█ █▀▀ █▀█ ▄▀█ █▀▄▀█ █▀▄▀█ █▀▀ █▀▄   █▄▄ █▄█\n\
      █▀▀ █▀▄ █▄█ █▄█ █▀▄ █▀█ █░▀░█ █░▀░█ ██▄ █▄▀   █▄█ ░█░\n\
    \n\
    ░░░░░██╗███████╗███╗░░██╗██████╗░░█████╗░██╗░░░░░██╗░░░░░\n\
    ░░░░░██║██╔════╝████╗░██║██╔══██╗██╔══██╗██║░░░░░██║░░░░░\n\
    ░░░░░██║█████╗░░██╔██╗██║██║░░██║██║░░██║██║░░░░░██║░░░░░\n\
    ██╗░░██║██╔══╝░░██║╚████║██║░░██║██║░░██║██║░░░░░██║░░░░░\n\
    ╚█████╔╝███████╗██║░╚███║██████╔╝╚█████╔╝███████╗███████╗\n\
    ░╚════╝░╚══════╝╚═╝░░╚══╝╚═════╝░░╚════╝░╚══════╝╚══════╝',
        
        'folder': '\n\
    █▀▀ █░█ █▀▀ █▀▀ █▄▀ █ █▄░█ █▀▀   █▀▄ ▄▀█ ▀█▀ ▄▀█   █▀ ▀█▀ █▀█ █░█ █▀▀ ▀█▀ █░█ █▀█ █▀▀\n\
    █▄▄ █▀█ ██▄ █▄▄ █░█ █ █░▀█ █▄█   █▄▀ █▀█ ░█░ █▀█   ▄█ ░█░ █▀▄ █▄█ █▄▄ ░█░ █▄█ █▀▄ ██▄',

        'checkpoint': None
    }
    
    print(switcher.get(text, "Invalid option"))
     
def pause():
    # TODO DO I even need this - pause()
    input("Press Enter to continue...")

def plot_losses(train_losses, test_losses):
    # TODO docstring for plot_losses
    """_summary_

    Args:
        train_losses (_type_): _description_
        test_losses (_type_): _description_
    """


    # Your code to plot the graph
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)

    # Get the current date and time
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Set the default filename
    default_filename = f"plot_{current_datetime}.png"

    # Create a Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Ensure the dialog box appears in front of other apps

    # Open the "Save As" dialog box
    file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                            initialfile=default_filename,
                                            filetypes=[("PNG files", "*.png"), ("All Files", "*.*")])

    # Check if a file was selected
    if file_path:
        # Save the plot to the selected file
        plt.savefig(file_path)
        print(f"Plot saved to {file_path}")
    else:
        print("Save operation was cancelled.")

    # Display the plot
    plt.show()

def check_input_type(input_path):
    """
    Check if the input is a single image or a folder containing multiple images.
    
    Args:
    - input_path (str): Path to the input image or folder.
    
    Returns:
    - str: 'single' if the input is a single image, 'folder' if it's a folder.
    """
    image_extensions = ['.jpg', '.jpeg', '.png']
    
    if os.path.isfile(input_path):
        # Check if the input is a single image by checking the file extension
        _, ext = os.path.splitext(input_path)
        if ext.lower() in image_extensions:
            return 'single'
    elif os.path.isdir(input_path):
        # Walk through all folders and check if image is present
        for root, dirs, files in os.walk(input_path):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext.lower() in image_extensions:
                    return 'folder'
    return 'unknown'

# TODO: Process a PIL image for use in a PyTorch model
def process_image(image_path):
    # TODO docstring process_image
    """_summary_

    Args:
        image_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = Image.open(image_path)
    image = transform(image)
    return image

def print_predictions(tensor, flower_names):
    # TODO docstring print_predictions
    """_summary_

    Args:
        tensor (_type_): _description_
        flower_names (_type_): _description_
    """
    tensor = tensor.cpu()
    tensor = tensor.numpy() if torch.is_tensor(tensor) else tensor
    for i in range(len(tensor)):
        percentage = tensor[i] * 100  # Convert the value to percentage
        rounded_percentage = round(percentage, 2)  # Round to 2 decimal places
        print(f"{flower_names[i]:<15}: {rounded_percentage:>6.2f}%")

def find_image_files(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                root = root.replace('\\', '/')
                image_files.append(root + '/' + filename)
                
    return image_files