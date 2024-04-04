#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                            
# PROGRAMMER: Jen Berenguel                                                   
# DATE CREATED: 03/23/2024                                  
# REVISED DATE: 04/03/2024                      

# Local Imports
from utils import (
    get_input_args,
    check_subfolders,
    setup_data,
    prettify,
    plot_losses,
    wait
)

from classifier import (
    build_network,
    train_network,
    save_checkpoint
)


def main():
    """
    Main function to train a neural network model using the specified configurations.

    The function performs the following steps:
    1. Prints a title using the prettify function.
    2. Retrieves command line arguments using get_input_args.
    3. Checks the data folder's structure and extracts the number of labels.
    4. Sets up the training and validation data loaders.
    5. Builds the neural network model.
    6. Trains the model and saves the checkpoint.
    7. Plots the training and validation losses.

    Raises:
    - Exception: If the folder structure is not correct.

    """
    prettify('title')

    # Get command line arguments using in_arg
    in_args = get_input_args('train')

    # Print a pretty status message
    prettify('folder')

    # check to see if the data folder's structure is correct.
    # If it is, extract the number of labels used
    result, count = check_subfolders(in_args.dir)

    if result:
        print(f"Folder structure is valid. Number of final layer subfolders: {count}")
    else:
        raise Exception("Folder Structure is not correct. \n\
                        \n\
                        data_folder/ \n\
                        ├── train/ \n\
                        │   ├── label_1/ \n\
                        │   └── ... \n\
                        ├── valid/ \n\
                        │   ├── label_1/ \n\
                        │   └── ... \n\
                        └── test/ \n\
                            ├── label_1/ \n\
                            └── ... \n\
                        \n\
                        'train', 'valid', and 'test' folders must be preset. \n\
                        You can have as many label folders as you want as long \n\
                        as they are consistent.")
    
    wait(5)

    # TODO ADD option to view configurations

    # Set up the data loaders
    trainloader, validloader, train_data = setup_data(in_args.dir)

    # Build the neural network model
    model = build_network(in_args.arch, in_args.hidden_layers, count, in_args.dropout)
    
    # Train the model and get the checkpoint and losses
    checkpoint, train_losses, test_losses = train_network(model, # Model built with the build_network function
                                                          trainloader, # Training dataloader
                                                          validloader,  # Validation dataloader
                                                          in_args.epochs, # Number of epochs to run training
                                                          in_args.lr, # Learning rate for optimizer
                                                          in_args.device, # 'cpu' or 'gpu'
                                                          in_args.dropout, # Dropout rate of hidden layers for training 
                                                          count, # Number of labels
                                                          in_args.hidden_layers, # Hidden layers
                                                          train_data) # Needed for class_idx_mapping
    # Save the model checkpoint
    save_checkpoint(checkpoint)

    # Plot the training and validation losses
    plot_losses(train_losses, test_losses)

# Call to main function to run the program
if __name__ == "__main__":
    main()
