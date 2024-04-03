#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                            
# PROGRAMMER: Jen Berenguel                                                   
# DATE CREATED: 03/23/2024                                  
# REVISED DATE:                                 
# PURPOSE:  Trains on a set of images using a pretrained CNN model.  

from utils import (
    get_input_args, 
    check_subfolders, 
    label_mapping, 
    setup_data, 
    prettify, 
    plot_losses)

from classifier import (
    build_network, 
    train_network, 
    save_checkpoint)


def main():
    prettify('title')

    # Get command line arguments using in_arg
    in_args = get_input_args('train')

    prettify('folder')

    # check to see if the data folder's structure is correct.
    # If it is extract the number of labels used
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
    
    # TODO ADD option to view configurationss

    trainloader, testloader, validloader = setup_data(in_args.dir)

    model = build_network(in_args.arch, in_args.hidden_layers, count, in_args.dropout)
    
    checkpoint, train_losses, test_losses = train_network(model, trainloader, validloader, in_args.epochs, in_args.lr, in_args.device, in_args.dropout, in_args.arch, count, in_args.hidden_layers)

    save_checkpoint(checkpoint)

    plot_losses(train_losses, test_losses)

# Call to main function to run the program
if __name__ == "__main__":
    main()
