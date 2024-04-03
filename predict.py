#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                            
# PROGRAMMER: Jen Berenguel                                                   
# DATE CREATED: 03/23/2024                                  
# REVISED DATE:                                 
# PURPOSE:

# Imports functions for this program
import os
from utils import get_input_args, check_input_type, label_mapping, print_predictions, find_image_files
from classifier import load_model, predict
import csv
import pandas as pd

def main():
    # Get command line arguments using in_arg  
    in_args = get_input_args('predict')

    model = load_model(in_args.model_path)

    input_type = check_input_type(in_args.input_data)

    # print(input_type)
    labels = label_mapping(in_args.labels)

    if input_type == 'folder':

        field_names = ['image_file', 'expected', 'prediction', 'probability', 'match']
        all_predictions = []

        image_files = find_image_files(in_args.input_data)

        print(image_files)

        for image_file in image_files:
                probs, classes = predict(image_file, model, labels, 1)
                expected = labels[image_file.split('/')[-2]]
                all_predictions.append({
                    'image_file': image_file,
                    'expected': expected,
                    'prediction': classes[0],
                    'probability': round(float(probs[0]) * 100, 2), 
                    'match':  expected == classes[0]
                })

        df = pd.DataFrame(all_predictions)

        print(df.head())
        
        with open('predictions.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = field_names) 
            writer.writeheader() 
            writer.writerows(all_predictions) 



    elif input_type == 'single':
        probs, classes = predict(in_args.input_data, model, labels, in_args.topk)
        print_predictions(probs, classes)
    else:
        raise Exception("Input type must be a single images or a folder containing images")

    # TODO Plot barchart and image if it is a single image
    # TODO Save predictions csv? if it is a set of images

# Call to main function to run the program
if __name__ == "__main__":
    main()