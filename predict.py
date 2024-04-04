#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                            
# PROGRAMMER: Jen Berenguel                                                   
# DATE CREATED: 03/23/2024                                  
# REVISED DATE: 04/03/2024                                 

# Standard Library Imports
import csv

# Third-Party Library Imports
import pandas as pd
import tkinter as tk
from tkinter import filedialog

# Local Imports
from classifier import load_model, predict
from utils import (
    get_input_args,
    check_input_type,
    label_mapping,
    print_predictions,
    find_image_files
)

def main():
    """
    Main function to handle the prediction process based on the input type.

    The function gets command line arguments, loads the model, and predicts the classes and probabilities 
    for either a single image or a folder containing images. The predictions are saved to a CSV file or 
    printed to the console.

    """

    # Get command line arguments using in_arg  
    in_args = get_input_args('predict')

    # Load the model and class mapping
    model, idx_class_mapping = load_model(in_args.model_path)

    # Check the type of input data
    input_type = check_input_type(in_args.input_data)

    # Map the labels
    labels = label_mapping(in_args.labels)

    if input_type == 'folder':

        # Define field names for the CSV file
        field_names = ['image_file', 'expected', 'prediction', 'probability', 'match']
        all_predictions = []

        # Find all image files in the folder
        image_files = find_image_files(in_args.input_data)

        # Loop through each image file to make predictions
        for image_file in image_files:
                probs, classes = predict(image_file, model, labels, in_args.device, idx_class_mapping, 1)
                expected = labels[image_file.split('/')[-2]]
                all_predictions.append({
                    'image_file': image_file,
                    'expected': expected,
                    'prediction': classes[0],
                    'probability': round(float(probs[0]) * 100, 2), 
                    'match':  expected == classes[0]
                })

        # Create a DataFrame from the predictions
        df = pd.DataFrame(all_predictions)

        # Create a Tkinter root window
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.attributes('-topmost', True)  # Ensure the dialog box appears in front of other apps

        # Open the "Save As" dialog box
        file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                initialfile='predictions.csv',
                                                filetypes=[("CSV files", "*.csv"), ("All Files", "*.*")])
        
        # Save the predictions to a CSV file
        if file_path:
            with open('predictions.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = field_names) 
                writer.writeheader() 
                writer.writerows(all_predictions) 
            print(f"Predictions saved to {file_path}")
        else:
            print("Save operation was cancelled.")
            
    elif input_type == 'single':
        # Predict classes and probabilities for a single image
        probs, classes = predict(in_args.input_data, model, labels, idx_class_mapping, in_args.device, in_args.topk)
        print_predictions(probs, classes)
    else:
        raise Exception("Input type must be a single images or a folder containing images")

    # TODO Plot barchart and image if it is a single image

# Call to main function to run the program
if __name__ == "__main__":
    main()