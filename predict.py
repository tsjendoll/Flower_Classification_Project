#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                            
# PROGRAMMER: Jen Berenguel                                                   
# DATE CREATED: 03/23/2024                                  
# REVISED DATE:                                 
# PURPOSE:

# Imports functions for this program
import tkinter as tk
from tkinter import filedialog, messagebox
from utils import get_input_args, check_input_type, label_mapping, print_predictions, find_image_files
from classifier import load_model, predict
import csv
import pandas as pd

def main():
    # Get command line arguments using in_arg  
    in_args = get_input_args('predict')

    model, idx_class_mapping = load_model(in_args.model_path)

    input_type = check_input_type(in_args.input_data)

    # print(input_type)
    labels = label_mapping(in_args.labels)

    if input_type == 'folder':

        field_names = ['image_file', 'expected', 'prediction', 'probability', 'match']
        all_predictions = []

        image_files = find_image_files(in_args.input_data)

        for image_file in image_files:
                probs, classes = predict(image_file, model, labels, idx_class_mapping, 1)
                expected = labels[image_file.split('/')[-2]]
                all_predictions.append({
                    'image_file': image_file,
                    'expected': expected,
                    'prediction': classes[0],
                    'probability': round(float(probs[0]) * 100, 2), 
                    'match':  expected == classes[0]
                })

        df = pd.DataFrame(all_predictions)

        # Create a Tkinter root window
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.attributes('-topmost', True)  # Ensure the dialog box appears in front of other apps

        # Open the "Save As" dialog box
        file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                initialfile='predictions.csv',
                                                filetypes=[("CSV files", "*.csv"), ("All Files", "*.*")])
        
        if file_path:
            with open('predictions.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = field_names) 
                writer.writeheader() 
                writer.writerows(all_predictions) 
            print(f"Predictions saved to {file_path}")
        else:
            print("Save operation was cancelled.")
            
    elif input_type == 'single':
        probs, classes = predict(in_args.input_data, model, labels, idx_class_mapping, in_args.topk)
        print_predictions(probs, classes)
    else:
        raise Exception("Input type must be a single images or a folder containing images")

    # TODO Plot barchart and image if it is a single image

# Call to main function to run the program
if __name__ == "__main__":
    main()