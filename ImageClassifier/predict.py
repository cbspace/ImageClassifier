#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ImageClassifier/predict.py
#                                                                         
# PROGRAMMER: Craig Brennan
# DATE CREATED: 22/02/2024                                 
# REVISED DATE: 
# PURPOSE: Classifies flower images using a pre-trained CNN model from torchvision.
#          This file takes a model checkpoint as an input and uses it to perform
#          inferences (predictions) to classify the image into flower types. 
#
# Use argparse Expected Call with <> indicating expected user input:
#      python predict.py [-h] <Input Image Path> <Model Checkpoint Path>
#                        [--top_k <Number of Top Results>] 
#                        [--category_names <Category Name File Path>] [--gpu]
#   Example call:
#      python predict.py ../sunflower_224.jpg ../checkpoints/checkpoint2.pth --gpu --top_k 5

# Imports for Image Classifier python files
from ic_model import ICModel
from ic_input_args import get_input_args_predict
from ic_utility import process_image, predict

# Process input arguments
args = get_input_args_predict()

# Create model instance and load model from checkpoint
model = ICModel(args.gpu, args.category_names)
model.load_checkpoint(args.checkpoint)

# Generate predictions
probs, class_idx = predict(args.input_image_path, model.model, model.device, args.top_k)

# Create index to class dictionary
idx_to_class = {v: k for k,v in model.class_to_idx.items()}

# Get class numbers
class_numbers = [idx_to_class[c] for c in class_idx]

# Get class names
class_names = [model.cat_to_name.get(c) for c in class_numbers]

# Print out top class names and probabilities
for i, (class_name, probability) in enumerate(zip(class_names, probs), 1):
    print('{:4s} {:30s} {:.6f}'.format(str(i), class_name.title(), probability))