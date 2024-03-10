#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ImageClassifier/ic_input_args.py
#                                                                         
# PROGRAMMER: Craig Brennan
# DATE CREATED: 22/02/2024                                 
# REVISED DATE: 
# PURPOSE: Process the command line inputs arguments for the Image Classifier program. 
#          Used by train.py and predict.py. Makes use of the argparse library. If --checkpoint
#          is used then the specified model checkpoint will be loaded and used for training.

import argparse

# Parse input arguments for training program
def get_input_args_train():
    
    parser = argparse.ArgumentParser(description='Image Classifier - Training')

    # Positional input arguments - can appear at anywhere in the input command
    parser.add_argument('data_dir', help='Directory containing, training, validation and test images')

    # Arch and Checkpoint are mutually exclusive input arguments
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--arch', default='resnet34', choices=['vgg19', 'resnet34' , 'resnet152', 'alexnet'], help='CNN model to use')
    group.add_argument('--checkpoint', help='Path to model checkpoint file (.pth) for continued training')

    parser.add_argument('--save_dir', default='../checkpoints/', help='Directory to save checkpoints')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--learning_rate', default='0.001', type=float, help='Set learning rate')
    parser.add_argument('--hidden_units', default='1000', type=int, help='Number of hidden units in classifier layer')
    parser.add_argument('--epochs', default='10', type=int, help='Number of epochs for training')

    return parser.parse_args()

# Parse input arguments for prediction program
def get_input_args_predict():
    parser = argparse.ArgumentParser(description='Image Classifier - Prediction')

    # Positional input arguments - can appear at anywhere in the input command but must be in order
    parser.add_argument('input_image_path', help='Path to input image to be classified')
    parser.add_argument('checkpoint', help='Path to model checkpoint file (.pth)')

    # Optional input arguments
    parser.add_argument('--top_k', default=5, type=int, help='Number of top results to return')
    parser.add_argument('--category_names', default='../res/cat_to_name.json', help='Category numbers to names file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    return parser.parse_args()