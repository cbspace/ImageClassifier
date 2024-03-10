#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ImageClassifier/train.py
#                                                                         
# PROGRAMMER: Craig Brennan
# DATE CREATED: 22/02/2024                                 
# REVISED DATE: 
# PURPOSE: Classifies flower images using a pre-trained CNN model from torchvision.
#          The classification layers of the model are replaced and the model is fine tuned
#          using the images provided. The model is then saved as a checkpoint for later use.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py [-h] <Directory With Images> [--save_dir <Save Dir>] 
#                      [--arch <{vgg19,resnet34,alexnet}>] [--gpu] 
#                      [--learning_rate <Learning Rate>] [--hidden_units <Hidden Units>] 
#                      [--epochs <Epochs>]
#   Example call:
#      python train.py ../flowers/ --save_dir ../checkpoints/ --arch alexnet --gpu --epochs 5

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, models, transforms
from torchvision.transforms import v2
import numpy as np

# Imports for Image Classifier python files
from ic_model import ICModel
from ic_input_args import get_input_args_train

# Parse input arguments
args = get_input_args_train()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = v2.Compose([v2.RandomRotation(30),
                               v2.RandomResizedCrop(224),
                               v2.RandomHorizontalFlip(0.5),
                               transforms.ToTensor(),
                               v2.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])

test_transforms = v2.Compose([v2.Resize(224),
                              v2.CenterCrop(224),
                              transforms.ToTensor(),
                              v2.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
train_loader = DataLoader(train_dataset, batch_size=320, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=320)
test_loader = DataLoader(valid_dataset, batch_size=320)

# Create Image Classifier model, train and validate
model = ICModel(args.gpu, '../res/cat_to_name.json')

# If a checpoint is specified then load it, otherwise create a new model
if args.checkpoint:
    model.load_checkpoint(args.checkpoint)
else:
    model.create_model(args.arch, args.hidden_units, train_dataset.class_to_idx, args.learning_rate)

# Train the model for number of specified epochs
model.train(train_loader, valid_loader, args.epochs)
model.validate(test_loader)

# Create path to save model
file_path = args.save_dir

if file_path[-1] != '/':
    file_path += '/'

# Save the checkpoint as <model_arch>.pth
file_path += f'{model.arch}_{model.hidden_units}_{model.epochs_elapsed}.pth'
model.save_checkpoint(file_path)