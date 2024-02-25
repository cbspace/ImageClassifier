#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ImageClassifier/ic_model.py
#                                                                         
# PROGRAMMER: Craig Brennan
# DATE CREATED: 22/02/2024                                 
# REVISED DATE: 
# PURPOSE: Image Classifier Model Class definition. Used by train.py and predict.py

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, models, transforms
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import alexnet, AlexNet_Weights

import numpy as np
import json

class ICModel:
    ''' Image Classifier Model Class
        Encapsulates a model and all required methods for creating, training,
        saving and loading a pretrained CNN model.
    '''
    def __init__(self, gpu, cat_to_name_file):
        ''' When creating an ICModel object, the pytorch device is selected and
            cat to name file is loaded.
        '''
        # Select device for model (CPU or GPU)
        self.device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')

        # Load category names file
        with open(cat_to_name_file, 'r') as f:
            self.cat_to_name = json.load(f)

    def create_model(self, arch, hidden_units, class_to_idx, learning_rate=0.001):
        ''' This function is called manually after intialising an ICModel object
            and is passed model parameters as inputs. The function is also called after
            loading a checkpoint.
        '''
        self.arch = arch
        self.hidden_units = hidden_units

        # Create a model of specified architecture
        # Freeze parameters and replace classifier layer
        # Note the different default name of classifier layer for the models
        if self.arch == 'vgg19':
            self.model = models.vgg19(weights=VGG19_Weights.DEFAULT)
            self.freeze_parameters()
            self.model.classifier = self.create_classifier_layer(25088)
            parameters = self.model.classifier.parameters()
        elif self.arch == 'resnet34':
            self.model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
            self.freeze_parameters()
            self.model.fc = self.create_classifier_layer(512)
            parameters = self.model.fc.parameters()
        elif self.arch == 'alexnet':
            self.model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
            self.freeze_parameters()
            self.model.classifier = self.create_classifier_layer(9216)
            parameters = self.model.classifier.parameters()

        # Store class_to_idx and move model to device                 
        self.class_to_idx = class_to_idx
        self.model.to(self.device) 

        # Create criterion and optimizer
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(parameters, lr=learning_rate)

    # Freeze model layers
    def freeze_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False

    # Method to create the model classifier layer
    def create_classifier_layer(self, hidden_layer_in_size): 
        return nn.Sequential(nn.Dropout(p=0.2),
                             nn.Linear(hidden_layer_in_size, self.hidden_units),
                             nn.ReLU(),
                             nn.Dropout(p=0.2),
                             nn.Linear(self.hidden_units,102),
                             nn.LogSoftmax(dim=1))

    # Load model checkpoint file and re-create the model
    def load_checkpoint(self, filepath):
        checkpoint_loaded = torch.load(filepath)
        self.create_model(checkpoint_loaded['arch'], 
                          checkpoint_loaded['hidden_units'],
                          checkpoint_loaded['class_to_idx'])
        self.model.load_state_dict(checkpoint_loaded['state_dict'])

    # Save the current model checkpoint into a .pth file
    def save_checkpoint(self, filepath):
        checkpoint = {'arch': self.arch,
                      'hidden_units': self.hidden_units,
                      'state_dict': self.model.state_dict(),
                      'class_to_idx': self.class_to_idx}

        torch.save(checkpoint, filepath)

    # Method to train the model
    def train(self, train_loader, valid_loader, epochs):
        for epoch in range(1, epochs+1):
            self.model.train()
            running_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                logps = self.model.forward(images)
                loss = self.criterion(logps, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Validation after each epoch
            else:
                self.model.eval()
                validation_loss = 0.0
                accuracy = 0
                
                for images, labels in valid_loader:
                    images, labels = images.to(self.device), labels.to(self.device)

                    logps = self.model.forward(images)
                    loss = self.criterion(logps, labels)
                    validation_loss += loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor))

                print('Epoch: {}/{}'.format(epoch, epochs),
                    'Training Loss: {:.5f}'.format(running_loss/len(train_loader)),
                    'Validation Loss: {:.5f}'.format(validation_loss/len(valid_loader)),
                    'Validation Accuracy: {:.3f}'.format(accuracy/len(valid_loader)))

    # Perform training validation
    def validate(self, test_loader):
        self.model.eval()
        test_loss = 0.0
        accuracy = 0

        for batch, (images, labels) in enumerate(test_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            logps =  self.model.forward(images)
            loss = self.criterion(logps, labels)
            test_loss += loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equality.type(torch.FloatTensor))

            print('Batch: {}/{}'.format(batch+1, len(test_loader)),
                  'Test Loss: {:.5f}'.format(test_loss/(batch+1)),
                  'Test Accuracy: {:.3f}'.format(accuracy/(batch+1)))

