#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ImageClassifier/ic_utility.py
#                                                                         
# PROGRAMMER: Craig Brennan
# DATE CREATED: 22/02/2024                                 
# REVISED DATE: 
# PURPOSE: Utility functions that belong to the Image Classifier program. These 
#          functions are used by predict.py to perform image processing and model
#          predictions.

import torch
import numpy as np

from PIL import Image

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)

    # Scale image to size with shortest dimension being 256 pixels while preserving aspect ratio
    max = 16000
    if im.width >= im.height:
        im.thumbnail((max, 256))
    else:
        im.thumbnail((256, max))

    # Crop 224, 224 pixels from center of image
    w_gap = (im.width - 224) / 2
    h_gap = (im.height - 224) / 2
    im = im.crop((w_gap, h_gap, w_gap + 224, h_gap + 224))

    # Convert to np array and normalize
    np_image = np.array(im) / 255.0
    np_image = (np_image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    return torch.Tensor(np_image.transpose((2, 0, 1)))

def predict(image_path, model, device, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image_tensor = process_image(image_path)

    # Add additional dimension (batch dimension) to tensor
    image_tensor = image_tensor[None, :, :, :]
    image_tensor = image_tensor.to(device)
    
    # Perform class predictions using the model
    model.eval()
    logps = model.forward(image_tensor)
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk)

    # Before returning, move to cpu, remove gradients, make tensor
    # a singular dimension and convert to python list
    probs = list(top_p[0].cpu().detach().numpy())
    class_idx = list(top_class[0].cpu().detach().numpy())
    return probs, class_idx