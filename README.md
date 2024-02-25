# AI Programming with Python Project - Image Classifier

Project code for Udacity's AI Programming with Python Nanodegree program. This program explores image classification with Pytorch.
A pretrained CNN model of specified type is loaded and then fine tuned using the input data set. The program uses a test set of images of flowers and can classify an input image as one of 102 flower categories.

There are 2 Python script files, train.py for training the model and predict.py for performing image classification predictions. The program has the ability to save model checkpoints into .pth files and load them for furture use. This repository includes one such checkpoint file that can be used for image predictions.

## Command Line Usage:

train.py is used to load a pretrained CNN model and fine tune it using the specified path contraining labelled training images.

```
usage: train.py [-h] data_dir [--save_dir SAVE_DIR] [--arch {vgg19,resnet34,alexnet}] [--gpu] [--learning_rate LEARNING_RATE]
                [--hidden_units HIDDEN_UNITS] [--epochs EPOCHS]

Image Classifier - Training

positional arguments:
  data_dir              Directory containing, training, validation and test images

options:
  -h, --help            show this help message and exit
  --save_dir SAVE_DIR   Directory to save checkpoints
  --arch {vgg19,resnet34,alexnet}
                        CNN model to use
  --gpu                 Use GPU for training
  --learning_rate LEARNING_RATE
                        Set learning rate
  --hidden_units HIDDEN_UNITS
                        Number of hidden units in classifier layer
  --epochs EPOCHS       Number of epochs for training
```

predict.py performs image predictions for the specified input image and displays the top number of category class names to the output.

```
usage: predict.py [-h] input_image_path checkpoint [--top_k TOP_K] [--category_names CATEGORY_NAMES] [--gpu]

Image Classifier - Prediction

positional arguments:
  input_image_path      Path to input image to be classified
  checkpoint            Path to model checkpoint file (.pth)

options:
  -h, --help            show this help message and exit
  --top_k TOP_K         Number of top results to return
  --category_names CATEGORY_NAMES
                        Category numbers to names file
  --gpu                 Use GPU for inference
```
