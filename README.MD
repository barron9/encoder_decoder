# DotAutoencoder Model for Pong Game Dot Recognition
![DotAutoencoder Architecture](https://github.com/barron9/encoder_decoder_for_pong_v5/blob/master/dot_gen.png)
## Introduction

The **DotAutoencoder** model is a neural network designed for recognizing and processing dots in the context of a Pong game. It is an autoencoder architecture consisting of an encoder and decoder that work together to both compress and reconstruct the input data, which includes images and associated coordinates (`x`, `y`) of the dot in the Pong environment.

### Model Architecture

The model is structured as follows:

1. **Encoder**: The encoder extracts important features from the input data (which consists of the game image and dot coordinates). It uses convolutional layers to capture spatial patterns and fully connected layers to compress the learned features into a compact representation.
   
2. **Decoder**: The decoder aims to reconstruct the original input, including the image and coordinates, from the compressed feature representation. It uses transposed convolutional layers (deconvolution) to generate an output that closely matches the original input.

### Key Components

- **Convolutional layers**: Extract features from the input image.
- **Max pooling**: Reduces spatial dimensions to control the complexity.
- **Fully connected layers**: Compress the learned features and help in reconstruction.
- **Transposed convolution layers (deconvolution)**: Upsample the data and reconstruct the original input.

### Purpose

This model is tailored to handle the dot recognition problem in the Pong game. It takes an image of the game state along with the `x` and `y` coordinates of the dot, processes it through the encoder, and then reconstructs the image and the coordinates through the decoder. This can be useful for applications such as tracking the movement of the Pong ball or learning more about the game dynamics from visual data.
