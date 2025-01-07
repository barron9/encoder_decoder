# DotAutoencoder Model for Pong Game Dot Recognition
![DotAutoencoder Architecture](https://github.com/barron9/encoder_decoder_for_pong_v5/blob/master/dot_gen.png)

### Key Components

- **Convolutional layers**: Extract features from the input image.
- **Max pooling**: Reduces spatial dimensions to control the complexity.
- **Fully connected layers**: Compress the learned features and help in reconstruction.
- **Transposed convolution layers (deconvolution)**: Upsample the data and reconstruct the original input.

### Purpose

This model is to handle the point problem in the Pong game that makes training time increase only if used linear layers and pgradient. It takes an image of the game state, processes it through the encoder, and then reconstructs the image and the x,y coordinates through the decoder.
