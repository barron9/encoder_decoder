# DotAutoencoder Model for Pong Game Dot Recognition
![DotAutoencoder Architecture](https://github.com/barron9/encoder_decoder_for_pong_v5/blob/master/dot_gen.png)

### Key Components

- **Convolutional layers**: Extract features from the input image.
- **Max pooling**: Reduces spatial dimensions to control the complexity.
- **Fully connected layers**: Compress the learned features and help in reconstruction.
- **Transposed convolution layers (deconvolution)**: Upsample the data and reconstruct the original input.

### Purpose

This model is tailored to handle the dot recognition problem in the Pong game. It takes an image of the game state along with the `x` and `y` coordinates of the dot, processes it through the encoder, and then reconstructs the image and the coordinates through the decoder. This can be useful for applications such as tracking the movement of the Pong ball or learning more about the game dynamics from visual data.
