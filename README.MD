# Autoencoder Model for Pong Game Point Recognition
![DotAutoencoder Architecture](https://github.com/barron9/encoder_decoder_for_pong_v5/blob/master/dot_gen.png)
```python
DotAutoencoder(

 (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

 (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

 (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

 (fc1): Linear(in_features=16384, out_features=128, bias=True)

 (fc2): Linear(in_features=128, out_features=16384, bias=True)

 (deconv1): ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))

 (deconv2): ConvTranspose2d(32, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))

)
```
### Key Components

- **Convolutional layers**: Extract features from the input image.
- **Max pooling**: Reduces spatial dimensions to control the complexity.
- **Fully connected layers**: Compress the learned features and help in reconstruction.
- **Transposed convolution layers (deconvolution)**: Upsample the data and reconstruct the original input.

### Purpose

This model is to handle the point problem in the Pong game that makes training time increase only if used linear layers and pgradient. It takes an image of the game state, processes it through the encoder, and then reconstructs the image and the x,y coordinates through the decoder.
