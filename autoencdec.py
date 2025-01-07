import numpy as np
import matplotlib.pyplot as plt

def generate_random_dot_image(image_size=(64, 64), dot_value=255):
    """
    Generates a random 64x64 image with a single random dot.
    
    Args:
    - image_size: Tuple, dimensions of the image (height, width).
    - dot_value: The value (e.g., 255 for white) to set at the dot's position.
    
    Returns:
    - A 2D numpy array representing the image with the dot.
    """
    # Create a blank black image (all zeros)
    image = np.zeros(image_size, dtype=np.uint8)
    
    # Generate a random position (x, y) for the dot
    x = np.random.randint(0, image_size[0])  # Random x position (0 to 63)
    y = np.random.randint(0, image_size[1])  # Random y position (0 to 63)
    
    # Set the pixel at the random position to the dot value
    image[x, y] = dot_value
    
    return image

# # Generate a single random dot image
# random_dot_image = generate_random_dot_image(image_size=(80, 80))

# # Display the generated image (optional)
# plt.imshow(random_dot_image, cmap='gray')
# plt.title('Random Dot Image')
# plt.axis('off')  # Hide axes
# plt.show()

# import torch
# import torch.nn as nn

# class DotLocationCNN(nn.Module):
#     def __init__(self):
#         super(DotLocationCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Conv layer with 32 filters
#         self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Second Conv layer with 64 filters
#         self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Fully connected layer
#         self.fc2 = nn.Linear(128, 2)  # Output layer for (x, y) coordinates

#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))  # First convolution + pooling
#         x = self.pool(torch.relu(self.conv2(x)))  # Second convolution + pooling
#         x = x.view(-1, 64 * 16 * 16)  # Flatten for fully connected layers
#         x = torch.relu(self.fc1(x))  # First fully connected layer
#         x = self.fc2(x)  # Output layer for (x, y)
#         return x


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

# Define an Autoencoder model
class DotAutoencoder(nn.Module):
    def __init__(self):
        super(DotAutoencoder, self).__init__()
        
        # Encoder: Learn a compact representation of the image
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Conv layer 1
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Conv layer 2
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Fully connected layer
        
        # Decoder: Reconstruct the image
        self.fc2 = nn.Linear(128, 64 * 16 * 16)  # Fully connected layer
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  # Deconv layer 1
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)  # Deconv layer 2
        
    def forward(self, x):
        # Encoder
        x = self.pool(torch.relu(self.conv1(x)))  # First conv + pool
        x = self.pool(torch.relu(self.conv2(x)))  # Second conv + pool
        x = x.view(-1, 64 * 16 * 16)  # Flatten the feature map
        x = torch.relu(self.fc1(x))  # First fully connected
        
        # Decoder (Reconstruct the image)
        x = torch.relu(self.fc2(x))  # First fully connected layer for reconstruction
        x = x.view(-1, 64, 16, 16)  # Reshape to feature map
        x = torch.relu(self.deconv1(x))  # First deconv layer
        x = self.deconv2(x)  # Output image (reconstructed)
        
        return x

# Define the dataset to generate random dot images (same as before)
class RandomDotDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=(64, 64)):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create a black image (64x64)
        image = np.zeros(self.image_size, dtype=np.uint8)
        
        # Random position for the dot (x, y coordinates)
        x = np.random.randint(0, self.image_size[0])  # Random x position
        y = np.random.randint(0, self.image_size[1])  # Random y position
        
        # Place the dot (white dot in a black image)
        image[x, y] = 255  # Dot is represented by a value of 255 (white)
        
        # Convert image to a tensor (add channel dimension)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return image  # Return only the image (no label)

# Step 1: Initialize the dataset, model, loss function, and optimizer
train_dataset = RandomDotDataset(num_samples=10000)  # 10,000 samples
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = DotAutoencoder()
criterion = nn.MSELoss()  # Mean Squared Error for reconstruction
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 2: Train the autoencoder
num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    for images in train_loader:
        optimizer.zero_grad()  # Zero gradients before each step
        outputs = model(images)  # Forward pass (reconstruct image)
        loss = criterion(outputs, images)  # Compute loss (reconstruction error)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Step 3: Evaluate the model
with torch.no_grad():
    # Generate a random test image
    test_image = train_dataset[0]
    test_image = test_image.unsqueeze(0)  # Add batch dimension
    
    # Get the reconstructed image from the model
    model.eval()  # Set model to evaluation mode
    reconstructed_image = model(test_image)

    # Visualize the test image and its reconstruction
    plt.subplot(1, 2, 1)
    plt.imshow(test_image.squeeze(), cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image.squeeze().numpy(), cmap='gray')
    plt.title('Reconstructed Image')

    plt.show()
