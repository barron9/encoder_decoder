import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
class RandomDotImageDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=(64, 64), dot_value=255, transform=None):
        """
        Custom dataset to generate random images with a dot at a random location.
        
        Args:
        - num_samples: Number of samples to generate for the dataset.
        - image_size: Tuple, dimensions of the image (height, width).
        - dot_value: The value to set at the dot's position.
        - transform: Optional transform to apply to the images.
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.dot_value = dot_value
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def generate_random_dot_image(self):
        """
        Generates a random image with a dot at a random position.
        """
        image = np.zeros(self.image_size, dtype=np.uint8)  # Create a blank black image (zeros)
        
        # Generate a random (x, y) position for the dot
        x = np.random.randint(0, self.image_size[0])
        y = np.random.randint(0, self.image_size[1])
        
        # Set the dot in the image
        image[x, y] = self.dot_value
        
        return image, x, y  # Return image and the coordinates of the dot

    def __getitem__(self, idx):
        # Generate a random image and its (x, y) position
        image, x, y = self.generate_random_dot_image()
        
        # Convert the image to a torch tensor
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add a channel dimension (1, H, W)
        
        # Normalize x, y to be in range [0, 1] by dividing by image size
        x = x / self.image_size[0]
        y = y / self.image_size[1]
        
        # Convert x and y to tensor
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1,)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)  # (1,)
        
        if self.transform:
            image = self.transform(image)  # Apply any transformations
        
        return image, x, y


# Create the full dataset with 1000 random samples
dataset = RandomDotImageDataset(num_samples=1000)

# Optionally, you can split the dataset into training and testing sets (e.g., 80% training, 20% testing)
train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for testing

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create the DataLoader for training with batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create the DataLoader for testing with batching, but no shuffling
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class DotAutoencoder(nn.Module):
    def __init__(self):
        super(DotAutoencoder, self).__init__()
        
        # Encoder: Learn a compact representation of the (image, x, y)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Conv layer 1 (input has 3 channels now)
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Conv layer 2
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Fully connected layer
        
        # Decoder: Reconstruct the (image, x, y)
        self.fc2 = nn.Linear(128, 64 * 16 * 16)  # Fully connected layer
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  # Deconv layer 1
        self.deconv2 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)  # Deconv layer 2 (output 3 channels for image, x, y)
        
    def forward(self, image, x, y):
        # Concatenate image, x, and y along the channel dimension (now input has 3 channels)
        # Expand coordinates to match the spatial dimensions of the image (i.e., (32, 1, 64, 64))
        expanded_coordinatesx = x.unsqueeze(2).unsqueeze(3)  # Shape becomes (32, 1, 1, 1)
        expanded_coordinatesx = expanded_coordinatesx.expand(-1, -1, 64, 64)  # Now it has shape (32, 1, 64, 64)
        expanded_coordinatesy = y.unsqueeze(2).unsqueeze(3)  # Shape becomes (32, 1, 1, 1)
        expanded_coordinatesy = expanded_coordinatesy.expand(-1, -1, 64, 64)  # Now it has shape (32, 1, 64, 64)

        input_data = torch.cat((image, expanded_coordinatesx, expanded_coordinatesy), dim=1)  # Concatenate along the channel dimension
        
        # Encoder
        x = self.pool(torch.relu(self.conv1(input_data)))  # First conv + pool
        x = self.pool(torch.relu(self.conv2(x)))  # Second conv + pool
        x = x.view(-1, 64 * 16 * 16)  # Flatten the feature map
        x = torch.relu(self.fc1(x))  # First fully connected
        
        # Decoder (Reconstruct the image, x, and y)
        x = torch.relu(self.fc2(x))  # First fully connected layer for reconstruction
        x = x.view(-1, 64, 16, 16)  # Reshape to feature map
        x = torch.relu(self.deconv1(x))  # First deconv layer
        x = self.deconv2(x)  # Output image (reconstructed image, x, y)
        
        return x


import torch.optim as optim
import torch.nn.functional as F

# Create the model
model = DotAutoencoder()
print(model)
# Loss function and optimizer
criterion = nn.MSELoss()  # Use MSE Loss for reconstruction
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    for i, (image, x, y) in enumerate(train_loader):
        optimizer.zero_grad()  # Zero out previous gradients
        
        # Forward pass
        output = model(image, x, y)  # Pass the image, x, and y
        expanded_coordinatesx = x.unsqueeze(2).unsqueeze(3)  # Shape becomes (32, 1, 1, 1)
        expanded_coordinatesx = expanded_coordinatesx.expand(-1, -1, 64, 64)  # Now it has shape (32, 1, 64, 64)
        expanded_coordinatesy = y.unsqueeze(2).unsqueeze(3)  # Shape becomes (32, 1, 1, 1)
        expanded_coordinatesy = expanded_coordinatesy.expand(-1, -1, 64, 64)  # Now it has shape (32, 1, 64, 64)

        # Target is the concatenation of image, x, and y (same as input)
        target = torch.cat((image, expanded_coordinatesx, expanded_coordinatesy), dim=1)  # Concatenate along channel dimension
        
        # Compute loss between the reconstructed output and the original pair (image, x, y)
        loss = criterion(output, target)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        #if (i + 1) % 1 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
        running_loss = 0.0
            
import matplotlib.pyplot as plt

# After training, you can evaluate the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for i, (image, x, y) in enumerate(test_loader):
        output = model(image, x, y)  # Forward pass with image, x, y
        
        # Visualize the first few reconstructed images (image, x, y)
        if i == 0:
            fig, ax = plt.subplots(2, 8, figsize=(12, 4))
            for j in range(8):
                # Show original (image, x, y) and reconstructed output
                ax[0, j].set_title(f"({x[j].item()*64},{y[j].item()*64})", fontsize=6)
                ax[0, j].imshow(image[j].squeeze(), cmap='gray')  # Show original image
                ax[0, j].axis('off')
                ax[1, j].set_title(f"")
                ax[1, j].imshow(output[j].squeeze()[0, :, :], cmap='gray')  # Show reconstructed output
                ax[1, j].axis('off')
            plt.show()
        break  # Display just the first batch
