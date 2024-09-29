import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Define the forward diffusion process
def forward_diffusion(x0, noise, t, T):
    """
    Applies the forward diffusion process.
    
    Args:
    - x0: Original image tensor (batch_size, channels, height, width)
    - noise: Gaussian noise tensor with the same shape as x0
    - t: Current time step in the diffusion process
    - T: Total number of time steps
    
    Returns:
    - xt: Noised image tensor at time step t
    """
    alpha = 1 - (t / T)  # Simple linear schedule
    return alpha * x0 + (1 - alpha) * noise

# Simple U-Net-like model for reverse process (denoising)
class SimpleUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleUNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in range [0, 1] for image generation
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Hyperparameters
T = 1000  # Number of time steps
batch_size = 32
learning_rate = 1e-4
epochs = 10

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='mnist_data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model, optimizer, loss
model = SimpleUNet(in_channels=1, out_channels=1)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Training loop
for epoch in range(epochs):
    model.train()
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        optimizer.zero_grad()
        
        # Apply forward diffusion (add noise)
        noise = torch.randn_like(data)
        t = torch.randint(0, T, (data.shape[0],))  # Random time steps
        xt = forward_diffusion(data, noise, t, T)
        
        # Denoising step: predict original image from noisy image
        reconstructed = model(xt)
        loss = criterion(reconstructed, data)
        
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# Generate new images by denoising random noise
model.eval()
with torch.no_grad():
    noise = torch.randn(batch_size, 1, 28, 28)
    generated_images = model(noise)

    # (Save or visualize generated_images as needed)
