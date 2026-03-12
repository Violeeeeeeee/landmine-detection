"""
Translated to PyTorch
@Author: Francesco Picetti - francesco.picetti@polimi.it
"""

import os
import GPUtil
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Use the GPU with the lowest memory usage
# We set the environment variable BEFORE initializing CUDA in PyTorch
DEVICE_ID = str(GPUtil.getFirstAvailable(order='memory')[0])
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID
print('GPU selected:', DEVICE_ID)

# Set the device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Settings:
    def __init__(self):
        self.patience = 10
        self.epochs = 100
        self.lr_factor = 0.1
        self.batch_size = 128


# Note: In PyTorch, we combine the grayscale (1-channel) and RGB (3-channel "3D")
# models by passing `out_channels` as a parameter.
# Auto1 -> out_channels=1 | Auto3D1 -> out_channels=3

class Autoencoder1(nn.Module):
    """Autoencoder with a hidden representation of 32 elements"""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Note: Symmetric padding is used here (kernel_size // 2).
        # Keras 'same' padding can be slightly asymmetric for even kernel sizes.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=6, stride=1, padding=2),
            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 8, kernel_size=1, stride=2, padding=0)
        )

        self.decoder_net = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2, padding=0),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ConvTranspose2d(16, out_channels, kernel_size=6, stride=1, padding=2)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return torch.tanh(self.decoder_net(x))

    def forward(self, x):
        enc = self.encode(x)
        dec = self.decode(enc)
        return dec, enc


class Autoencoder2(nn.Module):
    """Autoencoder with a hidden representation of 16 elements"""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=6, stride=1, padding=2),
            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 16, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 16, kernel_size=1, stride=2, padding=0)
        )

        self.decoder_net = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2, padding=0),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2, padding=0),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ConvTranspose2d(16, out_channels, kernel_size=6, stride=1, padding=2)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return torch.tanh(self.decoder_net(x))

    def forward(self, x):
        enc = self.encode(x)
        dec = self.decode(enc)
        return dec, enc


class Autoencoder3(nn.Module):
    """Autoencoder with a hidden representation of 64 elements"""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=6, stride=1, padding=2),
            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 16, kernel_size=2, stride=2, padding=0)
        )

        self.decoder_net = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2, padding=0),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ConvTranspose2d(16, out_channels, kernel_size=6, stride=1, padding=2)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return torch.tanh(self.decoder_net(x))

    def forward(self, x):
        enc = self.encode(x)
        dec = self.decode(enc)
        return dec, enc


# --- Helper Function to mimic Keras instantiation and compilation ---

def create_model(model_class, in_channels=1, out_channels=1, lr=0.001):
    """
    Instantiates the model, moves it to GPU, and sets up the loss and optimizer.
    Replaces the AutoX() functions from the Keras script.
    """
    model = model_class(in_channels=in_channels, out_channels=out_channels).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer

# Example usage mapping to your old functions:
# Auto1()    -> model, criterion, opt = create_model(Autoencoder1, out_channels=1)
# Auto3D1()  -> model, criterion, opt = create_model(Autoencoder1, out_channels=3)
