import torch
import torch.nn as nn
import torch.optim as optim

class Settings:
    def __init__(self):
        self.patience = 10
        self.epochs = 100
        self.lr_factor = 0.1
        self.batch_size = 128

class Autoencoder1(nn.Module):
    """Autoencoder with a hidden representation of 32 elements"""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.encoder = nn.Sequential(
            # Using padding='same' ensures 64 -> 64
            nn.Conv2d(in_channels, 16, kernel_size=6, stride=1, padding='same'),
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
            # Swapped to Conv2d with padding='same' to lock the output to exactly 64x64
            nn.Conv2d(16, out_channels, kernel_size=6, stride=1, padding='same')
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
            nn.Conv2d(in_channels, 16, kernel_size=6, stride=1, padding='same'),
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
            nn.Conv2d(16, out_channels, kernel_size=6, stride=1, padding='same')
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
            nn.Conv2d(in_channels, 16, kernel_size=6, stride=1, padding='same'),
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
            nn.Conv2d(16, out_channels, kernel_size=6, stride=1, padding='same')
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return torch.tanh(self.decoder_net(x))

    def forward(self, x):
        enc = self.encode(x)
        dec = self.decode(enc)
        return dec, enc

def create_model(model_class, in_channels=1, out_channels=1, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(in_channels=in_channels, out_channels=out_channels).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer
