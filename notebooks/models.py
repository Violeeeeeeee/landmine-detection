import torch
import torch.nn as nn
import torch.optim as optim

class AutoencoderA1(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=6, stride=1, padding=2), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=2, stride=2, padding=0), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2, padding=0, output_padding=0), nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1, output_padding=0), nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2, output_padding=0), nn.ReLU(),
            nn.ConvTranspose2d(16, out_channels, kernel_size=6, stride=1, padding=2, output_padding=0)
        )

    def forward(self, x):
        z1 = self.encoder(x)
        x_hat = torch.tanh(self.decoder(z1))
        z2 = self.encoder(x_hat)
        return x_hat, z1, z2
    
class AutoencoderA2(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=6, stride=1, padding=2), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1, stride=2, padding=0), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=1, stride=2, padding=0, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1, output_padding=0), nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2, output_padding=0), nn.ReLU(),
            nn.ConvTranspose2d(16, out_channels, kernel_size=6, stride=1, padding=2, output_padding=0)
        )

    def forward(self, x):
        z1 = self.encoder(x)
        x_hat = torch.tanh(self.decoder(z1))
        z2 = self.encoder(x_hat)
        return x_hat, z1, z2
    
class AutoencoderA3(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # 5 шарів (як в A1). Стиснення x64 досягається виключно за рахунок 4 каналів у кінці.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=6, stride=1, padding=2), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=2, stride=2, padding=0), nn.ReLU() 
        )
        # Декодер дзеркально повторює ту саму математику паддінгів, що й в A1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, kernel_size=2, stride=2, padding=0, output_padding=0), nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1, output_padding=0), nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2, output_padding=0), nn.ReLU(),
            nn.ConvTranspose2d(16, out_channels, kernel_size=6, stride=1, padding=2, output_padding=0)
        )

    def forward(self, x):
        z1 = self.encoder(x)
        x_hat = torch.tanh(self.decoder(z1))
        z2 = self.encoder(x_hat)
        return x_hat, z1, z2
    
def train_autoencoder(model, train_loader, epochs=50, lr=0.001, device=None):
    """
    Універсальна функція для тренування будь-якого з наших автоенкодерів.
    Повертає натреновану модель та історію втрат.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_mse = nn.MSELoss()
    
    print(f"\n=== ПОЧАТОК ТРЕНУВАННЯ: {model.__class__.__name__} ===")
    train_loss_history = []
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            
            # Нам потрібен тільки x_hat для розрахунку Loss
            x_hat, _, _ = model(batch_x)
            loss = criterion_mse(x_hat, batch_x)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        train_loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Епоха [{epoch+1:02d}/{epochs}], Loss: {avg_loss:.6f}")
            
    return model, train_loss_history