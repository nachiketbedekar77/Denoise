import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import SpectrogramDataset

class UNetDenoiseMask(nn.Module):
    def __init__(self):
        super(UNetDenoiseMask, self).__init__()
        # Encoder
        self.e1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2))
        self.p1 = nn.MaxPool2d(2, 2)
        
        self.e2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.p2 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.b = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        
        # Mask Output (Values between 0 and 1)
        self.out_mask = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, 1, 256, 256)
        x1 = self.e1(x)
        x2 = self.e2(self.p1(x1))
        
        bot = self.b(self.p2(x2))
        
        u1 = self.up1(bot)
        u1 = torch.cat([u1, x2], dim=1)
        d1 = self.d1(u1)
        
        u2 = self.up2(d1)
        u2 = torch.cat([u2, x1], dim=1)
        d2 = self.d2(u2)
        
        mask = self.out_mask(d2)
        return mask

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SpectrogramDataset("dataset/spectrograms_noisy", "dataset/spectrograms_clean")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=True)

    model = UNetDenoiseMask().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    # Compressed Power Loss to ensure clarity in high frequencies
    def compressed_loss(pred_clean, target_clean):
        # Apply 0.5 power compression (Square root)
        return nn.MSELoss()(torch.pow(pred_clean + 1e-8, 0.5), torch.pow(target_clean + 1e-8, 0.5))

    epochs = 30
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for noisy_batch, clean_batch in dataloader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)

            optimizer.zero_grad()
            
            # Predict Mask
            mask = model(noisy_batch)
            
            # Apply Mask to Noisy Input
            estimated_clean = noisy_batch * mask
            
            loss = compressed_loss(estimated_clean, clean_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * noisy_batch.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.5f}")

    torch.save(model.state_dict(), "denoise_model.pth")
    print("Training Complete! Saved as 'denoise_model.pth'")

if __name__ == "__main__":
    train_model()