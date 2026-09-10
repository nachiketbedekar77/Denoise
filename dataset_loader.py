import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class SpectrogramDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, fixed_width=256):
        self.noisy_dir = Path(noisy_dir)
        self.clean_dir = Path(clean_dir)
        self.fixed_width = fixed_width
        
        self.noisy_files = sorted(list(self.noisy_dir.glob("*.npy")))
        self.clean_files = sorted(list(self.clean_dir.glob("*.npy")))

        assert len(self.noisy_files) == len(self.clean_files), "Mismatch in number of noisy and clean files!"

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_spec = np.load(self.noisy_files[idx])
        clean_spec = np.load(self.clean_files[idx])

        # --- MATH FIX: Log Scaling so model can see the patterns ---
        noisy_spec = np.log1p(noisy_spec)
        clean_spec = np.log1p(clean_spec)

        # Slice frequency bins (257 -> 256) for U-Net pooling compatibility
        noisy_spec = noisy_spec[:256, :]
        clean_spec = clean_spec[:256, :]

        # Ensure both match in width/time frames
        min_w = min(noisy_spec.shape[1], clean_spec.shape[1])
        noisy_spec = noisy_spec[:, :min_w]
        clean_spec = clean_spec[:, :min_w]

        # Pad or Crop to fixed_width (256 frames)
        current_width = noisy_spec.shape[1]
        if current_width < self.fixed_width:
            pad_width = self.fixed_width - current_width
            noisy_spec = np.pad(noisy_spec, ((0, 0), (0, pad_width)), mode='constant')
            clean_spec = np.pad(clean_spec, ((0, 0), (0, pad_width)), mode='constant')
        else:
            noisy_spec = noisy_spec[:, :self.fixed_width]
            clean_spec = clean_spec[:, :self.fixed_width]

        # Convert to PyTorch Tensors (Shape: [1, 256, 256])
        noisy_tensor = torch.from_numpy(noisy_spec).float().unsqueeze(0)
        clean_tensor = torch.from_numpy(clean_spec).float().unsqueeze(0)

        return noisy_tensor, clean_tensor

if __name__ == "__main__":
    dataset = SpectrogramDataset("dataset/spectrograms_noisy", "dataset/spectrograms_clean")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(f"Dataset ready with {len(dataset)} samples.")