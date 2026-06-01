import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Define standard normalization parameters (placeholders)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class SimulatedTestDataset(Dataset):
    """Generates synthetic images and masks for testing."""
    def __init__(self, num_samples=10, patch_size=128):
        self.num_samples = num_samples
        self.patch_size = patch_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Simulate a single RGB patch (C, H, W)
        image = np.random.rand(self.patch_size, self.patch_size, 3).astype(np.float32) * 255
        # Simulate a binary mask (H, W)
        mask = (np.random.rand(self.patch_size, self.patch_size) > 0.8).astype(np.float32)
        
        # Simple placeholder transformation: Convert to Tensor and normalize (HWC -> CWH)
        image = torch.from_numpy(image).permute(2, 0, 1) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0) # Add channel dim (1, H, W)
        
        # Apply normalization
        for t, m, s in zip(image, MEAN, STD):
            t.sub_(m).div_(s)
            
        return image, mask

def get_test_dataloader(batch_size=4, patch_size=128, num_samples=10):
    dataset = SimulatedTestDataset(num_samples=num_samples, patch_size=patch_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def get_full_resolution_image(H=1024, W=1024):
    """Simulate a large image for sliding window inference."""
    # Simulate a full-resolution RGB image (H, W, C)
    img_np = np.random.rand(H, W, 3).astype(np.uint8) * 255
    return img_np