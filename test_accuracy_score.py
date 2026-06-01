import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from datetime import datetime

# --- MOCK UTILITIES (To make the script runnable) ---

# 1. Mock Model Architecture (Simulates the DeepLabV3+ with a single output channel for binary seg)
class PlaceholderSegmentationModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, patch_size=128):
        super().__init__()
        # A simple structure to simulate the required output shape [B, 1, H, W]
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.final_conv(x) # Returns raw logits [B, 1, H, W]


# 2. Mock Dataset
class MockTestDataset(Dataset):
    def __init__(self, num_samples=10, patch_size=128, channels=3):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.channels = channels
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Image: [C, H, W]
        image = torch.randn(self.channels, self.patch_size, self.patch_size).float()
        # Mask: [C, H, W] (Binary 0 or 1)
        # We generate a random binary mask for simulation
        mask = torch.randint(0, 2, (1, self.patch_size, self.patch_size)).float()
        return image, mask

# 3. Mock DataLoader Function
def get_test_dataloader(batch_size, patch_size, num_samples):
    dataset = MockTestDataset(num_samples, patch_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 4. Model File Preparation (Required to prevent FileNotFoundError)
def _prepare_model_file(model, path):
    """Saves a dummy model state to ensure torch.load does not fail."""
    if not os.path.exists(path):
        print(f"INFO: Creating mock model file at {path} for load test.")
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'best_iou': 0.88 
        }
        torch.save(checkpoint, path)

# --- ACCURACY CHECK FUNCTIONS ---

# 1. Define Model/Weights paths
MODEL_PATH = 'best_model.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_best_model():
    """Loads the model architecture and weights from the checkpoint."""
    # NOTE: Assuming the model is for binary segmentation based on the original code
    model = PlaceholderSegmentationModel(out_channels=1).to(DEVICE) 
    
    # Run helper function to ensure a file exists before trying to load it
    _prepare_model_file(model, MODEL_PATH) 

    try:
        # Load the checkpoint dictionary
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        # Load the state dict into the model
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model successfully loaded from {MODEL_PATH}. Best IoU from training: {checkpoint['best_iou']:.4f}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
        
    return model

def check_accuracy(model, test_loader):
    """
    Checks the accuracy and Mean IoU of the model on the test dataloader 
    using standard (single patch/tile) inference.
    """
    print(f"\n--- Checking Standard Accuracy on Test Dataloader ({len(test_loader.dataset)} samples) ---")
    model.eval()
    total_correct = 0
    total_pixels = 0
    
    # Variables for IoU calculation
    total_intersection = 0
    total_union = 0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE) # True masks (ground truth) [B, 1, H, W]

            # Standard Forward Pass (outputs are raw logits [B, 1, H, W])
            outputs = model(images) 
            
            # Convert logits to probabilities (Sigmoid for binary segmentation)
            probs = torch.sigmoid(outputs)
            
            # Convert probabilities to predicted mask (binary: 0 or 1)
            predicted_mask = (probs > 0.5).float() # [B, 1, H, W]

            # --- Pixel Accuracy Calculation ---
            correct = (predicted_mask == masks).sum().item()
            total_correct += correct
            total_pixels += masks.numel() # Total number of pixels in the batch
            
            # --- IoU Calculation (Simplified/Binary) ---
            intersection = (predicted_mask * masks).sum().item()
            union = (predicted_mask + masks).clamp(0, 1).sum().item() # union = A + B - (A * B) is approximated by clamp(A+B)
            total_intersection += intersection
            total_union += union
            
    accuracy = total_correct / total_pixels
    mean_iou = total_intersection / (total_union + 1e-6) # Add epsilon for stability
    
    print("\n--- Final Test Metrics ---")
    print(f"Overall Pixel Accuracy: {accuracy:.4f}")
    print(f"Mean IoU (Jaccard Index): {mean_iou:.4f}")

def main():
    
    # 1. Load the model
    model = load_best_model()
    if model is None:
        return

    # 2. Prepare DataLoaders
    # Using mock function to get a test set of 10 samples of 128x128 patches
    test_loader = get_test_dataloader(batch_size=4, patch_size=128, num_samples=10)
    
    # 3. Check accuracy using standard tile inference
    check_accuracy(model, test_loader)
    
    # 4. Clean up mock file
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

if __name__ == "__main__":
    main()