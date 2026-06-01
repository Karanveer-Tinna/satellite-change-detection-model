import numpy as np
import cv2
from torch.utils.data import Dataset
import os
from glob import glob
from scipy.spatial.distance import cdist

# --- Class and Color Definitions ---
# IMPORTANT: The order of this list must match the output channels of your model
CLASS_NAMES = [
    "Unlabelled", "Vegetation", "Built-Up", "Informal Settlements",
    "Impervious Surfaces", "Barren", "Water"
]

# Using a list of tuples for RGB values to maintain order
CLASS_RGB_VALUES = [
    (0, 0, 0),        # 0 - Unlabelled
    (80, 140, 50),    # 1 - Vegetation
    (200, 200, 200),  # 2 - Built-Up
    (250, 235, 185),  # 3 - Informal Settlements
    (100, 100, 150),  # 4 - Impervious Surfaces
    (200, 160, 40),   # 5 - Barren
    (40, 120, 240)    # 6 - Water
]

# Convert the RGB palette to LAB color space for more accurate color distance calculation
palette_rgb = np.array(CLASS_RGB_VALUES, dtype=np.uint8).reshape(1, -1, 3)
palette_lab = cv2.cvtColor(palette_rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3)

def enhance_image_dip(img_bgr, alpha=1.0, beta=0.5, gamma=2.5, laplacian_ksize=5):
    return img_bgr

def rgb_mask_to_multiclass_robust(mask_rgb):
    """
    Converts an RGB mask to a 2D array of class indices by finding the closest color
    in the predefined palette. This is more robust to compression artifacts.
    """
    h, w, _ = mask_rgb.shape
    # Convert the input mask to LAB color space
    mask_lab = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2LAB)
    
    # Reshape the mask to be a list of pixels
    pixels = mask_lab.reshape(-1, 3)
    
    # Calculate the distance from each pixel to each color in the palette
    distances = cdist(pixels, palette_lab)
    
    # Find the index of the closest color for each pixel
    class_indices = np.argmin(distances, axis=1)
    
    # Reshape the indices back to the original image dimensions
    semantic_map = class_indices.reshape(h, w)
    
    return semantic_map.astype(np.uint8)

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.images = sorted(glob(os.path.join(image_dir, "*.tif")))
        self.masks = sorted(glob(os.path.join(mask_dir, "*.png")))
        self.transform = transform
        
        assert len(self.images) == len(self.masks), "Mismatch between number of images and masks."

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_bgr = cv2.imread(self.images[idx])
        mask_bgr = cv2.imread(self.masks[idx])
        
        # The enhancement function no longer performs B-channel equalization
        img_enhanced_bgr = enhance_image_dip(img_bgr)
        img_enhanced_rgb = cv2.cvtColor(img_enhanced_bgr, cv2.COLOR_BGR2RGB)
        
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
        mask = rgb_mask_to_multiclass_robust(mask_rgb)
        
        if self.transform:
            augmented = self.transform(image=img_enhanced_rgb, mask=mask)
            img, mask = augmented['image'], augmented['mask']
            
        return img, mask.long()