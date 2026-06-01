import numpy as np
import cv2
from torch.utils.data import Dataset
import os
from glob import glob
from scipy.spatial.distance import cdist
import random

# --- Class and Color Definitions ---
CLASS_NAMES = [
    "Unlabelled", "Vegetation", "Built-Up", "Informal Settlements",
    "Impervious Surfaces", "Barren", "Water"
]

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

def enhance_image_dip(img_bgr):
    """Applies CLAHE enhancement in the LAB color space."""
    blurred_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
    lab = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    final_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return final_bgr

def rgb_mask_to_multiclass_robust(mask_rgb):
    """
    Converts an RGB mask to a 2D array of class indices by finding the closest color
    in the predefined palette using LAB color space.
    """
    h, w, _ = mask_rgb.shape
    mask_lab = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2LAB)
    pixels = mask_lab.reshape(-1, 3)
    distances = cdist(pixels, palette_lab)
    class_indices = np.argmin(distances, axis=1)
    semantic_map = class_indices.reshape(h, w)
    return semantic_map.astype(np.uint8)

class SegmentationDataset(Dataset):
    """
    Dataset for loading satellite images and corresponding masks.
    Handles different image enhancement strategies ('none', 'all', 'hybrid').
    """
    def __init__(self, image_dir, mask_dir, transform=None, use_enhancement='all'):
        self.images = sorted(glob(os.path.join(image_dir, "*.tif")))
        self.masks = sorted(glob(os.path.join(mask_dir, "*.png")))
        self.transform = transform
        self.use_enhancement = use_enhancement
        
        assert len(self.images) == len(self.masks), "Mismatch between number of images and masks."

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_bgr = cv2.imread(self.images[idx])
        mask_bgr = cv2.imread(self.masks[idx])

        should_enhance = False
        if self.use_enhancement == 'all':
            should_enhance = True
        elif self.use_enhancement == 'hybrid':
            should_enhance = random.choice([True, False])

        if should_enhance:
            img_bgr = enhance_image_dip(img_bgr)
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
        mask = rgb_mask_to_multiclass_robust(mask_rgb)
        
        if self.transform:
            augmented = self.transform(image=img_rgb, mask=mask)
            img, mask = augmented['image'], augmented['mask']
            
        return img, mask.long()