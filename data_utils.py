import numpy as np
import cv2
from torch.utils.data import Dataset
import os
from glob import glob

# --- Configuration Placeholder ---
# NOTE: Replace these with the actual RGB values from your class_dict.csv
# Assuming the mask images use these specific RGB values for classes.
CLASS_RGB_MAP = {
    (0, 0, 0): "Unlabelled",
    (80, 140, 50): "Vegetation",
    (200, 200, 200): "Built-Up",
    (250, 235, 185): "Informal Settlements",
    (100, 100, 150): "Impervious Surfaces",
    (200, 160, 40): "Barren",
    (40, 120, 240): "Water",
}

# Building classes for binary extraction
BUILDING_RGBS = [(200, 200, 200), (250, 235, 185)] # Built-Up and Informal Settlements

# --- DIP Pre-Processing Functions ---

def enhance_image_dip(img_bgr):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    for local contrast enhancement and Gaussian Blur for noise reduction.
    Inputs: BGR numpy array (H, W, 3)
    Outputs: Enhanced BGR numpy array (H, W, 3)
    """
    # 1. Noise Reduction (Mild Gaussian Blur)
    blurred_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)

    # 2. Contrast Enhancement (CLAHE)
    lab = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    final_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return final_bgr

# --- Mask Conversion Function ---

def rgb_mask_to_binary(mask_rgb):
    """
    Converts a multi-class RGB mask to a binary (0/1) building mask.
    Inputs: mask_rgb: HxWx3 numpy uint8 (RGB)
    Outputs: HxW numpy uint8 (0 or 1)
    """
    building = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.uint8)
    for rgb in BUILDING_RGBS:
        # Check where ALL 3 channels match the target RGB
        building |= np.all(mask_rgb == np.array(rgb, dtype=np.uint8), axis=-1).astype(np.uint8)
    return building

# --- PyTorch Dataset ---

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Initializes the dataset, incorporating DIP pre-processing.
        """
        # Assuming filenames match between images and masks
        image_paths = sorted(glob(os.path.join(image_dir, "*.tif"))) # <-- CHANGED to *.tif
        mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))   # <-- CHANGED to *.tif

        # Simple check for matching paths (must be run per split)
        assert len(image_paths) == len(mask_paths)
        
        self.images = image_paths
        self.masks = mask_paths
        self.transform = transform
        
    def __len__(self): 
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Load BGR image and mask (OpenCV default)
        img_bgr = cv2.imread(self.images[idx])
        mask_bgr = cv2.imread(self.masks[idx])
        
        # 2. DIP Image Enhancement (Phase 1 Pre-processing)
        img_enhanced_bgr = enhance_image_dip(img_bgr)
        img_enhanced_rgb = img_enhanced_bgr[:,:,::-1] # BGR->RGB for model input
        
        # 3. Binary Mask Conversion (Phase 1 Pre-processing)
        mask_rgb = mask_bgr[:,:,::-1] # Convert mask to RGB first
        mask = rgb_mask_to_binary(mask_rgb)  # H,W -> 0/1
        mask = mask[..., None] # H,W,1 for albumentations compatibility
        
        # 4. Augmentation/Normalization (Phase 2)
        if self.transform:
            # Albumentations takes RGB image and mask
            augmented = self.transform(image=img_enhanced_rgb, mask=mask)
            img, mask = augmented['image'], augmented['mask']
            
        # mask is HxW or HxWx1 -> convert to torch Float for BCEWithLogitsLoss
        return img, mask.float()

# --- Example Usage (Not run in script) ---
if __name__ == '__main__':
    # This block shows how to get the paths (dummy)
    print("Data utility file ready.")