import numpy as np
import cv2
from torch.utils.data import Dataset
import os
from glob import glob
from scipy.spatial.distance import cdist
from skimage.exposure import match_histograms

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

# def enhance_image_dip(img_bgr, gamma=0.8, alpha=0.8, beta=0.25):
#     """
#     Applies a chained satellite-image enhancement pipeline:
#     1. Gamma Correction
#     2. Edge-Preserving Denoising (Bilateral Filter)
#     3. Sobel-based Unsharp Masking
#     4. Laplacian Fine-Detail Boost
#     5. CLAHE on L-Channel
#     """
#     # Convert to float [0, 1] for precise operations
#     img_f = img_bgr.astype(np.float32) / 255.0

#     # --- 1. Gamma Correction ---
#     gamma_corrected = np.power(img_f, gamma)
#     gamma_corrected = np.clip(gamma_corrected, 0, 1)

#     # --- 2. Edge-Preserving Denoising (Bilateral Filter) ---
#     # Convert back to uint8 for cv2.bilateralFilter
#     denoised_uint8 = (gamma_corrected * 255).astype(np.uint8)
#     denoised = cv2.bilateralFilter(denoised_uint8, d=7, sigmaColor=50, sigmaSpace=50)
#     # Convert back to float [0, 1]
#     denoised_f = denoised.astype(np.float32) / 255.0

#     # Convert to grayscale (uint8) for Sobel/Laplacian
#     gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY) 

#     # --- 3. Unsharp Mask (Sobel-based) ---
#     sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
#     sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
#     edges = np.sqrt(sobel_x**2 + sobel_y**2)
#     edges = cv2.normalize(edges, None, 0, 1, cv2.NORM_MINMAX)
#     # Convert edges to 3-channel float [0, 1]
#     edges_color = cv2.cvtColor((edges*255).astype(np.uint8), cv2.COLOR_GRAY2BGR) / 255.0

#     # Apply unsharp mask: original + alpha * edge_mask
#     sobel_sharp = np.clip(denoised_f + alpha * edges_color, 0, 1)

#     # --- 4. Laplacian Fine-Detail Boost ---
#     laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
#     laplacian = np.abs(laplacian)
#     laplacian = cv2.normalize(laplacian, None, 0, 1, cv2.NORM_MINMAX)
#     # Convert laplacian to 3-channel float [0, 1]
#     lap_color = cv2.cvtColor((laplacian*255).astype(np.uint8), cv2.COLOR_GRAY2BGR) / 255.0

#     # Apply detail boost: sharpened + beta * detail_mask
#     detail_enhanced = np.clip(sobel_sharp + beta * lap_color, 0, 1)

#     # --- 5. CLAHE on Lightness Channel ---
#     # Convert final float [0, 1] result to uint8 [0, 255]
#     clahe_input = (detail_enhanced * 255).astype(np.uint8)
#     lab = cv2.cvtColor(clahe_input, cv2.COLOR_BGR2LAB)
#     L, A, B = cv2.split(lab)

#     clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
#     L_clahe = clahe.apply(L)

#     lab_clahe = cv2.merge([L_clahe, A, B])
#     final_img_bgr = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
#     return final_img_bgr

def enhance_image_dip(img_bgr, gamma = 0.8):
    img_f = img_bgr / 255.0
    gamma_img = np.power(img_f, gamma)
    gamma_img = (gamma_img * 255).astype(np.uint8)

    return gamma_img

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
    
    print("palette_rgb shape:", palette_rgb.shape, "dtype:", palette_rgb.dtype)

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