import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import os
from tqdm import tqdm

# Geoprocessing libraries for final output
import rasterio.features
from shapely.geometry import shape

# Import utilities from the other files
from data_utils import SegmentationDataset, enhance_image_dip
from model_utils import get_model, combined_loss, iou_score


# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16 
NUM_EPOCHS = 10
LR = 1e-4

# --- Data Path Placeholders ---
BASE_PATH = "Dataset/Prepared_Dataset"
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train/images")
TRAIN_MASK_DIR = os.path.join(BASE_PATH, "train/masks")
VAL_IMG_DIR = os.path.join(BASE_PATH, "val/images")
VAL_MASK_DIR = os.path.join(BASE_PATH, "val/masks")


# --- Augmentations ---
train_transform = A.Compose([
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.Affine(rotate=(-15,15), scale=(0.9,1.1), translate_percent=(0.06,0.06)), 
    # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    
    # --- ADD THIS LINE ---
    A.Resize(height=128, width=128), # <--- Ensure all outputs are 120x120
    # ---------------------
    
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(height=128, width=128),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

# --- Validation Function ---

def validate_epoch(model, val_loader):
    model.eval()
    val_loss_sum = 0.0
    val_miou_sum = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE).squeeze(3).unsqueeze(1)

            # with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss = combined_loss(logits, masks)
            
            val_loss_sum += loss.item() * imgs.size(0)
            val_miou_sum += iou_score(logits, masks) * imgs.size(0)

    avg_loss = val_loss_sum / len(val_loader.dataset)
    avg_miou = val_miou_sum / len(val_loader.dataset)
    return avg_loss, avg_miou


# --- DIP Post-Processing and Vectorization ---

def post_process_mask_dip(raw_prediction_mask, min_area_threshold=50):
    """
    Applies morphological closing and small object removal to the binary mask.
    Inputs: binary mask (H, W) of type np.uint8 (0 or 255)
    Outputs: cleaned binary mask (H, W)
    """
    # Ensure input is 8-bit for OpenCV ops
    if raw_prediction_mask.max() <= 1:
        raw_prediction_mask = raw_prediction_mask * 255 
        
    # 1. Morphological Closing (DIP)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed_mask = cv2.morphologyEx(raw_prediction_mask, cv2.MORPH_CLOSE, kernel)
    
    # 2. Small Object Removal (Connected Component Analysis) (DIP)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask, 4, cv2.CV_32S)
    final_mask = np.zeros_like(closed_mask)
    
    for i in range(1, num_labels): # Start from 1 to skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area_threshold:
            final_mask[labels == i] = 255 # Keep the component
            
    return final_mask.astype(np.uint8)


def mask_to_polygons(final_mask, transform=None):
    """
    Converts a cleaned raster mask into Shapely vector polygons.
    NOTE: transform is needed for georeferencing but is set to None for tiles.
    """
    polygons = []
    # 1 is the value of the building class in the mask
    for geom, value in rasterio.features.shapes(final_mask.astype(np.int16), mask=final_mask, transform=transform):
        if value > 0:
            polygons.append(shape(geom))
    return polygons


# --- Sliding Window Inference Sketch (Phase 3) ---

def sliding_window_predict(large_image_path, model, tile_size=120, overlap_ratio=0.5):
    """
    Sketch for sliding window inference on a full-resolution patch.
    This replaces a standard single-tile inference.
    """
    print("--- Running Sliding Window Inference Sketch ---")
    
    # 1. Load the full image
    img = cv2.imread(large_image_path)
    # Perform DIP enhancement on the full image once (optional, can also be done per tile)
    img_enhanced = enhance_image_dip(img) 

    H, W, _ = img.shape
    step = int(tile_size * (1 - overlap_ratio))
    
    # Initialize accumulator for raw logits (for averaging overlapping predictions)
    full_logit_map = np.zeros((H, W), dtype=np.float32)
    hit_count = np.zeros((H, W), dtype=np.uint8)
    
    # NOTE: The full implementation requires padding, tiling, and normalization logic per window.
    # For a sketch, we simulate the result:
    
    # ... (Sliding window logic loop) ...
    # This loop generates full_logit_map from model predictions
    
    # 2. Convert final logits to probability map
    # Simulating a result:
    full_logit_map = np.random.rand(H, W) * 5 - 2 # Example raw logits
    prob_map = 1 / (1 + np.exp(-full_logit_map)) # Sigmoid
    
    # 3. Apply threshold to get raw binary mask
    raw_mask = (prob_map > 0.5).astype(np.uint8) * 255
    
    # 4. DIP Post-Processing (Phase 3, Step 8)
    cleaned_mask = post_process_mask_dip(raw_mask)
    
    # 5. Vectorization (Phase 3, Step 9)
    # Here we assume no geographic transform for a simple tile (transform=None)
    polygons = mask_to_polygons(cleaned_mask, transform=None) 
    
    print(f"Inference complete. Total polygons found: {len(polygons)}")
    # cv2.imwrite("final_footprints.png", cleaned_mask) # Save the cleaned mask
    return cleaned_mask, polygons


# --- Main Execution ---

def main():
    if not os.path.exists(TRAIN_IMG_DIR) or not os.path.exists(VAL_IMG_DIR):
        print("!! ERROR: Please ensure data paths are correct and the dataset is downloaded/extracted.")
        print(f"Expected path: {BASE_PATH}")
        # Placeholder data creation for testing structure - REMOVE FOR REAL RUNS
        # return

    # 1. DataLoaders
    train_dataset = SegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
    val_dataset = SegmentationDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # 2. Model, Optimizer, Scheduler

    model = get_model(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, mode='max')
    # scaler = torch.cuda.amp.GradScaler()

    best_miou = 0.0

    # 3. Training Loop
    print(f"Starting training on {DEVICE} for {NUM_EPOCHS} epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        # Training
        for batch_idx, (imgs, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} (Train)")):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE).squeeze(3).unsqueeze(1) # Ensure correct channel dimension [B, 1, H, W]
            
            # print(imgs.shape)
            # print(masks.shape)
            # if batch_idx == 0 and epoch == 1:
            #     print("\n--- DEBUG TENSOR CHECK ---")
            #     print(f"Image Batch Shape: {imgs.shape} | Expected: [B, 3, H, W]")
            #     print(f"Image Element Type: {imgs.dtype}")
            #     # A quick check for HWC (the image height/width are 120)
            #     if imgs.shape[1] == 120 and imgs.shape[2] == 120:
            #          print("!!! WARNING: Image tensor is likely HWC ([B, H, W, C]) instead of CHW ([B, C, H, W]) !!!")
                
            #     print(f"Mask Batch Shape: {masks.shape} | Expected: [B, 1, H, W]")
            #     print(f"Mask Element Type: {masks.dtype}")
            #     print("--------------------------\n")
            

            optimizer.zero_grad()

            logits = model(imgs)
            loss = combined_loss(logits, masks)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        
        avg_train_loss = train_loss / len(train_loader.dataset)

        # Validation
        val_loss, val_miou = validate_epoch(model, val_loader)
        scheduler.step(val_miou) # Use mIoU for scheduling

        print(f"Epoch {epoch} finished. Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")

        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved!")
            
    
    # 4. Inference and Post-processing (using the best model)
    print("\n--- Starting Final Inference and Post-processing ---")
    model.load_state_dict(torch.load("best_model.pth"))
    
    # Use a sample large image from your test set or original patches
    # NOTE: You must provide a path to a full-size image (e.g., 600x600) here
    # SAMPLE_LARGE_IMAGE_PATH = "path/to/your/full_patch_test_image.png" 
    
    # if os.path.exists(SAMPLE_LARGE_IMAGE_PATH):
    #     cleaned_mask, polygons = sliding_window_predict(SAMPLE_LARGE_IMAGE_PATH, model)
    # else:
    #     print(f"Warning: Could not run full inference. Image path not found: {SAMPLE_LARGE_IMAGE_PATH}")

if __name__ == '__main__':
    # Add a try/except block for graceful failure due to required paths
    try:
        main()
    except AssertionError as e:
        print(f"Fatal Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Check if the error is due to missing data paths or libraries