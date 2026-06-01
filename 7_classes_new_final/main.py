import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, jaccard_score, precision_score, recall_score, f1_score, accuracy_score
import os
os.environ["DNNL_VERBOSE"] = "1"
os.environ["MKLDNN_VERBOSE"] = "1"

# NOTE: Geoprocessing libraries are kept but will only be useful if specific classes are vectorized.
import rasterio.features
from shapely.geometry import shape

# Import utilities from the other files
from data_utils_final import SegmentationDataset, enhance_image_dip
# IMPORTANT: get_model, combined_loss, and iou_score MUST be updated in model_utils.py 
# to handle the 7-class multi-class segmentation task.
from model_utils_final import get_model, combined_loss, iou_score # Assuming these are updated

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 10
NUM_EPOCHS = 10
LR = 1e-4
IMAGE_SIZE = 128 # Define image size for consistency
NUM_CLASSES = 7  # <--- CRITICAL: Set to 7 classes

# Class Names for visualization (must match the order in data_utils_7)
CLASS_NAMES = [
    'Informal Settlements', 'Built-Up', 'Impervious Surfaces', 
    'Vegetation', 'Barren', 'Water', 'Unlabelled (Class 7)'
]
# NOTE: The provided list is likely incorrect and does not align with the 
# data_utils class indices (0:Unlabelled, 1:Veg, 2:Built-Up, 3:Informal, etc.).
# ***Using the correct order from data_utils_new.py:***
CLASS_NAMES_CORRECTED = [
    "Unlabelled", "Vegetation", "Built-Up", "Informal Settlements",
    "Impervious Surfaces", "Barren", "Water"
]


# --- Data Path Placeholders ---
# Adjust BASE_PATH as necessary for your environment
BASE_PATH = "../Dataset/Prepared_Dataset" 
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
    A.RandomBrightnessContrast(p=0.5),
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

# --- Evaluation Function (Modified for Multi-Class) ---
def validate_epoch(model, val_loader):
    model.eval()
    val_loss_sum = 0.0
    val_miou_sum = 0.0 # Multi-Class mIoU
    all_preds = []
    all_masks = []
    
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(DEVICE)
            # CRITICAL: Multi-Class masks must be [B, H, W] and type long (integer labels 0 to 6)
            masks = masks.to(DEVICE).squeeze(-1).long() # Should be [B, H, W] and type long/int64
            
            with torch.cuda.amp.autocast():
                logits = model(imgs) # Logits [B, NUM_CLASSES, H, W]
                loss = combined_loss(logits, masks) 
            
            val_loss_sum += loss.item() * imgs.size(0)
            
            # Multi-class prediction is argmax over the class dimension
            preds = torch.argmax(logits, dim=1) # [B, H, W]

            # The iou_score function is assumed to return the Mean IoU
            val_miou_sum += iou_score(logits, masks, num_classes=NUM_CLASSES) * imgs.size(0)
            
            all_preds.append(preds.cpu().numpy())
            all_masks.append(masks.cpu().numpy())

    avg_loss = val_loss_sum / len(val_loader.dataset)
    avg_miou = val_miou_sum / len(val_loader.dataset)
    
    # Concatenate all predictions and masks
    all_preds = np.concatenate(all_preds, axis=0) # Shape [N, H, W]
    all_masks = np.concatenate(all_masks, axis=0) # Shape [N, H, W]
    
    return avg_loss, avg_miou, all_preds, all_masks


# --- DIP Post-Processing and Vectorization (Sketch) ---
# NOTE: These functions are for post-processing a SINGLE binary class mask.
# They are included here for completeness but are not part of the training core.
def post_process_mask_dip(raw_prediction_mask, min_area_threshold=50):
    if raw_prediction_mask.max() <= 1:
        raw_prediction_mask = raw_prediction_mask * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed_mask = cv2.morphologyEx(raw_prediction_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask, 4, cv2.CV_32S)
    final_mask = np.zeros_like(closed_mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area_threshold:
            final_mask[labels == i] = 255

    return final_mask.astype(np.uint8)


def mask_to_polygons(final_mask, transform=None):
    polygons = []
    # Assumes the single class to vectorize has value 1 (which will be > 0)
    for geom, value in rasterio.features.shapes(final_mask.astype(np.int16), mask=final_mask, transform=transform):
        if value > 0:
            polygons.append(shape(geom))
    return polygons


# --- NEW: Plotting Function (Updated for Multi-Class) ---
def plot_results(
    train_losses, val_losses, val_mious,
    all_val_preds, all_val_masks,
    val_dataset,
    num_sample_images=5,
    class_names=CLASS_NAMES_CORRECTED # Use the corrected, aligned names
):
    print("\n--- Generating Visualizations (Multi-Class) ---")
    
    # 1. Training & Validation Loss/Metrics Graphs
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_mious, label='Validation mIoU', color='orange')
    plt.title('Validation Mean IoU (mIoU) - Multi-Class')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    print("Generated: Training and Validation Curves")
    
    # 2. Confusion Matrix (Multi-Class)
    flat_true = all_val_masks.flatten()
    flat_pred = all_val_preds.flatten()
    
    # Ensure labels match the range of class indices (0 to NUM_CLASSES-1)
    cm = confusion_matrix(flat_true, flat_pred, labels=np.arange(NUM_CLASSES))
    
    plt.figure(figsize=(10, 8)) 
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Pixel-wise Confusion Matrix (Multi-Class)')
    plt.show()
    print("Generated: Confusion Matrix")

    # 3. Quantitative Metrics Summary (Table)
    overall_accuracy = accuracy_score(flat_true, flat_pred)
    # Using 'macro' average to exclude the 'Unlabelled' class from mean, 
    # a custom loop or subsetting flat_true/flat_pred might be needed.
    # For simplicity here, we use macro over all classes (0-6).
    overall_miou_macro = jaccard_score(flat_true, flat_pred, average='macro', zero_division=0) 
    f1_macro = f1_score(flat_true, flat_pred, average='macro', zero_division=0)
    
    print("\n--- Quantitative Metrics Summary (on Validation Set) ---")
    print(f"Overall Pixel Accuracy: {overall_accuracy:.4f}")
    print(f"Mean IoU (Jaccard Index) [Macro over 7 classes]: {overall_miou_macro:.4f}")
    print(f"F1-Score [Macro over 7 classes]: {f1_macro:.4f}")

    precision_per_class = precision_score(flat_true, flat_pred, average=None, labels=np.arange(NUM_CLASSES), zero_division=0)
    recall_per_class = recall_score(flat_true, flat_pred, average=None, labels=np.arange(NUM_CLASSES), zero_division=0)
    
    print("\n--- Per-Class Metrics ---")
    print(f"{'Class':<20} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 45)
    for i in range(NUM_CLASSES):
        print(f"{class_names[i]:<20} | {precision_per_class[i]:<10.4f} | {recall_per_class[i]:<10.4f}")


    # 4. Visual Examples of Segmentation
    # NOTE: Using a color map for visualization
    cmap = plt.cm.get_cmap('Spectral', NUM_CLASSES) # Use a suitable colormap
    plt.figure(figsize=(15, num_sample_images * 4)) 
    sample_indices = np.random.choice(len(val_dataset), num_sample_images, replace=False)
    
    for i, idx in enumerate(sample_indices):
        original_img_tensor, true_mask_tensor = val_dataset[idx]
        
        # Denormalize image for display
        img_display = original_img_tensor.permute(1, 2, 0).numpy() # CHW to HWC
        mean = np.array([0.485,0.456,0.406])
        std = np.array([0.229,0.224,0.225])
        img_display = std * img_display + mean
        img_display = np.clip(img_display, 0, 1)

        true_mask_display = true_mask_tensor.squeeze().numpy() # [H, W]
        predicted_mask_display = all_val_preds[idx].squeeze()  # [H, W]

        plt.subplot(num_sample_images, 3, i * 3 + 1)
        plt.imshow(img_display)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(num_sample_images, 3, i * 3 + 2)
        # Use cmap with the correct range for the 7 classes (0 to 6)
        plt.imshow(true_mask_display, cmap=cmap, vmin=0, vmax=NUM_CLASSES-1)
        plt.title('Ground Truth Mask')
        plt.axis('off')

        plt.subplot(num_sample_images, 3, i * 3 + 3)
        plt.imshow(predicted_mask_display, cmap=cmap, vmin=0, vmax=NUM_CLASSES-1)
        plt.title('Predicted Mask')
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()
    print(f"Generated: {num_sample_images} Visual Segmentation Examples")


# --- Main Execution ---
def main():
    if not os.path.exists(TRAIN_IMG_DIR) or not os.path.exists(VAL_IMG_DIR):
        print("!! ERROR: Please ensure data paths are correct and the dataset is downloaded/extracted.")
        print(f"Expected path: {BASE_PATH}")
        return 

    # 1. DataLoaders
    train_dataset = SegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
    val_dataset = SegmentationDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_transform)
    
    # Increase num_workers if your system allows
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # 2. Model, Optimizer, Scheduler
    model = get_model(DEVICE, classes=NUM_CLASSES) # Ensure get_model accepts 'classes' argument
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, mode='max')
    scaler = torch.cuda.amp.GradScaler()

    best_miou = 0.0
    
    train_losses_history = []
    val_losses_history = []
    val_mious_history = []
    
    final_val_preds = None
    final_val_masks = None
    BEST_MODEL_PATH = "best_model_10.pth" # Renamed from "best_model_50.pth" in original

    # 3. Training Loop
    print(f"Starting Multi-Class training on {DEVICE} for {NUM_EPOCHS} epochs with {NUM_CLASSES} classes...")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        # Training
        for batch_idx, (imgs, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} (Train)")):
            imgs = imgs.to(DEVICE)
            # CRITICAL: Mask handling for multi-class [B, H, W]
            masks = masks.to(DEVICE).squeeze(-1).long() 
            
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                logits = model(imgs) # [B, 7, H, W]
                loss = combined_loss(logits, masks) # Multi-class loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * imgs.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses_history.append(avg_train_loss) 

        # Validation
        val_loss, val_miou, current_epoch_preds, current_epoch_masks = validate_epoch(model, val_loader)
        val_losses_history.append(val_loss) 
        val_mious_history.append(val_miou)  
        
        # Save predictions and masks from the LAST epoch (if not saved as best)
        final_val_preds = current_epoch_preds
        final_val_masks = current_epoch_masks

        scheduler.step(val_miou)

        print(f"Epoch {epoch} finished. Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")

        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Model saved to {BEST_MODEL_PATH} (Best mIoU: {best_miou:.4f})")
            # Save predictions from the best model epoch
            final_val_preds = current_epoch_preds
            final_val_masks = current_epoch_masks


    # 4. Generate and Plot Results
    # Load the best model weights for final evaluation
    if os.path.exists(BEST_MODEL_PATH):
        print(f"Loading best model from {BEST_MODEL_PATH} for final evaluation...")
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        # Re-run evaluation on the best model to ensure final_val_preds/masks are from the best model
        _, _, final_val_preds, final_val_masks = validate_epoch(model, val_loader)
    else:
        print("Warning: Best model not found. Using last epoch model state for plotting.")


    plot_results(
        train_losses_history,
        val_losses_history,
        val_mious_history,
        final_val_preds,
        final_val_masks,
        val_dataset,
        num_sample_images=5 
    )
            

if __name__ == '__main__':
    try:
        main()
    except AssertionError as e:
        print(f"Fatal Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}. Check if model_utils_final.py functions (get_model, combined_loss, iou_score) support NUM_CLASSES=7 multi-class segmentation.")