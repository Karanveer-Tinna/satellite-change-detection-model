# analysis_utils.py

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, jaccard_score, precision_score, recall_score, f1_score, accuracy_score
import rasterio.features
from shapely.geometry import shape

# --- DIP Post-Processing and Vectorization ---

def post_process_mask_dip(raw_prediction_mask, min_area_threshold=50):
    """
    Applies morphological closing and small object removal to the binary mask.
    (Function body remains the same as your main_workflow.py)
    """
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
    """
    Converts a cleaned raster mask into Shapely vector polygons.
    (Function body remains the same as your main_workflow.py)
    """
    polygons = []
    for geom, value in rasterio.features.shapes(final_mask.astype(np.int16), mask=final_mask, transform=transform):
        if value > 0:
            polygons.append(shape(geom))
    return polygons


def plot_results(
    train_losses, val_losses, val_mious,
    all_val_preds, all_val_masks,
    val_dataset,
    num_sample_images=5,
    class_names=['Background', 'Building'] 
):
    """
    Generates all required plots: Loss/mIoU curves, Confusion Matrix, Metrics Summary, and Visual Examples.
    (Function body remains the same as your main_workflow.py)
    """
    print("\n--- Generating Visualizations ---")
    
    # --- 1. Training and Validation Loss/Metrics Graphs ---
    # Only plot if history is available (not empty)
    if train_losses and val_losses and val_mious:
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
        plt.title('Validation Mean IoU (mIoU)')
        plt.xlabel('Epoch')
        plt.ylabel('mIoU')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        print("Generated: Training and Validation Curves")
    else:
        print("Skipped: Training/Validation Curves (No history data provided).")
    
    # --- 2. Confusion Matrix & Metrics (Uses all_val_preds/masks) ---
    flat_true = all_val_masks.flatten()
    flat_pred = all_val_preds.flatten()
    
    cm = confusion_matrix(flat_true, flat_pred, labels=np.arange(len(class_names)))
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Pixel-wise Confusion Matrix')
    plt.show()
    print("Generated: Confusion Matrix")

    # --- 3. Quantitative Metrics Summary (Table) ---
    overall_accuracy = accuracy_score(flat_true, flat_pred)
    overall_iou = jaccard_score(flat_true, flat_pred, average='binary') 
    precision_building = precision_score(flat_true, flat_pred, pos_label=1, average='binary', zero_division=0)
    recall_building = recall_score(flat_true, flat_pred, pos_label=1, average='binary', zero_division=0)
    f1_building = f1_score(flat_true, flat_pred, pos_label=1, average='binary', zero_division=0)
    
    print("\n--- Quantitative Metrics Summary (on Validation Set) ---")
    print(f"Overall Pixel Accuracy: {overall_accuracy:.4f}")
    print(f"Overall IoU (Jaccard Index): {overall_iou:.4f}")
    print(f"Building Class Precision: {precision_building:.4f}")
    print(f"Building Class Recall: {recall_building:.4f}")
    print(f"Building Class F1-Score: {f1_building:.4f}")

    # --- 4. Visual Examples of Segmentation ---
    plt.figure(figsize=(15, num_sample_images * 4)) 
    
    sample_indices = np.random.choice(len(val_dataset), num_sample_images, replace=False)
    
    for i, idx in enumerate(sample_indices):
        original_img_tensor, true_mask_tensor = val_dataset[idx]
        
        # Denormalize image for display
        img_display = original_img_tensor.permute(1, 2, 0).numpy()
        mean = np.array([0.485,0.456,0.406])
        std = np.array([0.229,0.224,0.225])
        img_display = std * img_display + mean
        img_display = np.clip(img_display, 0, 1)

        true_mask_display = true_mask_tensor.squeeze().numpy()
        predicted_mask_display = all_val_preds[idx].squeeze() 

        plt.subplot(num_sample_images, 3, i * 3 + 1)
        plt.imshow(img_display)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(num_sample_images, 3, i * 3 + 2)
        plt.imshow(true_mask_display, cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')

        plt.subplot(num_sample_images, 3, i * 3 + 3)
        plt.imshow(predicted_mask_display, cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()
    print(f"Generated: {num_sample_images} Visual Segmentation Examples")