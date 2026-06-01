# analyse_utils_new.py (Corrected to Exclude "Unlabelled" from Mean)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, jaccard_score
import pandas as pd
from data_utils_final import CLASS_NAMES # Import class names for consistency

def plot_results(all_preds, all_masks):
    print("\n--- Generating Evaluation Results ---")
    
    flat_true = all_masks.flatten()
    flat_pred = all_preds.flatten()
    
    all_labels = np.arange(len(CLASS_NAMES))
    
    # --- 1. Confusion Matrix ---
    cm = confusion_matrix(flat_true, flat_pred, labels=all_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Pixel-wise Confusion Matrix')
    plt.show()
    print("Generated: Confusion Matrix")

    # ------------------------- CORE FIX: Exclude Unlabelled Class from Mean -------------------------
    # We only want to average the scores of the meaningful classes (from index 1 upwards).
    labels_to_include = all_labels[1:] # This will be [1, 2, 3, 4, 5, 6]
    # ---------------------------------------------------------------------------------------------

    # --- 2. Overall Metrics (Macro Average over classes of interest) ---
    # Overall Pixel Accuracy is still calculated over ALL pixels, as its definition implies.
    overall_accuracy = accuracy_score(flat_true, flat_pred)
    
    # For all other macro averages, we pass the list of labels we want to consider.
    overall_iou = jaccard_score(
        flat_true, flat_pred, average='macro', labels=labels_to_include, zero_division=0
    )
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        flat_true, flat_pred, average='macro', labels=labels_to_include, zero_division=0
    )
    
    print("\n--- Overall Performance Metrics (Macro Average on Relevant Classes) ---")
    print(f"Overall Pixel Accuracy: {overall_accuracy:.4f}")
    print(f"Mean IoU (mIoU):       {overall_iou:.4f}")
    print(f"Mean Precision:        {overall_precision:.4f}")
    print(f"Mean Recall:           {overall_recall:.4f}")
    print(f"Mean F1-Score:         {overall_f1:.4f}")

    # --- 3. Class-wise Metrics (This part remains the same to show all classes) ---
    # We still want to see the performance of every single class in the detailed table.
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_true, flat_pred, labels=all_labels, zero_division=0, average=None
    )
    iou = jaccard_score(flat_true, flat_pred, labels=all_labels, average=None)
    
    metrics_data = {
        'Class': CLASS_NAMES,
        'IoU': iou,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    print("\n--- Class-wise Performance Metrics ---")
    print(metrics_df.to_string(index=False, float_format="%.4f"))