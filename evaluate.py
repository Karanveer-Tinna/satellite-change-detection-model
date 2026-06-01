import torch
import torch.nn as nn
import numpy as np
from placeholder_model import PlaceholderSegmentationModel
from simulate_dataloader import get_test_dataloader, get_full_resolution_image
from inference_utils import sliding_window_predict
from datetime import datetime

# --- SETUP ---

# 1. Define Model/Weights paths
MODEL_PATH = 'best_model.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def simulate_and_save_model(model):
    """
    SIMULATION STEP: Creates a dummy model and saves its state to allow
    the torch.load function to run successfully for demonstration.
    """
    print(f"Simulating trained model and saving to {MODEL_PATH}...")
    # Use a dummy optimizer state and epoch for a realistic checkpoint
    checkpoint = {
        'epoch': 10,
        'model_state_dict': model.state_dict(),
        'best_iou': 0.85
    }
    torch.save(checkpoint, MODEL_PATH)
    print("Simulation save complete.")

def load_best_model():
    """Loads the model architecture and weights from the checkpoint."""
    model = PlaceholderSegmentationModel().to(DEVICE)
    try:
        # Load the checkpoint dictionary
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        # Load the state dict into the model
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model successfully loaded from {MODEL_PATH}. Best IoU: {checkpoint['best_iou']:.4f}")
    except FileNotFoundError:
        print(f"Error: Model file {MODEL_PATH} not found. Please check path.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
        
    return model

def check_accuracy(model, test_loader):
    """
    Checks the accuracy of the model on the test dataloader using
    STANDARD (single patch/tile) inference.
    """
    print(f"\n--- Checking Standard Accuracy on Test Dataloader ({len(test_loader.dataset)} samples) ---")
    model.eval()
    total_correct = 0
    total_pixels = 0
    
    # Placeholder for IoU calculation (simplified for this sketch)
    total_intersection = 0
    total_union = 0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE) # True masks (ground truth)

            # Standard Forward Pass
            outputs = model(images) 
            
            # Convert logits to probabilities (e.g., using Sigmoid for binary)
            probs = torch.sigmoid(outputs)
            
            # Convert probabilities to predicted mask (binary: 0 or 1)
            predicted_mask = (probs > 0.5).float()

            # --- Accuracy Calculation ---
            correct = (predicted_mask == masks).sum().item()
            total_correct += correct
            total_pixels += masks.numel()
            
            # --- IoU Calculation (Simplified) ---
            intersection = (predicted_mask * masks).sum().item()
            union = (predicted_mask + masks).clamp(0, 1).sum().item()
            total_intersection += intersection
            total_union += union
            
    accuracy = total_correct / total_pixels
    mean_iou = total_intersection / (total_union + 1e-6) # Add epsilon for stability
    
    print(f"Standard Model Accuracy: {accuracy:.4f}")
    print(f"Standard Model IoU: {mean_iou:.4f}")

def main():
    # 0. Initialize and simulate model saving
    # The PlaceholderSegmentationModel must be instantiated to have weights to save
    initial_model = PlaceholderSegmentationModel().to(DEVICE)
    simulate_and_save_model(initial_model)
    
    # 1. Load the model
    model = load_best_model()
    if model is None:
        return

    # 2. Prepare DataLoaders
    test_loader = get_test_dataloader(batch_size=4, patch_size=128, num_samples=10)
    
    # 3. Check accuracy using standard inference
    check_accuracy(model, test_loader)
    
    # 4. Sliding Window Inference Prediction for Evaluation
    
    # Simulate a single full-resolution image (e.g., a large field-of-view from a Landsat image)
    LARGE_IMAGE_SIZE = 512
    full_image_np = get_full_resolution_image(H=LARGE_IMAGE_SIZE, W=LARGE_IMAGE_SIZE)
    
    # Run sliding window prediction
    tile_size = 128
    overlap_ratio = 0.25 # 25% overlap
    
    start_time = datetime.now()
    cleaned_mask, polygons = sliding_window_predict(
        full_image_np, model, DEVICE, tile_size=tile_size, overlap_ratio=overlap_ratio
    )
    end_time = datetime.now()
    
    print(f"\nSliding Window Inference Runtime: {(end_time - start_time).total_seconds():.2f} seconds")
    print(f"Final Cleaned Mask shape: {cleaned_mask.shape}")


if __name__ == "__main__":
    main()