import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

# --- Model Definition ---

def get_model(device, encoder_name='resnet101', encoder_weights='imagenet', classes=1):
    """
    Initializes the DeepLabV3Plus model.
    """
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=classes,    # single-channel output for binary
        activation=None     # raw logits for BCEWithLogitsLoss
    ).to(device)
    return model

# --- Loss Function ---

# Initialize the components of the combined loss
BCE = nn.BCEWithLogitsLoss()
DICE = smp.losses.DiceLoss(mode='binary')

def combined_loss(preds, targets):
    """
    Weighted combination of BCE and Dice Loss for robust training.
    """
    # Weights can be tuned based on class imbalance.
    return 0.6 * BCE(preds, targets) + 0.4 * DICE(preds, targets)

# --- Metric ---

def iou_score(preds, targets, threshold=0.5, eps=1e-7):
    """
    Computes Mean IoU (Jaccard Index) for the binary prediction.
    """
    # Convert logits to probability, then to binary prediction
    preds = (torch.sigmoid(preds) > threshold).float()
    
    # Flatten across spatial dimensions
    inter = (preds * targets).sum(dim=(1,2,3))
    union = (preds + targets - preds * targets).sum(dim=(1,2,3))
    
    # Calculate IoU per image and average them (Mean over Batch)
    iou = ((inter + eps) / (union + eps)).mean().item()
    return iou

# --- Example Usage (Not run in script) ---
if __name__ == '__main__':
    print("Model utility file ready.")