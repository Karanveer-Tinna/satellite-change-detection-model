import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import numpy as np

def get_model(device, encoder_name='resnet101', encoder_weights='imagenet', classes=7):
    """
    Initializes a U-Net++ model for 7-class segmentation.
    """
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=classes,  # Output 7 classes
    ).to(device)
    return model

# Loss function for multiclass segmentation
loss_fn = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
ce_loss = nn.CrossEntropyLoss()

def combined_loss(preds, targets):
    """
    Combined Dice and Cross-Entropy loss for multiclass segmentation.
    """
    # DiceLoss expects [B, C, H, W], CrossEntropyLoss expects [B, C, H, W] for preds and [B, H, W] for targets
    return 0.6 * ce_loss(preds, targets) + 0.4 * loss_fn(preds, targets)

# Metric for multiclass segmentation
def iou_score(preds, targets, num_classes=7, eps=1e-7):
    """
    Computes Mean IoU for multiclass segmentation.
    """
    # Get class predictions from logits
    preds = torch.argmax(preds, dim=1) # a tensor of shape [B, H, W]
    
    iou_per_class = []
    # Loop over each class to compute its IoU
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        
        # Handle cases where a class is not present in targets
        if union == 0:
            iou_per_class.append(float('nan')) 
        else:
            iou_per_class.append((intersection + eps) / (union + eps))
            
    # Calculate mean IoU, ignoring NaN values
    mean_iou = np.nanmean(iou_per_class)
    return mean_iou