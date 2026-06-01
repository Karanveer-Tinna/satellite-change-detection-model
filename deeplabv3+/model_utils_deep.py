import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp

def get_model(model_name, device, encoder_name='resnet101', encoder_weights='imagenet', classes=7):
    """
    Initializes a segmentation model based on the provided model_name.
    
    Args:
        model_name (str): The name of the model ('unetplusplus' or 'deeplabv3plus').
        device (torch.device): The device to move the model to.
        encoder_name (str): Name of the encoder architecture.
        encoder_weights (str): Pre-trained weights for the encoder.
        classes (int): Number of output classes.

    Returns:
        torch.nn.Module: The initialized segmentation model.
    """
    if model_name.lower() == 'unetplusplus':
        print(f"Initializing U-Net++ model with encoder {encoder_name}.")
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=3, classes=classes
        ).to(device)
    elif model_name.lower() == 'deeplabv3plus':
        print(f"Initializing DeepLabV3+ model with encoder {encoder_name}.")
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=3, classes=classes
        ).to(device)
    else:
        raise ValueError(f"Model '{model_name}' is not supported. Choose 'unetplusplus' or 'deeplabv3plus'.")
    return model

# Loss function for multiclass segmentation
loss_fn_dice = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
loss_fn_ce = nn.CrossEntropyLoss()

def combined_loss(preds, targets):
    """Combined Dice and Cross-Entropy loss for multiclass segmentation."""
    return 0.6 * loss_fn_ce(preds, targets) + 0.4 * loss_fn_dice(preds, targets)

# Metric for multiclass segmentation
def iou_score(preds, targets, num_classes=7, eps=1e-7):
    """Computes Mean IoU for multiclass segmentation."""
    preds = torch.argmax(preds, dim=1)
    iou_per_class = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection + eps) / (union + eps))
            
    mean_iou = np.nanmean(iou_per_class)
    return mean_iou