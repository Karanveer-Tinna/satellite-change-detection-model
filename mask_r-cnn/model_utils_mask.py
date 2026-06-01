import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN

def get_model(num_classes):
    """
    Initializes a Mask R-CNN model with a Swin-S FPN backbone.
    
    This function manually constructs the backbone by:
    1. Loading a pre-trained Swin-S Transformer.
    2. Creating a graph of its feature extractor.
    3. Specifying which intermediate layers to use for the FPN.
    4. Building the FPN-wrapped backbone.
    5. Passing the custom backbone to the MaskRCNN constructor.
    
    Args:
        num_classes (int): The number of classes INCLUDING the background class.
    
    Returns:
        A PyTorch Mask R-CNN model ready for fine-tuning.
    """
    # 1. Load a pre-trained Swin Transformer model
    # We use swin_s (Swin Small) as a strong default.
    weights = torchvision.models.Swin_S_Weights.DEFAULT
    backbone_model = torchvision.models.swin_s(weights=weights)
    
    # 2. Isolate the feature extraction layers
    # We don't need the final norm, pooling, and classification head.
    backbone_body = backbone_model.features
    
    # 3. Specify the output layers and their channel dimensions for the FPN
    # These are the outputs of each of the 4 stages of the Swin Transformer.
    # The keys are the module names inside `backbone_body`.
    return_layers = {"1": "0", "3": "1", "5": "2", "7": "3"}
    
    # These are the output channels for the layers specified above for swin_s.
    # Stage 1: 96, Stage 2: 192, Stage 3: 384, Stage 4: 768
    in_channels_list = [96, 192, 384, 768]
    
    # 4. Build the FPN-wrapped backbone
    # `out_channels` is the number of channels in the FPN layers.
    backbone_with_fpn = BackboneWithFPN(backbone_body, return_layers, in_channels_list, out_channels=256)
    
    # 5. Pass the custom backbone to the MaskRCNN constructor
    model = MaskRCNN(backbone_with_fpn, num_classes=num_classes)
    
    print(f"Mask R-CNN model created with Swin-S FPN backbone for {num_classes} classes.")
    return model
