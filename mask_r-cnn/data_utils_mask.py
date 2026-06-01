import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
import os
from glob import glob
import random

# --- Class and Color Definitions ---
CLASS_NAMES = [
    "Unlabelled", "Vegetation", "Built-Up", "Informal Settlements",
    "Impervious Surfaces", "Barren", "Water"
]
CLASS_RGB_VALUES = [ (0,0,0), (80,140,50), (200,200,200), (250,235,185), (100,100,150), (200,160,40), (40,120,240) ]

def enhance_image_dip(img_bgr):
    """Applies CLAHE enhancement in the LAB color space."""
    blurred_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
    lab = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    final_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return final_bgr

def create_instance_masks(semantic_mask, class_id):
    """
    Finds contiguous blobs for a given class_id in a semantic mask and returns
    instance masks and bounding boxes.
    """
    binary_mask = (semantic_mask == class_id).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8, cv2.CV_32S)
    masks = []
    boxes = []
    
    for i in range(1, num_labels):
        masks.append((labels == i).astype(np.uint8))
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        boxes.append([x, y, x + w, y + h])
        
    return masks, boxes

class InstanceSegmentationDataset(Dataset):
    """
    Dataset for Mask R-CNN. Converts semantic masks to instance masks on the fly.
    """
    def __init__(self, image_dir, mask_dir, transform=None, use_enhancement='none'):
        self.images = sorted(glob(os.path.join(image_dir, "*.tif")))
        self.masks_paths = sorted(glob(os.path.join(mask_dir, "*.png")))
        self.transform = transform
        self.use_enhancement = use_enhancement
        
        self.instance_classes = {
            "Built-Up": 2, "Informal Settlements": 3, "Water": 6,
        }
        self.class_map = {v: i + 1 for i, v in enumerate(self.instance_classes.values())}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_bgr = cv2.imread(self.images[idx])
        mask_semantic_bgr = cv2.imread(self.masks_paths[idx], cv2.IMREAD_UNCHANGED)
        
        semantic_map = np.zeros(mask_semantic_bgr.shape[:2], dtype=np.uint8)
        for i, color in enumerate(CLASS_RGB_VALUES):
            semantic_map[(mask_semantic_bgr == color[::-1]).all(axis=2)] = i

        if self.use_enhancement == 'all': img_bgr = enhance_image_dip(img_bgr)
        elif self.use_enhancement == 'hybrid':
            if random.choice([True, False]): img_bgr = enhance_image_dip(img_bgr)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        all_masks, all_boxes, all_labels = [], [], []

        for class_name, original_class_id in self.instance_classes.items():
            masks, boxes = create_instance_masks(semantic_map, original_class_id)
            if masks:
                model_class_id = self.class_map[original_class_id]
                all_masks.extend(masks); all_boxes.extend(boxes)
                all_labels.extend([model_class_id] * len(masks))

        target = {}
        if all_boxes:
            target["boxes"] = torch.as_tensor(all_boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(all_labels, dtype=torch.int64)
            target["masks"] = torch.as_tensor(np.array(all_masks), dtype=torch.uint8)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros(0, dtype=torch.int64)
            target["masks"] = torch.zeros((0, img_rgb.shape[0], img_rgb.shape[1]), dtype=torch.uint8)

        img_tensor = torch.from_numpy(img_rgb).permute(2,0,1)

        if self.transform:
            img_tensor = self.transform(img_tensor)
        else:
            img_tensor = img_tensor.float() / 255.0

        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))