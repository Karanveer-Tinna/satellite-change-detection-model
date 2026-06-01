import cv2
import numpy as np

# --- Setup ---
img = cv2.imread(r"Dataset\Prepared_Dataset\train\images\tile_6.19_58.tif")

# 1. Normalize (Redundant if 8-bit, but kept for consistency)
norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

# --- Enhanced Output (enhanced.png) ---
# Better contrast enhancement
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
# TUNE HERE: Increased clipLimit for more contrast, decreased grid size for local detail
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4)) 
cl = clahe.apply(l)
enhanced = cv2.merge((cl, a, b))
final_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
cv2.imwrite("enhanced_tuned.png", final_enhanced)

# --- Edge Output (edges.png) ---
# 2. Use Bilateral Filter for better edge preservation while reducing noise
# Applies a stronger smoothing than Gaussian (9, 75, 75 are example params)
bilateral = cv2.bilateralFilter(norm, 9, 75, 75)

# 3. Canny Edge Detection (adjust thresholds as needed)
edges = cv2.Canny(bilateral, 120, 240) 

# 4. (Optional) Post-processing the edges: Dilation to connect broken lines
kernel = np.ones((3,3), np.uint8)
edges_cleaned = cv2.dilate(edges, kernel, iterations=1)

cv2.imwrite("edges_tuned.png", edges_cleaned)