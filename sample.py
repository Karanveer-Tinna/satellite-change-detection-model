import cv2
import numpy as np

# Read image
# img = cv2.imread(r"Dataset\Prepared_Dataset\train\images\tile_6.19_66.tif")
img = cv2.imread(r"Dataset\Main_Dataset\images\tile_5.18.tif")
cv2.imwrite("image.png", img)

# Normalize to 0-1 range
norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(norm, (5,5), 0)

# Edge detection for urban features
edges = cv2.Canny(blur, 100, 200)

# Contrast enhancement
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15,15))
cl = clahe.apply(l)
enhanced = cv2.merge((cl, a, b))
final = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

cv2.imwrite("enhanced.png", final)
cv2.imwrite("edges.png", edges)
