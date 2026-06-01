import cv2
import numpy as np

img = cv2.imread(r"Dataset\Prepared_Dataset\train\images")
mask = cv2.imread(r"Dataset\Prepared_Dataset\train\masks\jp22_1.10_5.png")

overlay = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
cv2.imshow("Overlay", overlay)
cv2.waitKey(0)
