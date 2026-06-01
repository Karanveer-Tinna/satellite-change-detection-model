import cv2
import numpy as np
from skimage.exposure import match_histograms

# Read image
# img = cv2.imread("tile_4.49.tif")
# img = cv2.imread("jp22_1.9.tif")
img = cv2.imread("images/tile_5.6_12.tif")
cv2.imwrite("test_image.png", img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray_image.png", gray)

ref = cv2.imread("reference.png")
ref2 = cv2.imread("reference_2.png")

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L,A,B = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
L2 = clahe.apply(L)
clahe_img = cv2.cvtColor(cv2.merge([L2,A,B]), cv2.COLOR_LAB2BGR)

gamma = 0.8
img_f = img / 255.0
gamma_img = np.power(img_f, gamma)
gamma_img = (gamma_img * 255).astype(np.uint8)

cv2.imwrite("gamma_image.png", gamma_img)


# matched = match_histograms(img, ref, channel_axis=-1)
# cv2.imwrite("matched.png", matched)

# matched_2 = match_histograms(img, ref, channel_axis=-1)
# cv2.imwrite("matched_2.png", matched)
lab = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2LAB)
L,A,B = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
L2 = clahe.apply(L)
g_clahe = cv2.cvtColor(cv2.merge([L2,A,B]), cv2.COLOR_LAB2BGR)
cv2.imwrite("gamma_clahe.png", g_clahe)

blur = cv2.GaussianBlur(img, (5,5), 0)
sharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

cv2.imwrite("sharp.png", sharp)

blur = cv2.GaussianBlur(g_clahe, (5,5), 0)
sharp = cv2.addWeighted(g_clahe, 1.5, blur, -0.5, 0)

cv2.imwrite("g_clahe_sharp.png", sharp)

# final_img = matched

# cv2.imwrite("test_enhanced.png", final_img)

print("Saved enhanced satellite-quality image!")