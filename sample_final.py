import cv2
import numpy as np

# --- Setup ---
IMAGE_PATH = r"Dataset\Prepared_Dataset\train\images\tile_6.19_58.tif"
img = cv2.imread(IMAGE_PATH)

# Check if image loaded
if img is None:
    print(f"Error: Could not load image at {IMAGE_PATH}")
    exit()

# Convert to grayscale for some steps (Non-Local Means is faster)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# =================================================================
# PIPELINE 1: Advanced Visual Enhancement (Focus on Noise and Contrast)
# =================================================================

# TWEAK 1: NON-LOCAL MEANS DENOISING (Best for removing severe noise/pixelation)
# Parameters: h is the most important (higher h = more aggressive denoising)
# 7, 21 are standard for template/search window sizes.
denoised_gray = cv2.fastNlMeansDenoising(gray, None, h=25, templateWindowSize=7, searchWindowSize=21) 
denoised_color = cv2.cvtColor(denoised_gray, cv2.COLOR_GRAY2BGR)


# TWEAK 2: UN-SHARP MASKING (To boost perceived detail / sharpness)
# 1. Blur the denoised image
blurred = cv2.GaussianBlur(denoised_color, (0, 0), 5) # Kernel size (0,0) lets sigma handle it
# 2. Subtract the blurred image from the original (with weights)
# This sharpens the image by boosting the high-frequency components
# cv2.addWeighted(src1, alpha, src2, beta, gamma)
# If alpha=1.5 and beta=-0.5, it's: 1.5*Denoised - 0.5*Blurred + 0
sharpened = cv2.addWeighted(denoised_color, 1.5, blurred, -0.5, 0)


# TWEAK 3: CLAHE (for final local contrast enhancement on the sharpened image)
lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
# Parameters for dense urban areas: higher clipLimit, smaller tileGridSize
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6,6)) 
cl = clahe.apply(l)
enhanced_lab = cv2.merge((cl, a, b))
final_visual = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

cv2.imwrite("final_visual_better.png", final_visual)


# =================================================================
# PIPELINE 2: Structural Extraction (Focus on Lines/Edges)
# =================================================================

# Use the Denoised Gray image from above
# TWEAK 4: CANNY Edge Detection (adjusted for a cleaner input)
# Use a higher low threshold to eliminate background noise
canny_edges = cv2.Canny(denoised_gray, 140, 280) 

# TWEAK 5: HOUGH LINE TRANSFORM (To identify straight lines like roads/buildings)
# This converts the messy Canny output into structured lines
lines = cv2.HoughLinesP(canny_edges, 1, np.pi/180, 
                        threshold=30, # Minimum number of votes/points needed for a line (TUNE THIS)
                        minLineLength=15, # Minimum line length (TUNE THIS)
                        maxLineGap=5) # Maximum gap allowed between segments

# Create an output image for the lines
line_output = np.zeros_like(img)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Draw the detected lines in white on a black background
        cv2.line(line_output, (x1, y1), (x2, y2), (255, 255, 255), 1)

cv2.imwrite("final_lines_better.png", line_output)