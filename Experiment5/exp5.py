import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# -------------------------------------------------------------
# Step 1 & 2: Load Image and Convert to Grayscale
# -------------------------------------------------------------
# Replace 'sample.jpg' with your test image path
image_path = 'sample.jpg'
image_bgr = cv2.imread(image_path)

if image_bgr is None:
    # Creating a synthetic image if no file is provided
    image_bgr = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(image_bgr, (50, 50), (200, 200), (255, 255, 255), -1)
    cv2.circle(image_bgr, (300, 300), 60, (200, 200, 200), -1)

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# -------------------------------------------------------------
# Step 3 & 4: SIFT Feature Detection and Visualization
# -------------------------------------------------------------
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(image_gray, None)

# Draw rich keypoints (showing size and orientation)
image_sift = cv2.drawKeypoints(
    image_rgb,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# -------------------------------------------------------------
# Step 5 & 6: HOG Feature Extraction and Visualization
# -------------------------------------------------------------
hog_features, hog_image = hog(
    image_gray,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True,
    channel_axis=None
)

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# -------------------------------------------------------------
# Display SIFT & HOG Results
# -------------------------------------------------------------
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(image_sift)
plt.title(f"SIFT Keypoints (Total: {len(keypoints)})")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(hog_image_rescaled, cmap="gray")
plt.title("HOG Feature Representation")
plt.axis("off")

plt.tight_layout()
plt.savefig("sift_hog_result.png")

# -------------------------------------------------------------
# Step 8: SIFT Feature Matching between Two Images
# -------------------------------------------------------------
# Creating a rotated version of the image to demonstrate invariance
rows, cols = image_gray.shape
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 0.8)  # 30 deg rotation, 0.8 scale
image_transformed = cv2.warpAffine(image_gray, M, (cols, rows))

# Detect keypoints and descriptors in transformed image
kp2, des2 = sift.detectAndCompute(image_transformed, None)

# Match features using FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(descriptors, des2, k=2)

# Lowe's ratio test to filter reliable matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw top matches
image_matches = cv2.drawMatches(
    image_rgb, keypoints,
    cv2.cvtColor(image_transformed, cv2.COLOR_GRAY2RGB), kp2,
    good_matches[:40], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(12, 6))
plt.imshow(image_matches)
plt.title(f"SIFT Feature Matching (Good Matches: {len(good_matches)})")
plt.axis("off")
plt.tight_layout()
plt.savefig("matching_result.png")