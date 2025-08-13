import cv2
import numpy as np

image_path = 'data/1.png'


def resize_keep_ar(img, scale=1.0):
    h,w = img.shape[:2]
    if scale == 1.0:
        return img
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LANCZOS4)

def gamma_correction(img, gamma=1.0):
    if gamma == 1.0:
        return img
    gamma_corrected = np.array(255*(img / 255) ** gamma, dtype = 'uint8')
    return gamma_corrected

def denoise_image(img, method='nl_means'):
    # For colored diagrams, fastNlMeansDenoisingColored often preserves edges well.
    if method == 'nl_means':
        # convert to uint8 if not
        if img.dtype != np.uint8:
            img = (img*255).astype(np.uint8)
        den = cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
        return den
    elif method == "bilateral":
        return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    else:
        return img.copy()
    


def hsv_color_masks(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Range for blue, green, white color in HSV
    hsv_ranges = {
        "green": {"lower": (40, 40, 40), "upper": (90, 255, 255)},
        "white": {"lower": (0, 0, 180), "upper": (180, 60, 255)},
        "blue": {"lower": (90, 50, 50), "upper": (140, 255, 255)}
    }

    # Prepare combined mask
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)

    # Generate a mask for each color range provided
    for _, rng in hsv_ranges.items():
        lower = np.array(rng["lower"], dtype=np.uint8)
        upper = np.array(rng["upper"], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        combined = cv2.bitwise_or(combined, mask)

    return combined

def morph_clean(mask, kernel_size=(3,3), close_iter=1, open_iter=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    return m

def remove_long_lines(mask, orig_img=None, min_len=150, thickness=3):
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=min_len, maxLineGap=20)
    line_mask = np.zeros_like(mask)
    if lines is None:
        return mask, line_mask
    for l in lines[:,0]:
        x1,y1,x2,y2 = l
        cv2.line(line_mask, (x1,y1), (x2,y2), 255, thickness)
    # Subtract line mask from mask
    cleaned = cv2.bitwise_and(mask, cv2.bitwise_not(line_mask))
    return cleaned, line_mask

# def deskew_image_using_mask(img_bgr, mask):
#     deskew_min_pixels = 50
#     # Compute angle from mask's non-zero pixels using cv.minAreaRect on their coords
#     coords = np.column_stack(np.where(mask > 0))
#     if coords.shape[0] < deskew_min_pixels:
#         return img_bgr, 0.0
#     rect = cv2.minAreaRect(coords)
#     angle = rect[-1]
#     if angle < -45:
#         angle = -(90 + angle)
#     else:
#         angle = -angle
#     (h, w) = img_bgr.shape[:2]
#     M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
#     rotated = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     return rotated, angle

def apply_clahe(gray):
    clipLimit = 3.0
    tileGridSize = (8,8)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(gray)

def binarize_image(gray, method='otsu', adaptive=None):
    if method == 'otsu':
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Decide whether to invert - prefer black text on white background
        if np.mean(th) < 127:
            th = cv2.bitwise_not(th)
        return th
    elif method == 'adaptive':
        bs = adaptive.get("blockSize", 31)
        C = adaptive.get("C", 10)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, bs, C)
        if np.mean(th) < 127:
            th = cv2.bitwise_not(th)
        return th
    else:
        # fallback simple threshold
        _, th = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        return th

def thick_font(img):
    img = cv2.bitwise_not(img)
    kernel = np.ones((1,2),np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.bitwise_not(img)
    return img



# ---------- Pipeline execution ----------
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Input not found: {image_path}")

# 1) Upscale
img_up = resize_keep_ar(img, scale=1.5)

img_gam = gamma_correction(img_up, gamma=1.4)

# 2) Denoise (preserve edges)
img_deno = denoise_image(img_gam, method="nl_means")

# 3) Color masks: green, white, blue (combine)
mask_combined = hsv_color_masks(img_deno)

# 4) Morphological clean on mask
mask_clean = morph_clean(mask_combined, kernel_size=(3,3), close_iter=1, open_iter=1)

# 5) Remove long CAD drawing lines (use mask edges via Hough)
mask_no_lines, line_mask = remove_long_lines(mask_clean, orig_img=img_deno, min_len=150, thickness=3)

# 6) Optional: Connected component filtering (remove tiny blobs)
nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_no_lines, connectivity=8)
filtered = np.zeros_like(mask_no_lines)
h_mean = img_deno.shape[0] / 100.0
for i in range(1, nb_components):
    area = stats[i, cv2.CC_STAT_AREA]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    # heuristics: keep components with reasonable area and height
    if area >= 8 and h >= 6:   # tune thresholds if needed
        filtered[labels == i] = 255

# 7) Deskew using the filtered mask (rotate original denoised image)
# rotated, angle = deskew_image_using_mask(img_deno, filtered)
# cv2.imwrite(os.path.join(OUT_DIR, '7_rotated.png'), rotated)
# print(f"Deskew angle applied: {angle:.2f} degrees")

# 8) Convert to grayscale and CLAHE
gray_rot = cv2.cvtColor(img_deno, cv2.COLOR_BGR2GRAY)
gray_clahe = apply_clahe(gray_rot)

# 9) Binarize (choose adaptive or otsu)
# choose method automatically: if a lot of mask content, try otsu; otherwise adaptive
method = 'otsu' if np.count_nonzero(filtered) > 50 else 'adaptive'
adaptive_thresh = {"blockSize": 31, "C": 10}
if method == 'otsu':
    bin_img = binarize_image(gray_clahe, method='otsu')
else:
    bin_img = binarize_image(gray_clahe, method='adaptive', adaptive=adaptive_thresh)

# thickening the font
bin_img = thick_font(bin_img)


# 10) Small morphological cleanup on binary (strengthen glyphs)
kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
final_dilate_iter = 1
final_erode_iter = 0
if final_dilate_iter > 0:
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel_small, iterations=final_dilate_iter)
if final_erode_iter > 0:
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_ERODE, kernel_small, iterations=final_erode_iter)


# 11) Final scale if needed for OCR engines
final_scale_for_ocr = 1.0
if final_scale_for_ocr != 1.0:
    bin_img = resize_keep_ar(bin_img, final_scale_for_ocr)

final_path = 'temp/preprocessed_for_ocr.png'
cv2.imwrite(final_path, bin_img)

print("Preprocessing complete. Saved intermediate files to:", final_path)
print("Final preprocessed image:", final_path)
