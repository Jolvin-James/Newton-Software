# Preprocessing pipeline for OCR (no OCR/extraction included)
# This script implements the ordered steps discussed: upscale, denoise, color segmentation, morphology,
# line removal, deskew, CLAHE, binarization, small morphology, and saving intermediate results.
# It reads /mnt/data/1.png and writes several outputs to /mnt/data/.
#
# You can tweak parameters in the `params` dict below.
# No text extraction or pytesseract calls are included per request.

import cv2
import numpy as np
import os
from pathlib import Path

INPUT_PATH = 'data/1.png'
OUT_DIR = 'test_out'
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Parameters (tune these) ----------
params = {
    "upscale_factor": 2.0,               # initial upscale (2.0 = 2x)
    "denoise_method": "bilateral",       # 'bilateral' or 'nl_means' or None
    "bilateral": {"d":9, "sigmaColor":75, "sigmaSpace":75},
    "nl_means": {"h":10},
    # HSV ranges for color masks -- tune these with an HSV picker if needed
    "hsv_masks": {
        "green": {"lower": (40, 40, 40), "upper": (90, 255, 255)},
        "white": {"lower": (0, 0, 180), "upper": (180, 60, 255)},
        "blue":  {"lower": (90, 50, 50), "upper": (140, 255, 255)},
        # add others if needed
    },
    "morph_kernel_small": (3,3),
    "morph_kernel_long": (25,1),         # used for detecting long horizontal structures
    "remove_line_min_len": 150,          # min length for HoughLine removal
    "deskew_min_pixels": 50,             # min nonzero pixels required to compute skew
    "clahe": {"clipLimit":3.0, "tileGridSize":(8,8)},
    "adaptive_thresh": {"blockSize": 31, "C": 10},  # odd blockSize
    "final_dilate_iter": 1,
    "final_erode_iter": 0,
    "final_scale_for_ocr": 1.0           # optionally upscale final binary for OCR
}

# ---------- Helper functions ----------
def resize_keep_ar(img, scale=1.0):
    h,w = img.shape[:2]
    if scale == 1.0:
        return img
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

def denoise(img, method="bilateral"):
    if method == "bilateral":
        p = params["bilateral"]
        return cv2.bilateralFilter(img, d=p["d"], sigmaColor=p["sigmaColor"], sigmaSpace=p["sigmaSpace"])
    elif method == "nl_means":
        p = params["nl_means"]
        # convert to grayscale for fast Non-Local Means denoising
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            den = cv2.fastNlMeansDenoising(gray, h=p["h"])
            return cv2.cvtColor(den, cv2.COLOR_GRAY2BGR)
        else:
            return cv2.fastNlMeansDenoising(img, h=p["h"])
    else:
        return img.copy()

def hsv_color_masks(img_bgr, hsv_ranges):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    per_mask = {}
    for name, rng in hsv_ranges.items():
        lower = np.array(rng["lower"], dtype=np.uint8)
        upper = np.array(rng["upper"], dtype=np.uint8)
        m = cv2.inRange(hsv, lower, upper)
        per_mask[name] = m
        combined = cv2.bitwise_or(combined, m)
    return combined, per_mask

def morph_clean(mask, kernel_size=(3,3), close_iter=1, open_iter=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    return m

def remove_long_lines(mask, orig_img=None, min_len=150, thickness=3):
    # Use HoughLinesP on edges to detect long straight lines (useful for CAD lines)
    edges = cv2.Canny(mask, 50, 150)
    # Tune Hough parameters if needed
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

def deskew_image_using_mask(img_bgr, mask):
    # Compute angle from mask's non-zero pixels using cv.minAreaRect on their coords
    coords = np.column_stack(np.where(mask > 0))
    if coords.shape[0] < params["deskew_min_pixels"]:
        return img_bgr, 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle

def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=params["clahe"]["clipLimit"], tileGridSize=params["clahe"]["tileGridSize"])
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

# ---------- Pipeline execution ----------
img = cv2.imread(INPUT_PATH)
if img is None:
    raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

# 1) Upscale
img_up = resize_keep_ar(img, params["upscale_factor"])
cv2.imwrite(os.path.join(OUT_DIR, '1_upscaled.png'), img_up)

# 2) Denoise (preserve edges)
img_deno = denoise(img_up, method=params["denoise_method"])
cv2.imwrite(os.path.join(OUT_DIR, '2_denoised.png'), img_deno)

# 3) Color masks: green, white, blue (combine)
mask_combined, masks = hsv_color_masks(img_deno, params["hsv_masks"])
cv2.imwrite(os.path.join(OUT_DIR, '3_mask_combined.png'), mask_combined)
for k,v in masks.items():
    cv2.imwrite(os.path.join(OUT_DIR, f'3_mask_{k}.png'), v)

# 4) Morphological clean on mask
mask_clean = morph_clean(mask_combined, kernel_size=params["morph_kernel_small"], close_iter=1, open_iter=1)
cv2.imwrite(os.path.join(OUT_DIR, '4_mask_clean.png'), mask_clean)

# 5) Remove long CAD drawing lines (use mask edges via Hough)
mask_no_lines, line_mask = remove_long_lines(mask_clean, orig_img=img_deno, min_len=params["remove_line_min_len"], thickness=3)
cv2.imwrite(os.path.join(OUT_DIR, '5_mask_no_lines.png'), mask_no_lines)
cv2.imwrite(os.path.join(OUT_DIR, '5_line_mask.png'), line_mask)

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
cv2.imwrite(os.path.join(OUT_DIR, '6_mask_filtered_cc.png'), filtered)

# 7) Skip deskew - just use the filtered mask and denoised image directly
rotated = img_deno.copy()  # no rotation applied
angle = 0.0
cv2.imwrite(os.path.join(OUT_DIR, '7_rotated.png'), rotated)
print(f"Deskew skipped. Angle set to: {angle:.2f} degrees")

# 8) Convert to grayscale and CLAHE
gray_rot = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(OUT_DIR, '8_gray_rotated.png'), gray_rot)
gray_clahe = apply_clahe(gray_rot)
cv2.imwrite(os.path.join(OUT_DIR, '8_gray_clahe.png'), gray_clahe)

# 9) Binarize (choose adaptive or otsu)
# choose method automatically: if a lot of mask content, try otsu; otherwise adaptive
method = 'otsu' if np.count_nonzero(filtered) > 50 else 'adaptive'
if method == 'otsu':
    bin_img = binarize_image(gray_clahe, method='otsu')
else:
    bin_img = binarize_image(gray_clahe, method='adaptive', adaptive=params["adaptive_thresh"])
cv2.imwrite(os.path.join(OUT_DIR, '9_binarized_raw.png'), bin_img)

# 10) Small morphological cleanup on binary (strengthen glyphs)
kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
if params["final_dilate_iter"] > 0:
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel_small, iterations=params["final_dilate_iter"])
if params["final_erode_iter"] > 0:
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_ERODE, kernel_small, iterations=params["final_erode_iter"])
cv2.imwrite(os.path.join(OUT_DIR, '10_binarized_final.png'), bin_img)

# 11) Final scale if needed for OCR engines
if params["final_scale_for_ocr"] != 1.0:
    bin_img = resize_keep_ar(bin_img, params["final_scale_for_ocr"])

final_path = os.path.join(OUT_DIR, '11_preprocessed_for_ocr.png')
cv2.imwrite(final_path, bin_img)

print("Preprocessing complete. Saved intermediate files to:", OUT_DIR)
print("Final preprocessed image:", final_path)

# List saved files
for p in sorted(Path(OUT_DIR).glob('*')):
    print("-", p.name)
