import cv2
import numpy as np
import os

# ---------- Utility Functions ----------

def resize_keep_ar(img, scale=1.0):
    h, w = img.shape[:2]
    if scale == 1.0:
        return img
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LANCZOS4)

def gamma_correction(img, gamma=1.0):
    if gamma == 1.0:
        return img
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def denoise_image(img, method='nl_means', h=10, hColor=10):
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    if method == 'nl_means':
        den = cv2.fastNlMeansDenoisingColored(img, None, h, hColor, 7, 21)
    elif method == "median":
        den = cv2.medianBlur(img, 3)
    elif method == "gaussian":
        den = cv2.GaussianBlur(img, (3,3), 0.5)
    else:
        den = img.copy()
    return den

def sharpen_image(img):
    kernel_sharp = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel_sharp)

def hsv_color_masks(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Expand ranges with black/grey; tune as needed by your sample
    hsv_ranges = {
        "green": {"lower": (35, 30, 40), "upper": (85, 255, 255)},
        "white": {"lower": (0, 0, 210), "upper": (180, 45, 255)},
        "blue": {"lower": (95, 30, 40), "upper": (145, 255, 255)},
        "black": {"lower": (0, 0, 0), "upper": (180, 255, 40)},
        "grey":  {"lower": (0, 0, 40), "upper": (180, 30, 180)}
    }
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for _, rng in hsv_ranges.items():
        mask = cv2.inRange(hsv, np.array(rng["lower"], dtype=np.uint8), 
                                np.array(rng["upper"], dtype=np.uint8))
        combined = cv2.bitwise_or(combined, mask)
    return combined

def morph_clean(mask, kernel_size=(3,3), close_iter=1, open_iter=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    return m

def remove_long_lines(mask, min_len=150, thickness=3):
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=min_len, maxLineGap=20)
    line_mask = np.zeros_like(mask)
    if lines is not None:
        for l in lines[:,0]:
            x1, y1, x2, y2 = l
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness)
    cleaned = cv2.bitwise_and(mask, cv2.bitwise_not(line_mask))
    return cleaned, line_mask

def contour_filter(mask, min_area=8, min_height=6):
    nb_components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for i in range(1, nb_components):
        area = stats[i, cv2.CC_STAT_AREA]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if area >= min_area and h >= min_height:
            filtered[labels == i] = 255
    return filtered

def detect_skew_and_rotate(img, mask):
    coords = np.column_stack(np.where(mask > 0))
    if coords.shape[0] < 10:
        return img
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) > 1.0:
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    return img

def apply_clahe(gray, clipLimit=3.0, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(gray)

def binarize_image(gray, method='otsu', adaptive=None):
    if method == 'otsu':
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
        _, th = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        return th

def thick_font(img, iterations=2, shape=None):
    img = cv2.bitwise_not(img)
    k_h = max(2, int((shape[0] if shape is not None else 30) // 400))
    k_w = max(2, int((shape[1] if shape is not None else 30) // 700))
    kernel = np.ones((k_h,k_w),np.uint8)
    img = cv2.dilate(img, kernel, iterations=iterations)
    img = cv2.bitwise_not(img)
    return img

def crop_to_content(img):
    coords = np.argwhere(img)
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]

def save_step(img, fname, force_show=False):
    cv2.imwrite(fname, img)
    if force_show:
        print(f"Saved diagnostic step: {fname}")

# ---------- Pipeline Execution ----------

image_path = 'data/1.png'
final_path = 'temp/preprocessed_for_ocr.png'

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Input not found: {image_path}")

img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Failed to read: {image_path}")

# 1) Upscale
img_up = resize_keep_ar(img, scale=1.5)
save_step(img_up, "temp/step_upscaled.png")

# 2) Gamma correction
img_gam = gamma_correction(img_up, gamma=1.4)
save_step(img_gam, "temp/step_gamma.png")

# 3) Denoise + Sharpen
img_deno = denoise_image(img_gam, method="nl_means", h=8, hColor=8)
img_sharp = sharpen_image(img_deno)
save_step(img_sharp, "temp/step_sharp.png")

# 4) Color masks: green, white, blue, black, grey
mask_combined = hsv_color_masks(img_sharp)
save_step(mask_combined, "temp/step_mask_combined.png")

# 5) Morphological clean
mask_clean = morph_clean(mask_combined)
save_step(mask_clean, "temp/step_mask_clean.png")

# 6) Remove long lines
mask_no_lines, line_mask = remove_long_lines(mask_clean)
save_step(mask_no_lines, "temp/step_no_lines.png")
save_step(line_mask, "temp/step_line_mask.png")

# 7) Optional: Contour filter (remove small blobs)
filtered = contour_filter(mask_no_lines)
save_step(filtered, "temp/step_filtered_components.png")

# 8) Skew correction (will affect all further steps)
img_skew = detect_skew_and_rotate(img_sharp, filtered)
save_step(img_skew, "temp/step_deskew.png")
gray_rot = cv2.cvtColor(img_skew, cv2.COLOR_BGR2GRAY)

# 9) CLAHE
gray_clahe = apply_clahe(gray_rot)
save_step(gray_clahe, "temp/step_clahe.png")

# 10) Binarization
adaptive_thresh = {"blockSize": 31, "C": 10}
bin_otsu = binarize_image(gray_clahe, method='otsu')
bin_adapt = binarize_image(gray_clahe, method='adaptive', adaptive=adaptive_thresh)
bin_img = cv2.bitwise_or(bin_otsu, bin_adapt)
save_step(bin_img, "temp/step_binarized.png")

# 11) Font thickening (adaptive kernel)
bin_img = thick_font(bin_img, iterations=2, shape=bin_img.shape)
save_step(bin_img, "temp/step_thickened.png")

# 12) Light anti-aliasing (optional for OCR)
bin_img = cv2.GaussianBlur(bin_img, (3,3), 0.8)
save_step(bin_img, "temp/step_antialiased.png")

# 13) Morphological cleanup (final glyph strength)
kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel_small, iterations=1)

# 14) Crop to text content
bin_img = crop_to_content(bin_img)
save_step(bin_img, "temp/step_cropped.png")

# 15) Rescale for OCR if needed (Tesseract prefers ~300DPI)
if bin_img.shape[0] < 1000:
    bin_img = resize_keep_ar(bin_img, 2.0)
save_step(bin_img, final_path, force_show=True)

print(f"Preprocessing complete. Saved steps to ./temp/.")
print(f"Final preprocessed image for OCR: {final_path}")
