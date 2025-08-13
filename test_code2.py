import cv2
import numpy as np
from pathlib import Path

INPUT = "data/1.png"
OUT = "temp/preprocessed_for_ocr.png"

def resize_keep_ar(img, scale=1.0):
    h,w = img.shape[:2]
    if scale == 1.0:
        return img
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

def gamma_correction(img, gamma=1.0):
    invGamma = 1.0 / gamma if gamma!=0 else 1.0
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def denoise_image(img, method='nl_means'):
    if method == 'nl_means':
        if img.dtype != np.uint8:
            img = (img*255).astype(np.uint8)
        return cv2.fastNlMeansDenoisingColored(img, None, h=8, hColor=8, templateWindowSize=7, searchWindowSize=21)
    elif method == 'bilateral':
        return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    else:
        return img.copy()

def hsv_color_masks(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ranges = {
        "green": ((35, 40, 40), (90, 255, 255)),
        "blue" : ((85, 50, 50), (140, 255, 255)),
        "white": ((0, 0, 200), (180, 40, 255)),
    }
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges.values():
        mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        combined = cv2.bitwise_or(combined, mask)
    return combined

def morph_clean(mask, kernel_size=(3,3), close_iter=1, open_iter=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    return m

def remove_long_lines(mask, min_len=200, thickness=3, canny_thresh=(50,150)):
    edges = cv2.Canny(mask, canny_thresh[0], canny_thresh[1])
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=min_len, maxLineGap=20)
    line_mask = np.zeros_like(mask)
    if lines is None:
        return mask, line_mask
    for l in lines[:,0]:
        x1,y1,x2,y2 = l
        cv2.line(line_mask, (x1,y1), (x2,y2), 255, thickness)
    cleaned = cv2.bitwise_and(mask, cv2.bitwise_not(line_mask))
    return cleaned, line_mask

def filter_components(mask, min_area=30, min_h=6, max_h=None):
    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    if max_h is None:
        max_h = mask.shape[0]
    for i in range(1, nb_components):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if area >= min_area and h >= min_h and h <= max_h:
            aspect = w / float(h + 1e-6)
            if 0.2 <= aspect <= 7.0:
                out[labels == i] = 255
    return out

def deskew_image_using_mask(img_bgr, mask):
    coords = np.column_stack(np.where(mask > 0))
    if coords.shape[0] < 50:
        return img_bgr, 0.0
    rect = cv2.minAreaRect(coords[:, ::-1])
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
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def binarize_image(gray, method='adaptive', blockSize=25, C=8):
    if method == 'otsu':
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(th) < 127:
            th = cv2.bitwise_not(th)
        return th
    else:
        if blockSize % 2 == 0:
            blockSize += 1
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, blockSize, C)
        if np.mean(th) < 127:
            th = cv2.bitwise_not(th)
        return th

def thicken_text(img_binary, kernel=(1,2), iterations=1):
    inv = cv2.bitwise_not(img_binary)
    k = np.ones(kernel, np.uint8)
    dil = cv2.dilate(inv, k, iterations=iterations)
    out = cv2.bitwise_not(dil)
    return out

# ---------- Run pipeline ----------
img = cv2.imread(INPUT)
if img is None:
    raise FileNotFoundError(INPUT + " not found")

img_up = resize_keep_ar(img, scale=1.5)
img_gam = gamma_correction(img_up, gamma=1.15)
img_deno = denoise_image(img_gam, method='nl_means')

mask_colors = hsv_color_masks(img_deno)
mask_colors = morph_clean(mask_colors, kernel_size=(3,3), close_iter=1, open_iter=1)

mask_no_lines, line_mask = remove_long_lines(mask_colors, min_len=200, thickness=3)

gray = cv2.cvtColor(img_deno, cv2.COLOR_BGR2GRAY)
sob = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
_, sob_th = cv2.threshold(sob, 25, 255, cv2.THRESH_BINARY)
combined_mask = cv2.bitwise_or(mask_no_lines, sob_th)

filtered = filter_components(combined_mask, min_area=30, min_h=6, max_h=200)
filtered = morph_clean(filtered, kernel_size=(3,3), close_iter=1, open_iter=0)

rotated_img, angle = deskew_image_using_mask(img_deno, filtered)

gray_rot = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
gray_clahe = apply_clahe(gray_rot)

bin_img = binarize_image(gray_clahe, method='adaptive', blockSize=25, C=8)
bin_img = thicken_text(bin_img, kernel=(1,2), iterations=1)

kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel_small, iterations=1)

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(OUT, bin_img)

OUT
