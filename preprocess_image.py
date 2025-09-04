import cv2
import os
import numpy as np

# image_path = 'data/33.png'
# final_path = 'temp/preprocessed_for_ocr_33.png'
data_folder = 'data'
output_folder = 'temp'
os.makedirs(output_folder, exist_ok=True)

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
        "green": {"lower": (35, 30, 40), "upper": (95, 255, 255)},
        # "white": {"lower": (70, 20, 40), "upper": (179, 90, 255)},   
        "blue":  {"lower": (40, 20, 60), "upper": (170, 125, 255)}  
    }

    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)

    # A mask for each color range provided
    for name, rng in hsv_ranges.items():
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

def remove_long_lines(mask, orig_img=None, min_len=120, thickness=3):
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=min_len, maxLineGap=40)
    line_mask = np.zeros_like(mask)
    if lines is None:
        return mask, line_mask
    for l in lines[:,0]:
        x1,y1,x2,y2 = l
        cv2.line(line_mask, (x1,y1), (x2,y2), 255, thickness)
    # Subtract line mask from mask
    cleaned = cv2.bitwise_and(mask, cv2.bitwise_not(line_mask))
    return cleaned, line_mask



def apply_clahe(gray):
    clipLimit = 3.0
    tileGridSize = (8,8)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(gray)

def binarize_image(gray, method='otsu', adaptive=None):
    if method == 'otsu':
        _, th = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        bs = adaptive.get("blockSize", 31)
        C = adaptive.get("C", 10)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, bs, C)
    else:
        # fallback simple threshold
        _, th = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    if np.mean(th) < 127:
        th = cv2.bitwise_not(th)
    return th

# Contrast Stretching (for future use if contrast issues arise)
def contrast_stretch(img):
    p2, p98 = np.percentile(img, (2, 98))
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

def thick_font(img):
    img = cv2.bitwise_not(img)
    kernel = np.ones((1,2),np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.bitwise_not(img)
    return img

def thin_font(img):
    img = cv2.bitwise_not(img)
    kernel = np.ones((1,2),np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.bitwise_not(img)
    return img

# ---------- Pipeline execution ----------
# img = cv2.imread(image_path)
# if img is None:
#     raise FileNotFoundError(f"Input not found: {image_path}")

for fname in os.listdir(data_folder):
    in_path = os.path.join(data_folder, fname)

    # Only process image files
    if not (fname.lower().endswith(".png") or fname.lower().endswith(".jpg") or fname.lower().endswith(".jpeg")):
        continue

    img = cv2.imread(in_path)
    if img is None:
        print(f"Skipping {fname}, could not read.")
        continue

    # Convert to grayscale and CLAHE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_clahe = apply_clahe(gray)

    # Binarize
    bin_img = binarize_image(gray_clahe, method='otsu')

    # Invert for final output (white text on black bg)
    final_img = 255 - bin_img

    # cv2.imwrite(final_path, final_img)
    base_name = os.path.splitext(fname)[0]
    out_path = os.path.join(output_folder, f"preprocessed_for_ocr_{base_name}.png")
    cv2.imwrite(out_path, final_img)

    print(f"Preprocessed: {fname} -> {out_path}")


# print("Preprocessing complete. Saved intermediate files to:", final_path)
# print("Final preprocessed image:", final_path)
print("Preprocessing complete. Files saved in:", output_folder)
