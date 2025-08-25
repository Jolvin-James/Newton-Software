import cv2
import numpy as np
import re
import pandas as pd
from paddleocr import PaddleOCR

# -------------------------------
# 1. Initialize OCR
# -------------------------------
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # use_angle_cls is correct flag

# -------------------------------
# 2. Detect beam rectangle (works on B/W drawings too)
# -------------------------------
def detect_beam_rectangle(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # invert so lines are white on black
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # dilate to close gaps in lines
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No beam rectangle detected!")

    # Largest contour is likely the beam box
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    return (x, y, x+w, y+h), img

# -------------------------------
# 3. Classify text with regex + zones
# -------------------------------
def classify_text(text, x, y, beam_box):
    bx1, by1, bx2, by2 = beam_box
    width = bx2 - bx1

    # Zones
    if x < bx1 + width/3:
        zone = "LEFT"
    elif x < bx1 + 2*width/3:
        zone = "MID"
    else:
        zone = "RIGHT"

    if y < by1:
        vpos = "TOP"
    elif y > by2:
        vpos = "BOTTOM"
    else:
        vpos = "INSIDE"

    # --- Regex checks ---
    if re.match(r"^B\d+[a-zA-Z0-9]*", text):  # Beam ID
        return "BEAM NO", text

    if re.match(r"^\d+x\d+$", text):          # Dimensions
        w, d = text.split("x")
        return "WIDTH/DEPTH", (w, d)

    if re.match(r"^\+\d+(\.\d+)?", text):     # Level
        return "LEVEL", text

    if re.match(r"\d+-\d+\([TC]\)", text):    # Bars
        if "(T)" in text:
            return f"{zone} TOP", text
        else:
            return f"{zone} BOTTOM", text

    if re.match(r"\d+@\d+", text):            # Stirrups
        return f"SHEAR STIRRUPS DIA ({zone[0]})", text

    if "SFR" in text:
        return "SFR", text

    if "LD" in text or "Curtail" in text:
        return "CURTAIL AT (DIST)", text

    return None, None

# -------------------------------
# 4. Extract structured beam data
# -------------------------------
def extract_beam_data(image_path):
    # Detect the beam rectangle first
    beam_box, img = detect_beam_rectangle(image_path)

    # Run OCR
    results = ocr.predict(image_path)

    # Initialize structured output
    beam_data = {
        "BEAM NO": "",
        "WIDTH": "",
        "DEPTH": "",
        "LEVEL": "",
        "LEFT BOTTOM": "",
        "BOTTOM LEFT AT (DIST)": "",
        "MID BOTTOM": "",
        "CURTAIL AT (DIST)": "",
        "RIGHT BOTTOM": "",
        "BOTTOM RIGHT AT (DIST)": "",
        "BENT UP": "",
        "LEFT TOP": "",
        "LEFT AT (DIST)": "",
        "MID TOP": "",
        "RIGHT TOP": "",
        "RIGHT AT (DIST)": "",
        "SFR": "",
        "SHEAR STIRRUPS DIA (L)": "",
        "LEFT SPACE STIRRUPS": "",
        "SHEAR STIRRUPS DIA (M)": "",
        "MID SPACE STIRRUPS": "",
        "SHEAR STIRRUPS DIA (R)": "",
        "RIGHT SPACE STIRRUPS": "",
        "SHEAR STIRRUP NUMBER": "",
        "EXTRA STIRRUP NUMBER": "",
        "EXTRA STIRRUP DIA": "",
        "HORI LINK DIA": "",
        "STIRRUPSID": "",
        "CONTINUOUS END": "",
        "DISCONTINUOUS END": "",
        "ATTACH MASTER ID": "",
    }

    # Ensure we only loop detections, not metadata
    detections = results[0].get("data", []) if isinstance(results[0], dict) else results[0]

    for line in detections:
        try:
            bbox = line[0]      # list of 4 corner points
            text = line[1][0]   # recognized string
            conf = line[1][1]   # confidence
        except Exception as e:
            print("⚠️ Skipping malformed OCR line:", line, e)
            continue

        if isinstance(bbox, (list, np.ndarray)) and len(bbox) >= 4:
            try:
                x = (bbox[0][0] + bbox[2][0]) / 2
                y = (bbox[0][1] + bbox[2][1]) / 2
            except Exception:
                print("⚠️ Invalid bbox format, skipping:", bbox)
                continue
        else:
            continue

        # Classify recognized text
        col, val = classify_text(text, x, y, beam_box)
        if col:
            if col == "WIDTH/DEPTH":
                beam_data["WIDTH"], beam_data["DEPTH"] = val
            elif col in beam_data:
                beam_data[col] = val

    return beam_data




# -------------------------------
# 5. Run on uploaded image
# -------------------------------
beam1 = extract_beam_data("temp/preprocessed_for_ocr.png")

df = pd.DataFrame([beam1])
df.to_excel("beam_output.xlsx", index=False)
print(df.head())
