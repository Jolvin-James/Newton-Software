import cv2
import pandas as pd
import numpy as np
import os
import re
from paddleocr import PaddleOCR

def init_ocr():
    ocr = PaddleOCR(lang='en', use_textline_orientation=False, 
                    use_doc_orientation_classify=False, use_doc_unwarping=False)
    return ocr

def run_ocr(ocr, image_path):
    prediction_results = ocr.predict(image_path)
    if not prediction_results:
        return []
    res = prediction_results[0]
    texts = res.get('rec_texts', [])
    scores = res.get('rec_scores', [])
    boxes = res.get('rec_polys', [])
    return list(zip(texts, scores, boxes))

def filter_results(ocr_results, score_threshold=0.7):
    filtered = [item for item in ocr_results if item[1] >= score_threshold]
    return filtered

def draw_results(image_path, results, window_name="OCR Results"):
    img = cv2.imread(image_path)
    for text, score, box in results:
        pts = box.astype(int)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        x, y = pts[0]
        cv2.putText(img, f"{text}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def results_to_df(results):
    """Convert list of (text, score, box) to DataFrame."""
    data = []
    for text, score, box in results:
        data.append({
            "Text": text,
            "Score": score,
            "Box": box.tolist() if isinstance(box, np.ndarray) else box
        })
    return pd.DataFrame(data)


def extract_beam_numbers_and_sizes(results):
    """
    Extract beam numbers (B5, B101a) and nearby width-depth (230x600) from OCR results.
    Returns a list of dicts: [{"Beam": "B5", "Width": 230, "Depth": 600}, ...]
    """
    beam_pattern = re.compile(r"\bB\d+[A-Za-z0-9]*\b", re.IGNORECASE)
    width_depth_pattern = re.compile(r"(\d+)[xX](\d+)")

    extracted = []

    # Go through all recognized texts
    for i, (text, score, box) in enumerate(results):
        beam_match = beam_pattern.search(text)
        width_depth_match = width_depth_pattern.search(text)

        if beam_match:
            beam_no = beam_match.group()

            # First, check the same text for width-depth
            if width_depth_match:
                w, d = width_depth_match.groups()
                extracted.append({"Beam No": beam_no, "Width": int(w), "Depth": int(d), "Level": 1})
            else:
                # If not found in the same text, try the next text item
                if i + 1 < len(results):
                    next_text = results[i + 1][0]
                    wd_match_next = width_depth_pattern.search(next_text)
                    if wd_match_next:
                        w, d = wd_match_next.groups()
                        extracted.append({"Beam No": beam_no, "Width": int(w), "Depth": int(d), "Level": 1})
                    else:
                        extracted.append({"Beam No": beam_no, "Width": None, "Depth": None, "Level": 1})
                else:
                    extracted.append({"Beam No": beam_no, "Width": None, "Depth": None, "Level": 1})

    return extracted


def extract_bottom_reinforcement(results):
    """
    Extract left, mid, right bottom reinforcement details based on position.
    Returns dict with Left Bottom, Mid Bottom, Right Bottom.
    """
    reinforcement_pattern = re.compile(r"\d+-\d+\([TC]\)", re.IGNORECASE)

    # Collect all reinforcement matches
    matches = []
    for text, score, box in results:
        if reinforcement_pattern.search(text):
            # Get center point of box
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            matches.append((text.strip(), cx, cy))

    if not matches:
        return {}

    # Identify bottom by Y (larger is bottom)
    # Threshold: filter Y > 250
    bottom_matches = [m for m in matches if m[2] > 250]

    if not bottom_matches:
        return {}

    # Sort by X for left-mid-right
    bottom_matches = sorted(bottom_matches, key=lambda x: x[1])

    # Left bottom = first, Right bottom = last
    left_bottom = bottom_matches[0][0]
    right_bottom = bottom_matches[-1][0]

    # Mid bottom: we might have multiple values at mid (e.g., 3-16(T) and 2-12(C))
    # Collect all matches whose X is near center (within ±200 px from average)
    mid_texts = []
    avg_x = sum(m[1] for m in bottom_matches) / len(bottom_matches)
    for text, cx, cy in bottom_matches:
        if abs(cx - avg_x) < 200:  # tune this range if needed
            mid_texts.append(text)

    return {
        "Left Bottom": left_bottom,
        "Mid Bottom": ", ".join(mid_texts),
        "Right Bottom": right_bottom
    }



if __name__ == "__main__":
    image_path = "temp/preprocessed_for_ocr.png"
    ocr = init_ocr()
    raw_results = run_ocr(ocr, image_path)
    filtered_results = filter_results(raw_results, score_threshold=0.7)

    # Convert all results to DataFrame (optional, for debugging)
    df = results_to_df(filtered_results)
    print(df)

    # Extract beam numbers and sizes
    beam_data = extract_beam_numbers_and_sizes(filtered_results)
    print(beam_data)

    bottom_data = extract_bottom_reinforcement(filtered_results)
    print(bottom_data)

    # Merge bottom_data into each beam dict
    if beam_data and bottom_data:
        for beam in beam_data:
            beam.update(bottom_data)
    
    # Save to Excel
    if beam_data:
        pd.DataFrame(beam_data).to_excel("beam_numbers.xlsx", index=False)
        print("Beam data saved to beam_numbers.xlsx")


    # Optional: visualize
    # draw_results(image_path, filtered_results)
