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


def extract_beam_numbers(results):
    """Extract beam numbers like B5, B501a using regex."""
    beam_pattern = re.compile(r"\bB\d+[A-Za-z0-9]*\b", re.IGNORECASE)
    beam_numbers = []

    width_depth_pattern = re.compile(r"^\d+x\d+$")
    w, d = text.split("x")

    for text, score, box in results:
        match = beam_pattern.search(text)
        if match:
            beam_numbers.append(match.group())

    return sorted(set(beam_numbers))


if __name__ == "__main__":
    image_path = "temp/preprocessed_for_ocr.png"
    ocr = init_ocr()
    raw_results = run_ocr(ocr, image_path)
    filtered_results = filter_results(raw_results, score_threshold=0.7)

    # Convert all results to DataFrame (optional, for debugging)
    df = results_to_df(filtered_results)
    print(df)

    # Extract beam numbers using regex
    beam_numbers = extract_beam_numbers(filtered_results)
    print("Beam numbers found:", beam_numbers)

    # Save beam numbers to Excel
    if beam_numbers:
        pd.DataFrame({"BEAM NO": beam_numbers}).to_excel("beam_numbers.xlsx", index=False)
        print("Beam numbers saved to beam_numbers.xlsx")


    # Optional: visualize
    draw_results(image_path, filtered_results)
