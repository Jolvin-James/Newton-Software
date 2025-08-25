# pip install paddlepaddle paddleocr opencv-python pandas openpyxl
from paddleocr import PaddleOCR
import cv2, re, pandas as pd

IMG = "temp/preprocessed_for_ocr.png"  # your provided path

# High-accuracy English model; set language='en' or 'en_number'
ocr = PaddleOCR(use_angle_cls=True, lang='en')

img = cv2.imread(IMG)
h, w = img.shape[:2]

result = ocr.predict(IMG)  # or ocr.predict(img) if you already read with cv2

# result is a list, usually with one dict per input image
res = result[0]  # Only one image in batch

rec_texts = res["rec_texts"]
rec_scores = res["rec_scores"]
rec_boxes = res["rec_boxes"]

words = []
for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
    # rec_boxes: should be (4, 2) corners as np.array or list
    xs = [pt for pt in box]
    ys = [pt[1] for pt in box]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    words.append({"text": text.strip(), "conf": score, "x1": x1, "y1": y1, "x2": x2, "y2": y2})


# Helpers
def norm(s):
    return s.upper().replace('−','-').replace('–','-').replace('×','X').strip()

def in_roi(w, x1, y1, x2, y2):
    cx = (w["x1"]+w["x2"])/2; cy = (w["y1"]+w["y2"])/2
    return x1 <= cx <= x2 and y1 <= cy <= y2

# 1) find beam labels
beam_labels = []
for w in words:
    m = re.search(r'\bB\d+[A-Z]?\b', norm(w["text"]))
    if m:
        beam_labels.append({**w, "beam": m.group(0)})

rows = []
for b in beam_labels:
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    # ROI: above the label; tune margins for your scale
    roi_x1 = max(0, bx1 - 200); roi_x2 = min(w, bx2 + 200)
    roi_y1 = max(0, by1 - 380); roi_y2 = max(0, by1 - 5)

    cluster = [norm(wd["text"]) for wd in words if in_roi(wd, roi_x1, roi_y1, roi_x2, roi_y2)]

    size = next((m.group(0) for t in cluster for m in [re.search(r'\b\d{2,4}X\d{2,4}\b', t)] if m), "")
    bars = [t for t in cluster if re.search(r'\b\d+-\d+\([TC]\)', t)]
    center = [t for t in cluster if '(C)' in t]
    top    = [t for t in cluster if '(T)' in t and '+' not in t]
    plus   = [t for t in cluster if re.search(r'\+\d+-\d+\([TC]\)', t)]
    stir   = ' '.join([t for t in cluster if t=='ALL' or re.fullmatch(r'ST\d+', t)])
    spacing = ' '.join([m.group(0) for t in cluster for m in [re.search(r'\b\d{3,4}\b', t)] if m])

    rows.append({
        "Beam_No": b["beam"],
        "Size_mm": size,
        "Top_Bars": ' '.join(top+plus),
        "Center_Bars": ' '.join(center),
        "Bottom_Bars": '',  # fill similarly if your drawings label bottom bars
        "Stirrups": stir,
        "Spacing_mm": spacing
    })

df = pd.DataFrame(rows).drop_duplicates(subset=["Beam_No"])
df.to_excel("beam_schedule.xlsx", index=False)
print(df)
