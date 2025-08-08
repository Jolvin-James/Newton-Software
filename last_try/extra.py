import cv2
import numpy as np
import pytesseract
from scipy.spatial import cKDTree

IMG_PATH = "beam_data.png"
DEBUG_OUT = "debug_output.png"
TESS_CONFIG = r'-c tessedit_char_whitelist=0123456789CT()\-+ --psm 7'

def ocr_snippet(gray, rect):
    x,y,w,h = rect
    snip = gray[y:y+h, x:x+w]
    snip = cv2.resize(snip, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, snip = cv2.threshold(snip, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(snip, config=TESS_CONFIG).strip()

# 1. load
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Could not load '{IMG_PATH}'")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. text ROIs
_, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kern = cv2.getStructuringElement(cv2.MORPH_RECT, (25,3))
closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kern)
cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

text_rois = []
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if w>60 and 8<h<60 and w/h>3:
        text_rois.append((x,y,w,h))

print(f"Found {len(text_rois)} text ROIs")

labels = []
for rect in text_rois:
    txt = ocr_snippet(gray, rect)
    if txt:
        cx, cy = rect[0]+rect[2]//2, rect[1]+rect[3]//2
        labels.append({'text': txt, 'rect': rect, 'centroid': (cx,cy)})
print(f"OCR’d {len(labels)} labels:", [l['text'] for l in labels])

# 3. beam ROIs
_, bw2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cnts2, _ = cv2.findContours(bw2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

beams = []
for c in cnts2:
    x,y,w,h = cv2.boundingRect(c)
    if w>200 and h<80:
        bx, by = x+w//2, y+h//2
        beams.append({'rect':(x,y,w,h), 'centroid':(bx,by)})
print(f"Detected {len(beams)} beams")

# 4. link via KD‑Tree
if beams and labels:
    pts = np.array([b['centroid'] for b in beams])
    tree = cKDTree(pts)
    for l in labels:
        d, idx = tree.query(l['centroid'])
        l['beam_rect'] = beams[idx]['rect']
        l['beam_centroid'] = beams[idx]['centroid']
    for l in labels:
        print(f"→ '{l['text']}' @ {l['centroid']} → beam @ {l['beam_centroid']}")
else:
    print("Skipping linking (need at least one beam and one label)")

# 5. debug image
dbg = img.copy()
for l in labels:
    x,y,w,h = l['rect']
    cv2.rectangle(dbg,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(dbg, l['text'], (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
for b in beams:
    x,y,w,h = b['rect']
    cv2.rectangle(dbg,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imwrite(DEBUG_OUT, dbg)
print(f"Debug image written to {DEBUG_OUT}")
