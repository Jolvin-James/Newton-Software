import pandas as pd
import re
import os
from paddleocr import PaddleOCR

DEBUG = False
OUTPUT_XLSX = f"beam_extract_paddleocr.xlsx"

HEADERS = [
    "BEAM NO","WIDTH","DEPTH","LEVEL",
    "LEFT BOTTOM","BOTTOM LEFT AT (DIST)","MID BOTTOM","CURTAIL AT (DIST)","RIGHT BOTTOM","BOTTOM RIGHT AT (DIST)",
    "BENT UP","LEFT TOP","LEFT AT (DIST)","MID TOP","RIGHT TOP","RIGHT AT (DIST)",
    "SFR","SHEAR STIRUPPS LEG","SHEAR STIRRUPS DIA (L)","LEFT SPACE STIRRUPS",
    "SHEAR STIRRUPS DIA (M)","MID SPACE STIRRUPS","SHEAR STIRRUPS DIA (R)","RIGHT SPACE STIRRUPS",
    "SHEAR STIRRUP NUMBER","EXTRA STIRRUP NUMBER","EXTRA STIRRUP DIA","HORI LINK DIA","STIRRUPSID",
    "CONTINUOUS END","DISCONTINUOUS END","ATTACH MASTER ID"
]

def init_ocr():
    return PaddleOCR(lang='en', use_textline_orientation=False, 
                     use_doc_orientation_classify=False, use_doc_unwarping=False)

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
    return [item for item in ocr_results if item[1] >= score_threshold]

def _center(box):
    xs = [p[0] for p in box]; ys = [p[1] for p in box]
    return sum(xs)/len(xs), sum(ys)/len(ys)

def extract_beam_numbers_and_sizes(results):
    beam_pattern = re.compile(r"\bB\d+[A-Za-z0-9]*\b", re.IGNORECASE)
    width_depth_pattern = re.compile(r"(\d+)[xX](\d+)")
    extracted, seen = [], set()

    for i, (text, score, box) in enumerate(results):
        bm = beam_pattern.search(text)
        if not bm:
            continue
        beam_no = bm.group()
        if beam_no in seen:
            continue
        seen.add(beam_no)
        cx, cy = _center(box)

        w = d = None
        has_wd = False

        # try same text
        wd = width_depth_pattern.search(text)
        if wd:
            w, d = map(int, wd.groups()); has_wd = True
        else:
            # try immediate next text line (common in drawings)
            if i + 1 < len(results):
                next_text = results[i+1][0]
                wd2 = width_depth_pattern.search(next_text)
                if wd2:
                    w, d = map(int, wd2.groups()); has_wd = True

        extracted.append({
            "BEAM NO": beam_no, "WIDTH": w, "DEPTH": d, "LEVEL": 1,
            "_cx": cx, "_cy": cy, "_has_wd": has_wd
        })
    return extracted




def extract_shear_stirrups(results):
    stirrup_pattern = re.compile(r"(\d+)@(\d+)", re.IGNORECASE)   # e.g., 8@100
    leg_patterns = [
        re.compile(r"(\d+)\s*[-]?\s*leg", re.IGNORECASE),     # 2-leg, 2 leg, 4legs
        re.compile(r"\b(\d+)\s*L\b", re.IGNORECASE),          # 2L, 4L  (word boundary)
        re.compile(r"(\d+)\s*-\s*\d+\(C\)", re.IGNORECASE),   # 2-16(C)
    ]

    stirrup_matches = []
    for text, score, box in results:
        sm = stirrup_pattern.search(text)
        if sm:
            dia, spacing = sm.groups()
            cx, _ = _center(box)
            stirrup_matches.append((int(dia), int(spacing), cx))

    if not stirrup_matches:
        spacing = {"SHEAR STIRRUPS DIA (L)":"","LEFT SPACE STIRRUPS":"",
                   "SHEAR STIRRUPS DIA (M)":"","MID SPACE STIRRUPS":"",
                   "SHEAR STIRRUPS DIA (R)":"","RIGHT SPACE STIRRUPS":""}
    else:
        stirrup_matches.sort(key=lambda x: x[2])
        left = stirrup_matches[0] if len(stirrup_matches) > 0 else ("","",0)
        mid  = stirrup_matches[1] if len(stirrup_matches) > 1 else ("","",0)
        right= stirrup_matches[2] if len(stirrup_matches) > 2 else ("","",0)
        spacing = {
            "SHEAR STIRRUPS DIA (L)": left[0] if left else "",
            "LEFT SPACE STIRRUPS": left[1] if left else "",
            "SHEAR STIRRUPS DIA (M)": mid[0] if mid else "",
            "MID SPACE STIRRUPS": mid[1] if mid else "",
            "SHEAR STIRRUPS DIA (R)": right[0] if right else "",
            "RIGHT SPACE STIRRUPS": right[1] if right else ""
        }

    # legs
    legs = ""
    for text, score, box in results:
        for lp in leg_patterns:
            m = lp.search(text)
            if m:
                try:
                    legs = int(m.group(1))
                    break
                except:
                    continue
        if legs:
            break

    spacing["SHEAR STIRUPPS LEG"] = legs
    return spacing


def _collect_value_points(results):
    """Points that represent 'values' we want to associate with a single beam."""
    pats = [
        re.compile(r"\d+-\d+\([TC]\)", re.IGNORECASE),  # bars like 2-12(T)
        re.compile(r"(\d+)@(\d+)", re.IGNORECASE),      # stirrup spacing
        re.compile(r"\bALL\b", re.IGNORECASE),          # ALL 8@100 (often near spacing)
        re.compile(r"\d+[xX]\d+"),                      # width x depth
    ]
    pts = []
    for text, score, box in results:
        if any(p.search(text) for p in pats):
            pts.append(_center(box))
    return pts

def _select_primary_beam(beam_candidates, results):
    """Choose exactly one beam per image, the one closest to the value cluster."""
    if not beam_candidates:
        return None
    if len(beam_candidates) == 1:
        return beam_candidates[0]

    pts = _collect_value_points(results)
    if pts:
        avgx = sum(x for x, _ in pts) / len(pts)
        avgy = sum(y for _, y in pts) / len(pts)
        # prefer beams with width/depth recognized; tie-break by distance to cluster center
        def score(b):
            dist2 = (b["_cx"]-avgx)**2 + (b["_cy"]-avgy)**2
            bonus = -1e6 if b.get("_has_wd") or (b.get("WIDTH") is not None and b.get("DEPTH") is not None) else 0
            return dist2 + bonus
        chosen = min(beam_candidates, key=score)
    else:
        # fallback: prefer beam with width/depth; else the leftmost beam
        wd_beams = [b for b in beam_candidates if b.get("_has_wd") or (b.get("WIDTH") is not None and b.get("DEPTH") is not None)]
        chosen = wd_beams[0] if wd_beams else min(beam_candidates, key=lambda b: b["_cx"])

    if DEBUG:
        print(f"[DEBUG] primary beam selected: {chosen['BEAM NO']}")
    return chosen

def _merge_into_excel(df_new):
    if os.path.exists(OUTPUT_XLSX):
        old = pd.read_excel(OUTPUT_XLSX)
        # merge by BEAM NO: fill only empty cells in existing rows
        for _, row in df_new.iterrows():
            beam_no = row["BEAM NO"]
            if beam_no in old["BEAM NO"].values:
                idx = old.index[old["BEAM NO"] == beam_no][0]
                for col in HEADERS:
                    if col not in old.columns:
                        continue
                    if pd.isna(old.at[idx, col]) or old.at[idx, col] == "":
                        old.at[idx, col] = row.get(col, "")
            else:
                old = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
        old.to_excel(OUTPUT_XLSX, index=False)
    else:
        df_new.to_excel(OUTPUT_XLSX, index=False)

def process_image(image_path):
    ocr = init_ocr()
    raw = run_ocr(ocr, image_path)
    res = filter_results(raw)

    beam_candidates = extract_beam_numbers_and_sizes(res)
    if not beam_candidates:
        print("No beam numbers detected."); return

    # choose ONE beam for this image
    chosen = _select_primary_beam(beam_candidates, res)
    if not chosen:
        print("Could not select a primary beam."); return

    shear  = extract_shear_stirrups(res)

    # build a single row for the chosen beam
    row = {**{h:"" for h in HEADERS}, **chosen, **shear}
    # strip helper keys
    row.pop("_cx", None); row.pop("_cy", None); row.pop("_has_wd", None)

    df_new = pd.DataFrame([row], columns=HEADERS)

    # append/merge into Excel
    _merge_into_excel(df_new)
    print(f"Saved/updated: {row['BEAM NO']} → {OUTPUT_XLSX}")


if __name__ == "__main__":
    image_path = "temp/preprocessed_for_ocr_3.png"  
    process_image(image_path)