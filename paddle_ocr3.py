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
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
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

def extract_top_reinforcement(results, beam_center_y=None,
                              min_cluster_separation_ratio=0.08,
                              top_fraction_fallback=0.40,
                              beam_center_tol=2.0):
    """
    Extract only TOP reinforcement into LEFT TOP, MID TOP, RIGHT TOP.
    results: iterable of (text, score, box) as in your OCR output.
    - Uses 1D k-means style clustering on cy to separate top vs bottom.
    - If clustering isn't strong enough, falls back to taking top_fraction_fallback of bars.
    Returns dict {"LEFT TOP": "...", "MID TOP": "...", "RIGHT TOP": "..."}
    """
    bar_pattern = re.compile(r"(\+?\d+)\s*-\s*(\d+)\s*\(([TC])\)", re.IGNORECASE)

    bars = []
    for text, score, box in results:
        if not text:
            continue
        for m in bar_pattern.finditer(text):
            count_raw, dia, pos = m.groups()
            try:
                count = int(count_raw.replace("+", ""))
            except Exception:
                # keep raw if cannot parse, but normally it will parse
                count = count_raw
            cx, cy = _center(box)
            # normalized text (removes leading +); keep original style if you prefer
            normalized = f"{count}-{dia}({pos.upper()})"
            bars.append({
                "text": normalized,
                "count": count,
                "dia": int(dia),
                "pos": pos.upper(),  # 'T' or 'C'
                "cx": cx, "cy": cy
            })

    # default empty result
    if not bars:
        return {"LEFT TOP": "", "MID TOP": "", "RIGHT TOP": ""}

    ys = [b["cy"] for b in bars]
    xmin, xmax = min(b["cx"] for b in bars), max(b["cx"] for b in bars)
    ymin, ymax = min(ys), max(ys)
    y_range = ymax - ymin if ymax - ymin != 0 else 1.0

    # If only one bar, treat it as top (fallback)
    if len(bars) == 1:
        top_bars = bars.copy()
    else:
        # 1D two-cluster (k=2) iterative assignment on cy
        # init centers
        c1 = ymin
        c2 = ymax
        for _ in range(20):
            cluster1 = [b for b in bars if abs(b["cy"] - c1) <= abs(b["cy"] - c2)]
            cluster2 = [b for b in bars if abs(b["cy"] - c2) < abs(b["cy"] - c1)]
            new_c1 = sum(b["cy"] for b in cluster1) / len(cluster1) if cluster1 else c1
            new_c2 = sum(b["cy"] for b in cluster2) / len(cluster2) if cluster2 else c2
            if abs(new_c1 - c1) < 1e-3 and abs(new_c2 - c2) < 1e-3:
                break
            c1, c2 = new_c1, new_c2

        # choose the cluster with smaller mean cy as top
        center1 = sum(b["cy"] for b in cluster1) / len(cluster1) if cluster1 else float('inf')
        center2 = sum(b["cy"] for b in cluster2) / len(cluster2) if cluster2 else float('inf')
        top_cluster = cluster1 if center1 < center2 else cluster2
        other_cluster = cluster2 if top_cluster is cluster1 else cluster1
        center_dist = abs(center1 - center2)

        # accept clustering only if cluster centers are sufficiently separated relative to y_range
        if center_dist >= (min_cluster_separation_ratio * y_range):
            top_bars = top_cluster
        else:
            # fallback: take bars in top X% of the Y range
            cutoff = ymin + top_fraction_fallback * y_range
            top_bars = [b for b in bars if b["cy"] <= cutoff]
            if not top_bars:
                # last resort: take the half with smaller cy values
                top_bars = sorted(bars, key=lambda b: b["cy"])[: max(1, len(bars)//2)]

    # If beam_center_y given, filter to above it (smaller cy). Note: changed to beam_center_y - tol
    if beam_center_y is not None:
        top_filtered = [b for b in top_bars if b["cy"] <= (beam_center_y - beam_center_tol)]
        if top_filtered:
            top_bars = top_filtered
        # if filtering removed everything, keep previously computed top_bars (don't force empty)

    # Now assign to left/mid/right by x-position
    top_bars.sort(key=lambda b: b["cx"])
    xs = [b["cx"] for b in top_bars]
    xmin = min(xs); xmax = max(xs)
    width = xmax - xmin if xmax - xmin != 0 else 1.0
    third = width / 3.0

    def region_for_x(cx):
        if cx <= xmin + third:
            return "LEFT"
        elif cx >= xmin + 2*third:
            return "RIGHT"
        else:
            return "MID"

    reinforcement = {"LEFT TOP": [], "MID TOP": [], "RIGHT TOP": []}

    for b in top_bars:
        r = region_for_x(b["cx"])
        if b["pos"] == "T":
            if r == "MID":
                # replicate mid top into all three
                reinforcement["LEFT TOP"].append(b["text"])
                reinforcement["MID TOP"].append(b["text"])
                reinforcement["RIGHT TOP"].append(b["text"])
            else:
                reinforcement[f"{r} TOP"].append(b["text"])
        elif b["pos"] == "C":
            # rule: C values only go to LEFT or RIGHT (not MID)
            if r in ("LEFT", "RIGHT"):
                reinforcement[f"{r} TOP"].append(b["text"])

    # join multiple values with comma
    reinforcement = {k: " , ".join(v) if v else "" for k, v in reinforcement.items()}
    return reinforcement


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
    top_reinf = extract_top_reinforcement(res, beam_center_y=chosen.get("_cy"))


    # build a single row for the chosen beam
    row = {**{h:"" for h in HEADERS}, **chosen, **shear, **top_reinf}
    # strip helper keys
    row.pop("_cx", None); row.pop("_cy", None); row.pop("_has_wd", None)

    df_new = pd.DataFrame([row], columns=HEADERS)

    # append/merge into Excel
    _merge_into_excel(df_new)
    print(f"Saved/updated: {row['BEAM NO']} -> {OUTPUT_XLSX}")


if __name__ == "__main__":
    image_path = "temp/preprocessed_for_ocr_1.png"  
    process_image(image_path)