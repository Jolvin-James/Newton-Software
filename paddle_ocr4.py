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
                for reg in reinforcement:
                    reinforcement[reg].append(b["text"])
            else:
                reinforcement[f"{r} TOP"].append(b["text"])
        elif b["pos"] == "C":
            # allow C values in LEFT or RIGHT only, but if OCR put it with a T in same spot
            if r in ("LEFT", "RIGHT"):
                reinforcement[f"{r} TOP"].append(b["text"])

    for k in reinforcement:
        reinforcement[k] = list(dict.fromkeys(reinforcement[k]))  # preserve order, remove dupes

    # If exactly one distinct T bar was detected, treat it as global top.
    distinct_t_texts = {b["text"] for b in top_bars if b["pos"] == "T"}
    if len(distinct_t_texts) == 1:
        only_t = next(iter(distinct_t_texts))
        if any(only_t in reinforcement[reg] for reg in reinforcement):
            for reg in reinforcement:
                if only_t not in reinforcement[reg]:
                    reinforcement[reg].append(only_t)
                    
    # join multiple values with comma
    reinforcement = {k: " , ".join(v) if v else "" for k, v in reinforcement.items()}

    # special case: if only T bars exist but they ended up only in one region,
    # replicate them across all three (common in combined top labels)
    if any(v for v in reinforcement.values()):
        filled = [k for k, v in reinforcement.items() if v]
        if len(filled) == 1:
            only_vals = reinforcement[filled[0]]
            reinforcement = {k: only_vals for k in reinforcement.keys()}

    return reinforcement


def _extract_shear_stirrups_spacing(results):
    stirrup_pattern = re.compile(r"(\d+)@(\d+)", re.IGNORECASE)

    stirrup_matches = []
    r_x = None   # center marker
    beam_xmin, beam_xmax = None, None

    # Pass 1: find beam bounding box (B...)
    for text, score, box in results:
        if text.strip().upper().startswith("B"):
            (xmin, ymin), (xmax, ymax) = box[0], box[2]
            beam_xmin, beam_xmax = xmin, xmax
            break

    # Pass 2: collect stirrup matches
    for text, score, box in results:
        sm = stirrup_pattern.search(text)
        if sm:
            dia, spacing = sm.groups()
            cx, _ = _center(box)
            stirrup_matches.append((int(dia), int(spacing), cx, text))

        if text.strip().upper() == "R":
            r_x, _ = _center(box)

    spacing = {
        "SHEAR STIRRUPS DIA (L)": "", "LEFT SPACE STIRRUPS": "",
        "SHEAR STIRRUPS DIA (M)": "", "MID SPACE STIRRUPS": "",
        "SHEAR STIRRUPS DIA (R)": "", "RIGHT SPACE STIRRUPS": ""
    }

    if not stirrup_matches:
        return spacing

    # CASE 1: Only one spacing OR "ALL" → mid only
    if len(stirrup_matches) == 1 or any("ALL" in s[3].upper() for s in stirrup_matches):
        dia, spc, _, _ = stirrup_matches[0]
        spacing["SHEAR STIRRUPS DIA (M)"] = dia
        spacing["MID SPACE STIRRUPS"] = spc
        return spacing

    if beam_xmin is not None and beam_xmax is not None:
        width = beam_xmax - beam_xmin
        left_bound = beam_xmin + width * 0.25
        right_bound = beam_xmax - width * 0.25

        left_vals, mid_vals, right_vals = [], [], []

        for dia, spc, cx, txt in stirrup_matches:
            if cx <= left_bound:
                left_vals.append((dia, spc, cx))
            elif cx >= right_bound:
                right_vals.append((dia, spc, cx))
            else:
                mid_vals.append((dia, spc, cx))

        # Pick closest-to-region-center if multiple
        if left_vals:
            dia, spc, _ = min(left_vals, key=lambda v: abs(v[2] - beam_xmin))
            spacing["SHEAR STIRRUPS DIA (L)"] = dia
            spacing["LEFT SPACE STIRRUPS"] = spc
        if mid_vals:
            dia, spc, _ = min(mid_vals, key=lambda v: abs(v[2] - (beam_xmin + width/2)))
            spacing["SHEAR STIRRUPS DIA (M)"] = dia
            spacing["MID SPACE STIRRUPS"] = spc
        if right_vals:
            dia, spc, _ = min(right_vals, key=lambda v: abs(v[2] - beam_xmax))
            spacing["SHEAR STIRRUPS DIA (R)"] = dia
            spacing["RIGHT SPACE STIRRUPS"] = spc

    else:
        stirrup_matches.sort(key=lambda x: x[2])
        if len(stirrup_matches) == 2:
            left, right = stirrup_matches
            spacing["SHEAR STIRRUPS DIA (L)"] = left[0]
            spacing["LEFT SPACE STIRRUPS"] = left[1]
            spacing["SHEAR STIRRUPS DIA (R)"] = right[0]
            spacing["RIGHT SPACE STIRRUPS"] = right[1]
        elif len(stirrup_matches) >= 3:
            left, mid, right = stirrup_matches[0], stirrup_matches[1], stirrup_matches[-1]
            spacing["SHEAR STIRRUPS DIA (L)"] = left[0]
            spacing["LEFT SPACE STIRRUPS"] = left[1]
            spacing["SHEAR STIRRUPS DIA (M)"] = mid[0]
            spacing["MID SPACE STIRRUPS"] = mid[1]
            spacing["SHEAR STIRRUPS DIA (R)"] = right[0]
            spacing["RIGHT SPACE STIRRUPS"] = right[1]

    return spacing


def extract_stirrup_legs(results):
    leg_patterns = [
        re.compile(r"(\d+)\s*[-]?\s*leg", re.IGNORECASE),     # 2-leg, 2 leg, 4legs
        re.compile(r"\b(\d+)\s*L\b", re.IGNORECASE),          # 2L, 4L
        re.compile(r"(\d+)\s*-\s*\d+\(C\)", re.IGNORECASE),   # 2-16(C)
    ]

    for text, score, box in results:
        for lp in leg_patterns:
            m = lp.search(text)
            if m:
                try:
                    return {"SHEAR STIRUPPS LEG": int(m.group(1))}
                except:
                    continue
    return {"SHEAR STIRUPPS LEG": ""}


def extract_shear_stirrups(results):
    spacing = _extract_shear_stirrups_spacing(results)
    legs = extract_stirrup_legs(results)
    spacing.update(legs)
    return spacing

def _collect_value_points(results):
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
                    # if pd.isna(old.at[idx, col]) or old.at[idx, col] == "":
                    #     old.at[idx, col] = row.get(col, "")
                    if pd.isna(old.at[idx, col]) or old.at[idx, col] == "":
                        val = row.get(col, "")
                        # if val is empty string and column is numeric, set NaN instead
                        if val == "" and pd.api.types.is_numeric_dtype(old[col]):
                            old.at[idx, col] = pd.NA
                        else:
                            old.at[idx, col] = val
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


# if __name__ == "__main__":
#     image_path = "temp/preprocessed_for_ocr_28.png"  
#     process_image(image_path)

if __name__ == "__main__":
    folder_path = "temp" 
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, fname)
            print(f"\nProcessing: {fname}")
            try:
                process_image(image_path)
            except Exception as e:
                print(f"Error processing {fname}: {e}")