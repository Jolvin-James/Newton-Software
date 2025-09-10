import pandas as pd
import re
import os
import numpy as np
from statistics import median
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

def _natural_key(s: str):
    parts = re.split(r'(\d+)', s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

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

# def filter_results(ocr_results, score_threshold=0.7):
#     return [item for item in ocr_results if item[1] >= score_threshold]
def filter_results(ocr_results, score_threshold=0.6, debug=False):
    if ocr_results is None:
        return []
    filtered = [item for item in ocr_results if (item[1] or 0.0) >= score_threshold]
    if debug:
        print(f"[DEBUG] OCR raw count: {len(ocr_results)}, filtered (thr={score_threshold}): {len(filtered)}")
        # show first few entries for inspection
        for i, it in enumerate(ocr_results[:15]):
            txt, sc, box = it
            print(f"[DEBUG] raw[{i}]: text={repr(txt)}, score={sc}, box_len={len(box) if box is not None else 0}")
    return filtered

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

        wd = width_depth_pattern.search(text)
        if wd:
            w, d = map(int, wd.groups()); has_wd = True
        else:
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
                              beam_center_tol=2.0,
                              debug=False):
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
                count = count_raw
            try:
                cx, cy = _center(box)
            except Exception:
                # skip malformed box
                continue
            normalized = f"{count}-{dia}({pos.upper()})"
            bars.append({
                "text": normalized,
                "count": count,
                "dia": int(dia),
                "pos": pos.upper(),  # 'T' or 'C'
                "cx": cx, "cy": cy
            })

    if debug:
        print(f"[DEBUG] found {len(bars)} reinforcement bar tokens")

    if not bars:
        return {"LEFT TOP": "", "MID TOP": "", "RIGHT TOP": ""}

    ys = [b["cy"] for b in bars]
    xmin, xmax = min(b["cx"] for b in bars), max(b["cx"] for b in bars)
    ymin, ymax = min(ys), max(ys)
    y_range = ymax - ymin if ymax - ymin != 0 else 1.0

    # If only one bar, treat it as top (fallback)
    if len(bars) == 1:
        top_bars = bars.copy()
        if debug:
            print("[DEBUG] single bar detected, treat as top")
    else:
        # 1D two-cluster (k=2) iterative assignment on cy
        c1 = ymin
        c2 = ymax
        for _ in range(20):
            cluster1 = [b for b in bars if abs(b["cy"] - c1) <= abs(b["cy"] - c2)]
            cluster2 = [b for b in bars if abs(b["cy"] - c2) < abs(b["cy"] - c1)]
            new_c1 = sum(b["cy"] for b in cluster1) / len(cluster1) if cluster1 else c1
            new_c2 = sum(b["cy"] for b in cluster2) / len(cluster2) if cluster2 else c2
            if abs(new_c1 - c1) < 1e-3 and abs(new_c2 - c2) < 1e-3:
                cluster1, cluster2 = cluster1, cluster2
                break
            c1, c2 = new_c1, new_c2

        # choose the cluster with smaller mean cy as top
        center1 = sum(b["cy"] for b in cluster1) / len(cluster1) if cluster1 else float('inf')
        center2 = sum(b["cy"] for b in cluster2) / len(cluster2) if cluster2 else float('inf')
        top_cluster = cluster1 if center1 < center2 else cluster2
        other_cluster = cluster2 if top_cluster is cluster1 else cluster1
        center_dist = abs(center1 - center2)

        if debug:
            print(f"[DEBUG] top clustering centers: center1={center1:.1f}, center2={center2:.1f}, center_dist={center_dist:.1f}")

        # accept clustering only if cluster centers are sufficiently separated relative to y_range
        if center_dist >= (min_cluster_separation_ratio * y_range):
            top_bars = top_cluster
            if debug:
                print(f"[DEBUG] accepted two-cluster split, top cluster size: {len(top_bars)}")
        else:
            # fallback: take bars in top X% of the Y range
            cutoff = ymin + top_fraction_fallback * y_range
            top_bars = [b for b in bars if b["cy"] <= cutoff]
            if debug:
                print(f"[DEBUG] cluster separation too small, falling back to top_fraction cut (= {top_fraction_fallback}) -> chosen {len(top_bars)} bars")
            if not top_bars:
                # take the half with smaller cy values
                top_bars = sorted(bars, key=lambda b: b["cy"])[: max(1, len(bars)//2)]
                if debug:
                    print(f"[DEBUG] fallback to top half -> {len(top_bars)} bars")

    # If beam_center_y given, filter to above it (smaller cy).
    if beam_center_y is not None:
        top_filtered = [b for b in top_bars if b["cy"] <= (beam_center_y - beam_center_tol)]
        if top_filtered:
            if debug:
                print(f"[DEBUG] filtered top bars by beam_center_y -> {len(top_filtered)} remain (tol={beam_center_tol})")
            top_bars = top_filtered

    # assign to left/mid/right by x-position
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
            if r in ("LEFT", "RIGHT"):
                reinforcement[f"{r} TOP"].append(b["text"])

    # deduplicate per region
    for k in reinforcement:
        # keep order but remove duplicates
        seen = []
        for v in reinforcement[k]:
            if v not in seen:
                seen.append(v)
        reinforcement[k] = seen

    # If exactly one distinct T bar was detected, treat it as global top.
    distinct_t_texts = {b["text"] for b in top_bars if b["pos"] == "T"}
    if len(distinct_t_texts) == 1:
        only_t = next(iter(distinct_t_texts))
        if any(only_t in reinforcement[reg] for reg in reinforcement):
            for reg in reinforcement:
                if only_t not in reinforcement[reg]:
                    reinforcement[reg].append(only_t)
            if debug:
                print("[DEBUG] single distinct T bar found -> replicated across regions")

    # replicate near-mid top T bars across all three
    mid_x = (xmin + xmax) / 2.0
    mid_tol = width * 0.25
    for b in top_bars:
        if b["pos"] == "T" and abs(b["cx"] - mid_x) <= mid_tol:
            for reg in reinforcement:
                if b["text"] not in reinforcement[reg]:
                    reinforcement[reg].append(b["text"])

    # replicate all MID values into LEFT and RIGHT as well
    if reinforcement["MID TOP"]:
        mid_vals = reinforcement["MID TOP"][:]
        for val in mid_vals:
            if val not in reinforcement["LEFT TOP"]:
                reinforcement["LEFT TOP"].append(val)
            if val not in reinforcement["RIGHT TOP"]:
                reinforcement["RIGHT TOP"].append(val)

    # special case: if only T bars exist but they ended up only in one region, replicate them across all three
    if any(v for v in reinforcement.values()):
        filled = [k for k, v in reinforcement.items() if v]
        if len(filled) == 1:
            only_vals = reinforcement[filled[0]]
            reinforcement = {k: only_vals[:] for k in reinforcement.keys()}
            if debug:
                print("[DEBUG] values present in only one region -> replicated to all regions")

    # convert lists into comma-joined strings
    reinforcement = {k: " , ".join(v) if v else "" for k, v in reinforcement.items()}

    if debug:
        print(f"[DEBUG] final top reinforcement: {reinforcement}")

    return reinforcement


def extract_bottom_reinforcement(results, beam_center_y=None,
                                 bottom_fraction_fallback=0.40,
                                 beam_center_tol=5.0):

    bar_pattern = re.compile(r"(\+?\d+)\s*-\s*(\d+)\s*\(([TCB])\)", re.IGNORECASE)
    bars = []

    # parse all reinforcement notations
    for text, score, box in results:
        if not text:
            continue
        for m in bar_pattern.finditer(text):
            count_raw, dia, pos = m.groups()
            try:
                count = int(count_raw.replace("+", ""))
            except:
                count = count_raw
            cx, cy = _center(box)
            normalized = f"{count}-{dia}({pos.upper()})"
            bars.append({
                "text": normalized,
                "count": count,
                "dia": int(dia),
                "pos": pos.upper(),
                "cx": cx, "cy": cy
            })

    reinforcement = {"LEFT BOTTOM": [], "MID BOTTOM": [], "RIGHT BOTTOM": []}
    if not bars:
        return reinforcement

    # filter for bottom bars (below/near centerline)
    if beam_center_y is not None:
        bottom_bars = [b for b in bars if b["cy"] >= (beam_center_y - beam_center_tol)]
    else:
        bottom_bars = bars

    if not bottom_bars:
        ys = [b["cy"] for b in bars]
        cutoff = (max(ys) + min(ys)) / 2.0
        bottom_bars = [b for b in bars if b["cy"] >= cutoff]

    # geometry split
    xmin, xmax = min(b["cx"] for b in bottom_bars), max(b["cx"] for b in bottom_bars)
    width = xmax - xmin if xmax != xmin else 1.0
    mid_x = (xmin + xmax) / 2.0
    mid_tol = width * 0.10  # 10% of span considered "mid"

    # assign to regions based on rules
    for b in bottom_bars:
        if b["pos"] == "T":
            # replicate in all three regions
            for reg in reinforcement:
                if b["text"] not in reinforcement[reg]:
                    reinforcement[reg].append(b["text"])
        elif b["pos"] == "C":
            # only mid
            if b["text"] not in reinforcement["MID BOTTOM"]:
                reinforcement["MID BOTTOM"].append(b["text"])
        else:  # "B" or others → assign by geometry
            if abs(b["cx"] - mid_x) <= mid_tol:
                reinforcement["MID BOTTOM"].append(b["text"])
            elif b["cx"] < mid_x:
                reinforcement["LEFT BOTTOM"].append(b["text"])
            else:
                reinforcement["RIGHT BOTTOM"].append(b["text"])

    for k in reinforcement:
        reinforcement[k] = " , ".join(dict.fromkeys(reinforcement[k]))

    return reinforcement

def extract_top_left_right_dist(results, beam_center_y=None, debug=False):
    def _center(box):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return (float(np.mean(xs)), float(np.mean(ys)))

    # collect y-range and (T)-label y/xs to place a top band
    tbar_re = re.compile(r"\(T\)", re.IGNORECASE)
    ys, tbar_ys, tbar_xs = [], [], []
    for text, score, box in results:
        if text is None or box is None or len(box) == 0:
            continue
        cx, cy = _center(box)
        ys.append(cy)
        if tbar_re.search(str(text)):
            tbar_ys.append(cy)
            tbar_xs.append(cx)

    if not ys:
        return {"LEFT AT (DIST)": "", "RIGHT AT (DIST)": ""}

    y_min, y_max = min(ys), max(ys)
    y_span = max(1.0, y_max - y_min)

    # If (T) markers exist, make band narrow around their median
    if tbar_ys:
        tbar_ys.sort()
        mid_idx = len(tbar_ys) // 2
        band_center = (tbar_ys[mid_idx] if len(tbar_ys) % 2 == 1
                       else 0.5 * (tbar_ys[mid_idx - 1] + tbar_ys[mid_idx]))
        # make the band fairly tight around the (T) center to avoid bottom leakage
        band_half = 0.12 * y_span  # narrower than before
        band_top = max(y_min, band_center - band_half)
        band_bottom = min(y_max, band_center + band_half)
    else:
        # when no (T) labels - assume top region is near image top, but still stricter
        # earlier code used 0.35*y_span; reduce that to 0.28 for stricter behavior
        cutoff = y_min + 0.28 * y_span
        if beam_center_y is not None:
            band_top = y_min
            band_bottom = min(cutoff, beam_center_y - 0.06 * y_span)
        else:
            band_top, band_bottom = y_min, cutoff
        band_center = 0.5 * (band_top + band_bottom)

    # safety clamp: ensure band_bottom is not too low
    band_bottom = min(band_bottom, y_min + 0.40 * y_span)

    if debug:
        print("y_min,y_max,y_span:", y_min, y_max, y_span)
        print("band_top,bottom,center:", band_top, band_bottom, (band_top + band_bottom)/2.0)

    # numeric-only raw candidates (same regex logic)
    num_only = re.compile(r"^\s*(\d{3,5})\s*$")
    num_loose = re.compile(r"(?<!\d)(\d[\d\s]{2,6})(?!\d)")
    raw_cands = []
    for text, score, box in results:
        if text is None or box is None or len(box) == 0:
            continue
        t = str(text)
        m = num_only.match(t) or num_loose.search(t)
        if not m:
            continue
        val = int(re.sub(r"\s+", "", m.group(1)))
        if not (200 <= val <= 6000):
            continue
        cx, cy = _center(box)
        raw_cands.append((val, cx, cy, float(score or 0)))

    if not raw_cands:
        return {"LEFT AT (DIST)": "", "RIGHT AT (DIST)": ""}

    # keep only raw candidates inside the top band (strict)
    top_band_cands = [c for c in raw_cands if band_top <= c[2] <= band_bottom]

    # If nothing falls strictly inside the top band, allow a small tolerance upward (avoid picking bottom)
    if not top_band_cands:
        tol = 0.06 * y_span  # small tolerance
        top_band_cands = [c for c in raw_cands if (band_top - tol) <= c[2] <= (band_bottom + tol)]

    if not top_band_cands:
        y_cut = y_min + 0.30 * y_span
        top_band_cands = [c for c in raw_cands if c[2] <= y_cut]

    if not top_band_cands:
        return {"LEFT AT (DIST)": "", "RIGHT AT (DIST)": ""}

    # vertical clustering (group numerics into horizontal bands)
    top_band_cands.sort(key=lambda c: c[2])  # sort by cy
    clusters = []
    gap_thresh = 0.10 * y_span
    current = [top_band_cands[0]]
    for c in top_band_cands[1:]:
        if c[2] - current[-1][2] <= gap_thresh:
            current.append(c)
        else:
            clusters.append(current)
            current = [c]
    clusters.append(current)

    # compute cluster stats: mean_y and size
    cluster_info = [(sum(ci[2] for ci in cl) / len(cl), len(cl), cl) for cl in clusters]

    # pick cluster closest to band_center (prefer larger cluster on tie)
    cluster_info.sort(key=lambda x: (abs(x[0] - band_center), -x[1]))
    chosen_mean_y, chosen_size, chosen_cluster = cluster_info[0]

    # final safety: ensure chosen cluster is reasonably inside the top region
    if chosen_mean_y > band_top + 0.25 * y_span:
        # if cluster's mean is too low (i.e., likely bottom strip), pick the topmost cluster instead
        cluster_info.sort(key=lambda x: (x[0], -x[1]))
        chosen_cluster = cluster_info[0][2]

    cands = chosen_cluster

    if debug:
        print("clusters:", [(round(ci[0],1), ci[1]) for ci in cluster_info])
        print("chosen_mean_y, size:", chosen_mean_y, chosen_size)
        print("final candidate ys:", [round(c[2],1) for c in cands])

    xs = [cx for _, cx, _, _ in cands]
    xmin, xmax = min(xs), max(xs)
    width = max(1.0, xmax - xmin)

    # beam midline (from (T) labels if available, else candidates)
    if tbar_xs:
        mid_x = 0.5 * (min(tbar_xs) + max(tbar_xs))
        left_T = max([x for x in tbar_xs if x < mid_x], default=None)
        right_T = min([x for x in tbar_xs if x > mid_x], default=None)
    else:
        mid_x = 0.5 * (xmin + xmax)
        left_T, right_T = None, None

    # scoring & pick per side
    def pick_side(side):
        side_cands = [c for c in cands if (c[1] < mid_x if side == "left" else c[1] > mid_x)]
        if not side_cands:
            return ""

        anchor_x = left_T if side == "left" else right_T

        # if we have an anchor, check if any candidates live near it
        near_anchor = []
        if anchor_x is not None:
            near_anchor = [c for c in side_cands if abs(c[1] - anchor_x) <= 0.22 * width]

        use_edge_bias = (len(near_anchor) == 0)

        best_val, best_score = None, -1e9
        for val, cx, cy, sc in side_cands:
            s = -abs(cx - mid_x) / width  # closer to overall midline

            # strong attraction to the side's (T) anchor if present
            if anchor_x is not None:
                s += -0.90 * abs(cx - anchor_x) / width

            # vertical closeness to band center
            s += -0.20 * abs(cy - band_center) / y_span

            # edge rescue bias ONLY when nothing is near the anchor on this side
            if use_edge_bias:
                if side == "left":
                    s += 0.10 * (1.0 - (cx - xmin) / width)
                else:
                    s += 0.10 * ((cx - xmin) / width)

            # prefer round dimensions
            if val % 50 == 0:
                s += 0.15
            elif val % 25 == 0:
                s += 0.08

            # OCR confidence
            s += 0.05 * sc

            if s > best_score:
                best_score, best_val = s, val

        return best_val if best_val is not None else ""

    left_val = pick_side("left")
    right_val = pick_side("right")

    # robust fallbacks
    if left_val == "" or right_val == "":
        alt_mid = 0.5 * (xmin + xmax)
        if left_val == "":
            left_val = left_val or max([c for c in cands if c[1] < alt_mid], default=(None,))[0] or ""
        if right_val == "":
            right_val = right_val or max([c for c in cands if c[1] > alt_mid], default=(None,))[0] or ""

    return {"LEFT AT (DIST)": left_val or "", "RIGHT AT (DIST)": right_val or ""}

def extract_bottom_left_right_dist(results, beam_center_y=None, debug=False):
    import re
    from statistics import median

    def _center(box):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))

    strict_num = re.compile(r"^\s*(\d{2,5})\s*$")
    nxm = re.compile(r"\b\d+\s*[xX]\s*\d+\b")

    MIN_SCORE = 0.15
    MIN_VAL = 200
    MAX_VAL = 6000
    MIN_CENTER_SEP = 20.0
    MIN_X_SPAN = 60.0
    BOTTOM_BAND_RATIO = 0.60   

    cands = []

    for item in results:
        try:
            text, score, box = item
        except Exception:
            continue
        if text is None or box is None:
            continue
        t = str(text).strip()
        if nxm.search(t):
            continue
        m = strict_num.match(t)
        if not m:
            continue
        val = int(m.group(1))
        if not (MIN_VAL <= val <= MAX_VAL):
            continue
        try:
            cx, cy = _center(box)
        except Exception:
            continue
        s = 0.0 if score is None else float(score)
        if s < MIN_SCORE:
            continue
        cands.append({"val": val, "cx": cx, "cy": cy, "score": s})

    if not cands:
        return {"BOTTOM LEFT AT (DIST)": "", "BOTTOM RIGHT AT (DIST)": ""}

    ys = [c["cy"] for c in cands]
    y_min, y_max = min(ys), max(ys)
    y_span = max(1.0, y_max - y_min)
    bottom_threshold = y_min + (1.0 - BOTTOM_BAND_RATIO) * y_span
    bottom_cands = [c for c in cands if c["cy"] >= bottom_threshold]

    # fallback: if bottom band empty, use all candidates
    if not bottom_cands:
        bottom_cands = cands

    all_xs = sorted([c["cx"] for c in bottom_cands])
    mid_x = (min(all_xs) + max(all_xs)) / 2.0
    left_members = [c for c in bottom_cands if c["cx"] <= mid_x]
    right_members = [c for c in bottom_cands if c["cx"] > mid_x]

    if not left_members or not right_members:
        return {"BOTTOM LEFT AT (DIST)": "", "BOTTOM RIGHT AT (DIST)": ""}

    left_center = median([m["cx"] for m in left_members])
    right_center = median([m["cx"] for m in right_members])
    center_sep = abs(right_center - left_center)
    x_span = max(all_xs) - min(all_xs)

    if center_sep < MIN_CENTER_SEP or x_span < MIN_X_SPAN:
        return {"BOTTOM LEFT AT (DIST)": "", "BOTTOM RIGHT AT (DIST)": ""}

    def best_for_side(members, target_center):
        return min(members, key=lambda m: abs(m["cx"] - target_center))

    left_best = best_for_side(left_members, left_center)
    right_best = best_for_side(right_members, right_center)

    if left_best is right_best:
        return {"BOTTOM LEFT AT (DIST)": "", "BOTTOM RIGHT AT (DIST)": ""}

    return {
        "BOTTOM LEFT AT (DIST)": left_best["val"],
        "BOTTOM RIGHT AT (DIST)": right_best["val"]
    }


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

    # CASE 1: Only one spacing OR "ALL" --- mid only
    if len(stirrup_matches) == 1 or any("ALL" in s[3].upper() for s in stirrup_matches):
        dia, spc, _, _ = stirrup_matches[0]
        spacing["SHEAR STIRRUPS DIA (M)"] = dia
        spacing["MID SPACE STIRRUPS"] = spc
        return spacing

    # Assign based on beam thirds 
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
        # fallback: just use sorted order
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
    # ocr = init_ocr()
    raw = run_ocr(ocr, image_path)

    # print raw OCR size and sample immediately so we can see what's coming in
    print(f"[INFO] OCR returned {len(raw)} raw items for {image_path}")
    if DEBUG and len(raw) > 0:
        print("[DEBUG] sample raw OCR items:")
        for i, it in enumerate(raw[:12]):
            txt, sc, box = it
            print(f"  raw[{i}]: text={repr(txt)}, score={sc}, box_pts={len(box) if box is not None else 0}")

    # lower threshold while debugging; pass debug flag in so filter prints counts
    res = filter_results(raw, score_threshold=0.15, debug=DEBUG)

    # also show filtered sample
    if DEBUG:
        print(f"[DEBUG] filtered OCR items count: {len(res)}")
        for i, it in enumerate(res[:20]):
            txt, sc, box = it
            # safe check for box: explicit None + length check (works for lists and numpy arrays)
            try:
                has_points = (box is not None) and (hasattr(box, "__len__") and len(box) > 0)
            except Exception:
                has_points = False
            if has_points:
                try:
                    cx_cy = _center(box)
                except Exception:
                    cx_cy = (None, None)
            else:
                cx_cy = (None, None)
            print(f"  filtered[{i}]: text={repr(txt)}, score={sc}, cx_cy={cx_cy}")

    beam_candidates = extract_beam_numbers_and_sizes(res)
    if not beam_candidates:
        print("No beam numbers detected."); return

    # choose ONE beam for this image
    chosen = _select_primary_beam(beam_candidates, res)
    if not chosen:
        print("Could not select a primary beam."); return

    # pass debug flag through so the functions actually print debug info
    shear  = extract_shear_stirrups(res)
    top_reinf = extract_top_reinforcement(res, beam_center_y=chosen.get("_cy"), debug=DEBUG)
    bottom_reinf = extract_bottom_reinforcement(res, beam_center_y=chosen.get("_cy"))
    top_dist = extract_top_left_right_dist(res, beam_center_y=chosen.get("_cy"), debug=DEBUG)
    bottom_dist = extract_bottom_left_right_dist(res, beam_center_y=chosen.get("_cy"), debug=DEBUG)

    # build a single row for the chosen beam
    row = {**{h:"" for h in HEADERS}, **chosen, **shear, **top_reinf, **bottom_reinf, **top_dist, **bottom_dist}
    # strip helper keys
    row.pop("_cx", None); row.pop("_cy", None); row.pop("_has_wd", None)

    df_new = pd.DataFrame([row], columns=HEADERS)

    _merge_into_excel(df_new)
    print(f"Saved/updated: {row['BEAM NO']} -> {OUTPUT_XLSX}")


# if __name__ == "__main__":
#     image_path = "temp/preprocessed_for_ocr_27.png"  
#     process_image(image_path)

if __name__ == "__main__":
    ocr = init_ocr()
    folder_path = "temp" 
    imgs = [f for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    for fname in sorted(imgs, key=_natural_key):
        image_path = os.path.join(folder_path, fname)
        print(f"\nProcessing: {fname}")
        try:
            process_image(image_path)
        except Exception as e:
            print(f"Error processing {fname}: {e}")