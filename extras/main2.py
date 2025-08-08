import ezdxf
import openpyxl
import re
import math

def extract_rcc_data(dxf_path, excel_path, proximity_tol=100, debug=True):
    # 1) Load DXF and collect all TEXT/MTEXT labels + DIMENSION overrides
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    labels = []  # (text, (x,y))

    for e in msp:
        t = e.dxftype()
        # plain TEXT
        if t == "TEXT":
            txt = e.dxf.text.strip()
            x, y, _ = e.dxf.insert
            labels.append((txt, (x, y)))
        # multiline text
        elif t == "MTEXT":
            txt = e.text.strip()
            x, y, _ = e.dxf.insert
            labels.append((txt, (x, y)))
        # dimension entity override
        elif "DIMENSION" in t:
            val = e.dxf.text.strip()
            if not val or val == "<>":
                try:
                    meas = e.get_measurement()
                    val = str(int(meas))
                except Exception:
                    val = str(e.get_measurement())
            # anchor at defpoint (text location)
            try:
                x, y, _ = e.dxf.defpoint
            except:
                x, y, _ = e.dxf.insert
            labels.append((val, (x, y)))

    if debug:
        print(f"🔍 Collected {len(labels)} text/dim labels from '{dxf_path}'")
        # print first 10 for sanity
        for txt, (x, y) in labels[:10]:
            print(f"   → '{txt}' @ ({x:.1f}, {y:.1f})")

    # 2) Prep Excel
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "RCC Beam Data"
    ws.append(["BEAM NO", "WIDTH", "DEPTH"])  # extend as needed

    # 3) Regex (case‑insensitive)
    beam_re = re.compile(r"\bB\d{3}[A-Za-z]?\b", re.IGNORECASE)
    dim_re  = re.compile(r"(\d+)\s*[xXx]\s*(\d+)")

    # 4) Match BEAM → closest DIM
    for txt, (bx, by) in labels:
        bm = beam_re.search(txt)
        if not bm:
            continue
        beam_no = bm.group(0).upper()

        best_w, best_d, best_dist = None, None, proximity_tol + 1
        for lbl, (tx, ty) in labels:
            dm = dim_re.search(lbl)
            if not dm:
                continue
            w, d = dm.group(1), dm.group(2)
            dist = math.hypot(tx - bx, ty - by)
            if dist < best_dist and dist <= proximity_tol:
                best_w, best_d, best_dist = w, d, dist

        if debug:
            if best_w:
                print(f"✅ {beam_no}: matched {best_w}x{best_d} at {best_dist:.1f}")
            else:
                print(f"⚠️  {beam_no}: no dim within {proximity_tol}")

        ws.append([beam_no, best_w or "", best_d or ""])

    # 5) Save
    wb.save(excel_path)
    print(f"💾 Saved {ws.max_row - 1} beams to '{excel_path}'")


if __name__ == "__main__":
    extract_rcc_data(
        dxf_path="TRIAL1_BS.dxf",
        excel_path="RCC_Beam_Data.xlsx",
        proximity_tol=100,
        debug=True
    )
