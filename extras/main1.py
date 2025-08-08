import ezdxf
import openpyxl
import re

def extract_rcc_data(dxf_path, excel_path):
    # Load the DXF file
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # Excel workbook setup
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "RCC Beam Data"

    # Header row
    header_titles = [
        "BEAM NO", "WIDTH", "DEPTH", "LEVEL", "LEFT BOTTOM", "BOTTOM LEFT AT (DIST)", 
        "MID BOTTOM", "CURTAIL AT ( DIST )", "RIGHT BOTTOM", "BOTTOM RIGHT AT ( DIST )", 
        "BENT UP", "LEFT TOP", "LEFT AT ( DIST )", "MID TOP", "RIGHT TOP", "RIGHT AT ( DIST)", 
        "SFR", "SHEAR STIRUPPS LEG", "SHEAR STIRRUPS DIA ( L )", "LEFT SPACE STIRRUPS", 
        "SHEAR STIRRUPS DIA ( M )", "MID SPACE STIRRUPS", "SHEAR STIRRUPS DIA ( R )", 
        "RIGHT SPACE STIRRUPS", "SHEAR STIRRUP NUMBER", "EXTRA STIRRUP NUMBER", 
        "EXTRA STIRRUP DIA", "HORI LINK DIA", "STIRRUPSID CONTINUOUS END", 
        "DISCONTINUOUS END", "ATTACH MASTER ID"
    ]
    ws.append(header_titles)

    # Regex pattern for Beam numbers like B501a, B502, etc.
    beam_no_pattern = re.compile(r"\bB\d{3}[a-zA-Z]?\b")

    # Scan through all entities
    for entity in msp:
        text_content = None

        if entity.dxftype() == "TEXT":
            text_content = entity.dxf.text.strip()

        elif entity.dxftype() == "MTEXT":
            text_content = entity.text.strip()

        # If there's text and it matches the beam number pattern
        if text_content:
            match = beam_no_pattern.search(text_content)
            if match:
                beam_no = match.group(0).upper()
                row = [""] * len(header_titles)
                row[0] = beam_no
                ws.append(row)

    # Save the result
    wb.save(excel_path)
    print(f"Beam numbers extracted and saved to: {excel_path}")

# --- Usage ---
dxf_file_path = "TRIAL1_BS.dxf"
excel_file_path = "RCC_Beam_Data.xlsx"
extract_rcc_data(dxf_file_path, excel_file_path)
