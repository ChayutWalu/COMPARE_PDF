import pytesseract
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import io
import os
import sys
import difflib

# Attempt to locate tesseract executable if not in PATH
DEFAULT_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(DEFAULT_TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT_PATH

def extract_text_from_pdf(pdf_path, lang='thai+eng'):
    """
    Extracts text from a PDF file using OCR (via PyMuPDF and Tesseract).
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    print(f"Opening PDF: {pdf_path}...")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF with PyMuPDF: {e}")
        raise e

    print(f"Extracting text from {len(doc)} pages...")
    extracted_text = ""
    
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            extracted_text += f"--- Page {i+1} (Native) ---\n{text}\n"
            continue
            
        print(f"No text found on page {i+1}, attempting OCR...")
        try:
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            text = pytesseract.image_to_string(image, lang=lang)
            extracted_text += f"--- Page {i+1} (OCR) ---\n{text}\n"
        except Exception as e:
            print(f"Error processing page {i+1}: {e}")
            extracted_text += f"--- Page {i+1} (Error) ---\n[Text extraction failed]\n"

    return extracted_text

def merge_rectangles(rects, x_tolerance=5, y_tolerance=3):
    """
    รวมสี่เหลี่ยมที่อยู่ใกล้กันให้เป็นก้อนเดียว เพื่อลดความรกของหน้าจอ
    """
    if not rects:
        return []

    # เรียงลำดับตามแกน Y และ X
    rects.sort(key=lambda r: (r.y0, r.x0))
    
    merged = []
    current_rect = rects[0]

    for next_rect in rects[1:]:
        # ตรวจสอบว่าอยู่ในบรรทัดเดียวกัน (Y ใกล้เคียงกัน) และ X ต่อเนื่องกัน
        vertical_overlap = max(0, min(current_rect.y1, next_rect.y1) - max(current_rect.y0, next_rect.y0))
        line_height = min(current_rect.y1 - current_rect.y0, next_rect.y1 - next_rect.y0)
        
        is_same_line = vertical_overlap > (line_height * 0.5) # ซ้อนทับกันเกิน 50% ของความสูง
        is_nearby_x = (next_rect.x0 - current_rect.x1) <= x_tolerance

        if is_same_line and is_nearby_x:
            # รวม rect (union)
            current_rect = current_rect | next_rect # fitz.Rect รองรับ bitwise OR เพื่อ merge
        else:
            merged.append(current_rect)
            current_rect = next_rect
            
    merged.append(current_rect)
    return merged

def get_sorted_words(page):
    """
    ดึงคำจากหน้า PDF และเรียงลำดับใหม่ให้ถูกต้อง (บน->ล่าง, ซ้าย->ขวา)
    """
    words = page.get_text("words") # (x0, y0, x1, y1, "word", block_no, line_no, word_no)
    # Sort key: 
    # 1. ปัดเศษ Y (round y0) เพื่อจัดกลุ่มบรรทัดเดียวกัน (tolerance 2-3 pixel)
    # 2. x0 เพื่อเรียงจากซ้ายไปขวา
    words.sort(key=lambda w: (round(w[1] / 3) * 3, w[0]))
    return words

def highlight_text_differences(pdf1_path, pdf2_path):
    if not os.path.exists(pdf1_path) or not os.path.exists(pdf2_path):
         return []

    try:
        doc1 = fitz.open(pdf1_path)
        doc2 = fitz.open(pdf2_path)
    except Exception as e:
        print(f"Error opening PDFs for visual diff: {e}")
        return []

    images = []
    max_pages = max(len(doc1), len(doc2))

    for i in range(max_pages):
        # --- เตรียมข้อมูล Page 1 ---
        words1_info = [] 
        words1_strings = [] 
        page1 = None
        
        if i < len(doc1):
            page1 = doc1[i]
            words1_info = get_sorted_words(page1)
            # Normalization: strip และ lower (ถ้าต้องการ)
            words1_strings = [w[4].strip() for w in words1_info]

        # --- เตรียมข้อมูล Page 2 ---
        words2_info = []
        words2_strings = []
        page2 = None
        
        if i < len(doc2):
            page2 = doc2[i]
            words2_info = get_sorted_words(page2)
            words2_strings = [w[4].strip() for w in words2_info]

        # --- เปรียบเทียบด้วย Difflib ---
        matcher = difflib.SequenceMatcher(None, words1_strings, words2_strings)
        
        # เก็บ Rect ที่จะวาดแยกตามสีก่อน (ยังไม่วาดทันที เพื่อเอาไป merge)
        red_rects = []   # Delete / Modify (Old)
        green_rects = [] # Insert / Modify (New)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                if page1:
                    for k in range(i1, i2):
                        red_rects.append(fitz.Rect(words1_info[k][:4]))
                if page2:
                    for k in range(j1, j2):
                        green_rects.append(fitz.Rect(words2_info[k][:4]))

            elif tag == 'delete':
                if page1:
                    for k in range(i1, i2):
                        red_rects.append(fitz.Rect(words1_info[k][:4]))

            elif tag == 'insert':
                if page2:
                    for k in range(j1, j2):
                        green_rects.append(fitz.Rect(words2_info[k][:4]))

        # --- Merge Rects & Draw ---
        # รวมกล่องที่อยู่ติดกันให้ดูสะอาดตา
        red_rects = merge_rectangles(red_rects)
        green_rects = merge_rectangles(green_rects)

        if page1:
            for r in red_rects:
                page1.draw_rect(r, color=(1, 0.3, 0.3), fill=(1, 0.3, 0.3), fill_opacity=0.3, width=0)
                
        if page2:
            for r in green_rects:
                page2.draw_rect(r, color=(0.3, 1, 0.3), fill=(0.3, 1, 0.3), fill_opacity=0.3, width=0)

        # --- Render เป็นรูปภาพ ---
        img1 = None
        img2 = None
        
        if page1:
            pix = page1.get_pixmap(dpi=150)
            img1 = Image.open(io.BytesIO(pix.tobytes("png")))
        
        if page2:
            pix = page2.get_pixmap(dpi=150)
            img2 = Image.open(io.BytesIO(pix.tobytes("png")))
            
        images.append((img1, img2))

    return images

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(extract_text_from_pdf(sys.argv[1]))
    else:
        print("Usage: python ocr_processor.py <pdf_path>")