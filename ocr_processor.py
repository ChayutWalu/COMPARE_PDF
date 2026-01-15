import pytesseract
import fitz  # PyMuPDF
from PIL import Image
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
        # Try native extraction first
        text = page.get_text()
        if text.strip():
            extracted_text += f"--- Page {i+1} (Native) ---\n{text}\n"
            continue
            
        # Fallback to OCR
        print(f"No text found on page {i+1}, attempting OCR...")
        try:
            # Render page to image
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            
            # Perform OCR
            text = pytesseract.image_to_string(image, lang=lang)
            extracted_text += f"--- Page {i+1} (OCR) ---\n{text}\n"
        except Exception as e:
            print(f"Error processing page {i+1}: {e}")
            if "tesseract" in str(e).lower():
                print("Tesseract not found. Please install Tesseract-OCR.")
                # Don't raise, just continue with empty text for this page if OCR fails but maybe other pages worked? 
                # Or just append a warning.
                extracted_text += f"--- Page {i+1} (Error) ---\n[Text extraction failed: Tesseract not found]\n"
            else:
                raise e

    return extracted_text

def highlight_text_differences(pdf1_path, pdf2_path):
    """
    Generates images of PDF pages with highlighted text differences.
    Returns a list of tuples: [(pil_image1, pil_image2), ...]
    Using 'words' extraction to get exact coordinates avoiding duplicate highlight issues.
    """
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
        words1_info = [] # เก็บ (x0, y0, x1, y1, word, ...)
        words1_strings = [] # เก็บเฉพาะคำ string เพื่อเอาไป diff
        page1 = None
        
        if i < len(doc1):
            page1 = doc1[i]
            # get_text("words") คืนค่า: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
            words1_info = page1.get_text("words")
            words1_strings = [w[4] for w in words1_info]

        # --- เตรียมข้อมูล Page 2 ---
        words2_info = []
        words2_strings = []
        page2 = None
        
        if i < len(doc2):
            page2 = doc2[i]
            words2_info = page2.get_text("words")
            words2_strings = [w[4] for w in words2_info]

        # --- เปรียบเทียบด้วย Difflib ---
        matcher = difflib.SequenceMatcher(None, words1_strings, words2_strings)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                # Highlight ของเดิมใน Doc 1 (สีแดง)
                if page1:
                    for k in range(i1, i2):
                        # ดึงพิกัดจาก index k โดยตรง (ไม่ต้อง search)
                        rect = fitz.Rect(words1_info[k][:4])
                        page1.draw_rect(rect, color=(1, 0.5, 0.5), fill=(1, 0.5, 0.5), fill_opacity=0.35, width=0)
                
                # Highlight ของใหม่ใน Doc 2 (สีเขียว)
                if page2:
                    for k in range(j1, j2):
                        rect = fitz.Rect(words2_info[k][:4])
                        page2.draw_rect(rect, color=(0.5, 1, 0.5), fill=(0.5, 1, 0.5), fill_opacity=0.35, width=0)

            elif tag == 'delete':
                # ของที่หายไป Highlight ใน Doc 1 (สีแดง)
                if page1:
                    for k in range(i1, i2):
                        rect = fitz.Rect(words1_info[k][:4])
                        page1.draw_rect(rect, color=(1, 0.5, 0.5), fill=(1, 0.5, 0.5), fill_opacity=0.35, width=0)

            elif tag == 'insert':
                # ของที่เพิ่มมา Highlight ใน Doc 2 (สีเขียว)
                if page2:
                    for k in range(j1, j2):
                        rect = fitz.Rect(words2_info[k][:4])
                        page2.draw_rect(rect, color=(0.5, 1, 0.5), fill=(0.5, 1, 0.5), fill_opacity=0.35, width=0)

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


