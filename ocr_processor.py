import pytesseract
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import sys
import difflib

# ตั้งค่า Path ของ Tesseract (ถ้ามี)
DEFAULT_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(DEFAULT_TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT_PATH

def extract_text_from_pdf(pdf_path, lang='thai+eng'):
    """
    แกะข้อความจาก PDF (ใช้ OCR ถ้าจำเป็น) - ฟังก์ชันเดิม ไม่มีการเปลี่ยนแปลง
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

def get_lines_from_page(page):
    """
    ดึงข้อมูลเป็น 'บรรทัด' (Line Objects) โดยรวมกลุ่มคำที่อยู่ในระดับ Y ใกล้เคียงกัน
    Return: List of dict {'text': str, 'rect': fitz.Rect, 'words': list, 'matched': bool}
    """
    # ดึงคำทั้งหมด (x0, y0, x1, y1, "word", ...)
    words = page.get_text("words")
    
    # เรียงลำดับตาม Y (ปัดเศษเพื่อจัดกลุ่มบรรทัด) แล้วตาม X
    # การหาร 5 แล้วคูณ 5 คือการสร้าง Tolerance ประมาณ 5 pixel ในแนวตั้ง
    words.sort(key=lambda w: (round(w[1] / 5) * 5, w[0]))
    
    lines = []
    current_line_words = []
    
    for w in words:
        if not current_line_words:
            current_line_words.append(w)
            continue
        
        # เช็คว่าคำนี้อยู่บรรทัดเดียวกับคำก่อนหน้าหรือไม่ (ดูผลต่าง Y)
        last_w = current_line_words[-1]
        if abs(w[1] - last_w[1]) < 6: # Tolerance ความห่างบรรทัด
            current_line_words.append(w)
        else:
            # จบบรรทัดเดิม บันทึกลง list
            lines.append(create_line_obj(current_line_words))
            current_line_words = [w]
            
    # อย่าลืมบรรทัดสุดท้าย
    if current_line_words:
        lines.append(create_line_obj(current_line_words))
        
    return lines

def create_line_obj(word_list):
    """สร้าง Object บรรทัดจากรายการคำ"""
    # รวมข้อความ
    text = " ".join([w[4] for w in word_list]).strip()
    
    # สร้างกรอบสี่เหลี่ยมคลุมทั้งบรรทัด (Union Rects)
    r = fitz.Rect(word_list[0][:4])
    for w in word_list[1:]:
        r |= fitz.Rect(w[:4])
        
    return {
        'text': text,
        'rect': r,
        'words': word_list, # เก็บคำย่อยไว้เทียบ Diff ในระดับคำ
        'matched': False
    }

def similarity(s1, s2):
    """คำนวณความเหมือนของ String (0.0 - 1.0)"""
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def highlight_text_differences(pdf1_path, pdf2_path):
    """
    เปรียบเทียบเอกสารแบบ Smart Line Matching
    1. จับคู่บรรทัดที่เหมือนกัน (Similarity > 0.6)
    2. บรรทัดคู่กัน -> เทียบคำภายใน (Word Diff) -> ไฮไลท์เฉพาะคำที่แก้
    3. บรรทัดไม่มีคู่ -> ไฮไลท์ทั้งบรรทัด (แดง/เขียว)
    """
    if not os.path.exists(pdf1_path) or not os.path.exists(pdf2_path):
         return []

    try:
        doc1 = fitz.open(pdf1_path)
        doc2 = fitz.open(pdf2_path)
    except Exception as e:
        print(f"Error opening PDFs: {e}")
        return []

    images = []
    max_pages = max(len(doc1), len(doc2))

    for i in range(max_pages):
        page1 = doc1[i] if i < len(doc1) else None
        page2 = doc2[i] if i < len(doc2) else None
        
        # ดึงข้อมูลบรรทัด
        lines1 = get_lines_from_page(page1) if page1 else []
        lines2 = get_lines_from_page(page2) if page2 else []

        # จับคู่บรรทัด (Line Matching)
        # วนลูปบรรทัดใน Doc1 หาคู่ที่ดีที่สุดใน Doc2
        for l1 in lines1:
            best_match = None
            best_score = 0.0
            best_idx = -1
            
            for idx, l2 in enumerate(lines2):
                if l2["matched"]: continue # ข้ามบรรทัดที่มีคู่แล้ว
                
                score = similarity(l1["text"], l2["text"])
                if score > best_score:
                    best_score = score
                    best_match = l2
                    best_idx = idx
            
            # เกณฑ์การตัดสิน (Threshold): ถ้าเหมือนเกิน 60% ถือว่าเป็นบรรทัดเดียวกัน
            if best_score > 0.6:
                l1["matched"] = True
                lines2[best_idx]["matched"] = True
                
                # กรณีเจอคู่: เทียบคำภายในบรรทัด (Word Level Diff) 
                # ดึงเฉพาะ Text ของคำในบรรทัดนั้นมาเทียบ
                l1_words_str = [w[4] for w in l1['words']]
                l2_words_str = [w[4] for w in best_match['words']]
                
                matcher = difflib.SequenceMatcher(None, l1_words_str, l2_words_str)
                
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag == 'replace':
                        # แก้ไข: ไฮไลท์คำเดิม(แดง) และคำใหม่(เขียว)
                        if page1:
                            for k in range(i1, i2):
                                r = fitz.Rect(l1['words'][k][:4])
                                page1.draw_rect(r, color=(1, 0.5, 0.5), fill=(1, 0.5, 0.5), fill_opacity=0.35, width=0)
                        if page2:
                            for k in range(j1, j2):
                                r = fitz.Rect(best_match['words'][k][:4])
                                page2.draw_rect(r, color=(0.5, 1, 0.5), fill=(0.5, 1, 0.5), fill_opacity=0.35, width=0)
                                
                    elif tag == 'delete':
                        # ลบออก: ไฮไลท์แดงที่ Doc1
                        if page1:
                            for k in range(i1, i2):
                                r = fitz.Rect(l1['words'][k][:4])
                                page1.draw_rect(r, color=(1, 0.5, 0.5), fill=(1, 0.5, 0.5), fill_opacity=0.35, width=0)
                                
                    elif tag == 'insert':
                        # เพิ่มมา: ไฮไลท์เขียวที่ Doc2
                        if page2:
                            for k in range(j1, j2):
                                r = fitz.Rect(best_match['words'][k][:4])
                                page2.draw_rect(r, color=(0.5, 1, 0.5), fill=(0.5, 1, 0.5), fill_opacity=0.35, width=0)
            
            else:
                # --- กรณีหาคู่ไม่เจอ: แสดงว่าบรรทัดนี้ถูกลบหายไปทั้งบรรทัด ---
                if page1:
                    page1.draw_rect(l1["rect"], color=(1, 0.2, 0.2), fill=(1, 0.2, 0.2), fill_opacity=0.2, width=0)

        # 3. เก็บตกบรรทัดใน Doc2 ที่ไม่มีคู่ (แสดงว่าถูกเพิ่มเข้ามาทั้งบรรทัด)
        for l2 in lines2:
            if not l2["matched"]:
                if page2:
                    page2.draw_rect(l2["rect"], color=(0.2, 1, 0.2), fill=(0.2, 1, 0.2), fill_opacity=0.2, width=0)

        # 4. Render เป็นรูปภาพส่งกลับไปที่ GUI
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