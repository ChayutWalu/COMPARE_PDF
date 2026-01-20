import easyocr
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import sys
import difflib
import re
import numpy as np

# --- 1. ตั้งค่าคำทั่วไปที่ไม่ต้องการให้ไฮไลท์ (Stopwords) ---
# คุณสามารถเพิ่มคำที่เจอบ่อยๆ ในเอกสารแต่ไม่อยากให้ไฮไลท์ได้ที่นี่
STOPWORDS = {
    # ภาษาไทย
    "นาย", "นาง", "นางสาว", "เด็กชาย", "เด็กหญิง", "บริษัท", "จํากัด", "มหาชน", 
    "ถนน", "ซอย", "แขวง", "เขต", "จังหวัด", "อำเภอ", "ตำบล", 
    "วันที่", "เดือน", "พ.ศ.", "เลขที่", "หมู่", "ราคา", "บาท", "สตางค์",
    "กรมธรรม์", "ประกันภัย", "ผู้เอาประกัน", "ที่อยู่", "เบอร์โทร", 
    "โทร", "แฟกซ์", "รายละเอียด", "จำนวน", "รวม", "ภาษี", "อากร",
    # English
    "mr", "mrs", "miss", "ms", "company", "ltd", "public", "limited",
    "road", "soi", "district", "province", "date", "month", "year", "no",
    "price", "baht", "policy", "insurance", "insured", "address", "tel", "fax",
    "detail", "amount", "total", "tax", "vat", "sum", "premium", "copy", "original"
}

print("Initializing EasyOCR with GPU...")
reader = easyocr.Reader(['th', 'en'], gpu=True) 

def extract_text_from_pdf(pdf_path):
    """แกะข้อความจาก PDF (ใช้ EasyOCR) - สำหรับ LLM"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    print(f"Opening PDF: {pdf_path}...")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF with PyMuPDF: {e}")
        raise e

    extracted_text = ""
    for i, page in enumerate(doc):
        # 1. ลองดึง Text แบบ Native ก่อน
        text = page.get_text()
        if text.strip():
            extracted_text += f"--- Page {i+1} (Native) ---\n{text}\n"
            continue
            
        # 2. ถ้าไม่มี Text ให้ใช้ EasyOCR
        print(f"No text layer on page {i+1}, running EasyOCR...")
        try:
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            img_np = np.array(image)
            result_list = reader.readtext(img_np, detail=0, paragraph=True)
            ocr_text = "\n".join(result_list)
            extracted_text += f"--- Page {i+1} (EasyOCR) ---\n{ocr_text}\n"
        except Exception as e:
            extracted_text += f"--- Page {i+1} (Error) ---\n[Text extraction failed]\n"

    return extracted_text

def get_words_from_page_ocr(page):
    """ดึงคำและพิกัดด้วย EasyOCR"""
    rotation = page.rotation
    pix = page.get_pixmap(dpi=300) 
    img_data = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_data))
    img_np = np.array(image)
    
    results = reader.readtext(img_np)
    words = []
    
    if rotation in [90, 270]:
        scale_x = page.rect.width / pix.width
        scale_y = page.rect.height / pix.height
    else:
        scale_x = page.rect.width / pix.width
        scale_y = page.rect.height / pix.height

    for (bbox, text, prob) in results:
        if prob > 0.2 and text.strip():
            (tl, tr, br, bl) = bbox
            x_min = min(tl[0], bl[0])
            y_min = min(tl[1], tr[1])
            x_max = max(tr[0], br[0])
            y_max = max(bl[1], br[1])
            
            x0 = x_min * scale_x
            y0 = y_min * scale_y
            x1 = x_max * scale_x
            y1 = y_max * scale_y
            
            words.append((x0, y0, x1, y1, text, 0, 0, 0))
            
    return words

def get_lines_from_page(page):
    """ดึงบรรทัด"""
    words = page.get_text("words")
    if len(words) < 5: 
        try:
            words = get_words_from_page_ocr(page)
        except Exception as e:
            words = []

    words.sort(key=lambda w: (round(w[1] / 8) * 8, w[0]))
    
    lines = []
    current_line_words = []
    
    for w in words:
        if not current_line_words:
            current_line_words.append(w)
            continue
        if abs(w[1] - current_line_words[-1][1]) < 10:
            current_line_words.append(w)
        else:
            lines.append(create_line_obj(current_line_words))
            current_line_words = [w]
            
    if current_line_words:
        lines.append(create_line_obj(current_line_words))
    return lines

def create_line_obj(word_list):
    text = " ".join([w[4] for w in word_list]).strip()
    r = fitz.Rect(word_list[0][:4])
    for w in word_list[1:]:
        r |= fitz.Rect(w[:4])
    return {'text': text, 'rect': r, 'words': word_list, 'matched': False}

def clean_text(text):
    """ลบอักขระพิเศษ"""
    # เก็บ ก-ฮ, สระ, วรรณยุกต์, 0-9, a-z
    text = re.sub(r'[^\w\u0E00-\u0E7F]', '', text) 
    return text.lower()

# --- 2. ฟังก์ชันตรวจสอบว่าเป็น "สาระสำคัญ" หรือไม่ (Key Function) ---
def is_significant(text):
    """
    เช็คว่าคำนี้ควรค่าแก่การเอาไปเทียบหรือไม่
    Return: True ถ้าเป็นคำสำคัญ, False ถ้าเป็นคำทั่วไป
    """
    clean = clean_text(text)
    
    # ถ้าสั้นเกินไป (น้อยกว่า 2 ตัวอักษร) และไม่ใช่ตัวเลข -> ไม่เอา
    if len(clean) < 2 and not clean.isdigit():
        return False
        
    # ถ้าเป็น Stopwords -> ไม่เอา
    if clean in STOPWORDS:
        return False
        
    # ถ้าเป็นตัวเลข (Money, ID, Date) -> เอาแน่นอน (สำคัญมาก)
    if any(char.isdigit() for char in clean):
        return True
        
    # ถ้าไม่ใช่ตัวเลข แต่ยาวพอสมควร (เช่น ชื่อคน) -> เอา
    if len(clean) >= 3:
        return True
        
    return False

def is_similar_word(w1, w2):
    """Fuzzy Match Logic"""
    c1 = clean_text(w1)
    c2 = clean_text(w2)
    
    if not c1 or not c2: return False
    
    # ต้องเป็นคำสำคัญทั้งคู่ถึงจะเริ่มเทียบ
    if not is_significant(w1) or not is_significant(w2):
        return False

    if c1 == c2: return True
    if len(c1) > 2 and len(c2) > 2:
        if c1 in c2 or c2 in c1: return True

    ratio = difflib.SequenceMatcher(None, c1, c2).ratio()
    if ratio > 0.75: return True
    
    return False

def highlight_text_differences(pdf1_path, pdf2_path, mode='diff'):
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
        
        lines1 = get_lines_from_page(page1) if page1 else []
        lines2 = get_lines_from_page(page2) if page2 else []

        if mode == 'same':
            # --- Logic ใหม่: กรองเฉพาะคำสำคัญก่อนเทียบ ---
            
            # Flatten & Filter Words from Doc 2
            # เก็บเฉพาะคำที่ Significant ไว้ใน List ของ Doc 2
            significant_words2 = []
            for l in lines2:
                for w in l['words']:
                    if is_significant(w[4]): # เช็คว่าเป็นคำสำคัญไหม
                        significant_words2.append(w)

            # Flatten Words from Doc 1
            words1_flat = [w for l in lines1 for w in l['words']]

            # Check Doc 1 words against Significant Doc 2 words
            if page1 and significant_words2:
                for w1 in words1_flat:
                    # ถ้า w1 เองไม่ใช่คำสำคัญ ก็ข้ามไปเลย (ไม่ต้องเสียเวลาเทียบ)
                    if not is_significant(w1[4]): 
                        continue
                        
                    found = False
                    for w2 in significant_words2:
                        # เทียบเฉพาะกับคำสำคัญของ Doc 2
                        if is_similar_word(w1[4], w2[4]):
                            found = True
                            break
                    if found:
                        r = fitz.Rect(w1[:4])
                        page1.draw_rect(r, color=(0, 1, 0), fill=(0, 1, 0), fill_opacity=0.35, width=0)

            # --- ทำกลับกันสำหรับ Doc 2 ---
            
            significant_words1 = []
            for l in lines1:
                for w in l['words']:
                    if is_significant(w[4]):
                        significant_words1.append(w)
            
            words2_flat = [w for l in lines2 for w in l['words']]
            
            if page2 and significant_words1:
                for w2 in words2_flat:
                    if not is_significant(w2[4]):
                        continue
                        
                    found = False
                    for w1 in significant_words1:
                        if is_similar_word(w2[4], w1[4]):
                            found = True
                            break
                    if found:
                        r = fitz.Rect(w2[:4])
                        page2.draw_rect(r, color=(0, 1, 0), fill=(0, 1, 0), fill_opacity=0.35, width=0)

        else:
            # Mode Diff (ใช้ Logic เดิม)
            for l1 in lines1:
                best_match = None
                best_score = 0.0
                best_idx = -1
                for idx, l2 in enumerate(lines2):
                    if l2["matched"]: continue
                    score = difflib.SequenceMatcher(None, l1["text"], l2["text"]).ratio()
                    if score > best_score:
                        best_score = score
                        best_match = l2
                        best_idx = idx
                
                if best_score > 0.6:
                    l1["matched"] = True
                    lines2[best_idx]["matched"] = True
                    l1_words_str = [w[4] for w in l1['words']]
                    l2_words_str = [w[4] for w in best_match['words']]
                    matcher = difflib.SequenceMatcher(None, l1_words_str, l2_words_str)
                    
                    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                        if tag == 'replace':
                            if page1:
                                for k in range(i1, i2):
                                    if k < len(l1['words']):
                                        r = fitz.Rect(l1['words'][k][:4])
                                        page1.draw_rect(r, color=(1, 0.5, 0.5), fill=(1, 0.5, 0.5), fill_opacity=0.5, width=0)
                            if page2:
                                for k in range(j1, j2):
                                    if k < len(best_match['words']):
                                        r = fitz.Rect(best_match['words'][k][:4])
                                        page2.draw_rect(r, color=(1, 0.8, 0.2), fill=(1, 0.8, 0.2), fill_opacity=0.5, width=0)
                        elif tag == 'delete':
                            if page1:
                                for k in range(i1, i2):
                                    if k < len(l1['words']):
                                        r = fitz.Rect(l1['words'][k][:4])
                                        page1.draw_rect(r, color=(1, 0.2, 0.2), fill=(1, 0.2, 0.2), fill_opacity=0.5, width=0)
                        elif tag == 'insert':
                            if page2:
                                for k in range(j1, j2):
                                    if k < len(best_match['words']):
                                        r = fitz.Rect(best_match['words'][k][:4])
                                        page2.draw_rect(r, color=(0.2, 1, 0.2), fill=(0.2, 1, 0.2), fill_opacity=0.5, width=0)
                else:
                    if page1:
                        page1.draw_rect(l1["rect"], color=(1, 0.2, 0.2), fill=(1, 0.2, 0.2), fill_opacity=0.2, width=0)

            for l2 in lines2:
                if not l2["matched"] and page2:
                    page2.draw_rect(l2["rect"], color=(0.2, 1, 0.2), fill=(0.2, 1, 0.2), fill_opacity=0.2, width=0)

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