import easyocr
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import sys
import difflib
import re
import numpy as np

# --- 1. STOPWORDS (คำที่จะไม่ไฮไลท์ในโหมด Same) ---
STOPWORDS = {
    "นาย", "นาง", "นางสาว", "เด็กชาย", "เด็กหญิง", "บริษัท", "จํากัด", "มหาชน", 
    "ถนน", "ซอย", "แขวง", "เขต", "จังหวัด", "อำเภอ", "ตำบล", 
    "วันที่", "เดือน", "พ.ศ.", "เลขที่", "หมู่", "ราคา", "บาท", "สตางค์",
    "กรมธรรม์", "ประกันภัย", "ผู้เอาประกัน", "ที่อยู่", "เบอร์โทร", 
    "โทร", "แฟกซ์", "รายละเอียด", "จำนวน", "รวม", "ภาษี", "อากร", "หมายเหตุ",
    "mr", "mrs", "miss", "ms", "company", "ltd", "public", "limited",
    "road", "soi", "district", "province", "date", "month", "year", "no",
    "price", "baht", "policy", "insurance", "insured", "address", "tel", "fax",
    "detail", "amount", "total", "tax", "vat", "sum", "premium", "copy", "original"
}

print("Initializing EasyOCR with GPU...")
reader = easyocr.Reader(['th', 'en'], gpu=True) 

def extract_text_from_pdf(pdf_path):
    """แกะข้อความสำหรับ LLM"""
    if not os.path.exists(pdf_path): return ""
    try:
        doc = fitz.open(pdf_path)
    except: return ""

    extracted_text = ""
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            extracted_text += f"--- Page {i+1} (Native) ---\n{text}\n"
            continue
            
        try:
            pix = page.get_pixmap(dpi=300)
            img_np = np.array(Image.open(io.BytesIO(pix.tobytes("png"))))
            result_list = reader.readtext(img_np, detail=0, paragraph=True)
            extracted_text += f"--- Page {i+1} (EasyOCR) ---\n{chr(10).join(result_list)}\n"
        except: pass
    return extracted_text

def get_words_from_page_ocr(page):
    """ดึงคำและพิกัดด้วย EasyOCR"""
    rotation = page.rotation
    pix = page.get_pixmap(dpi=300) 
    img_np = np.array(Image.open(io.BytesIO(pix.tobytes("png"))))
    
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

def get_words_all(page):
    """ฟังก์ชันรวม: ดึงคำทั้งหมดจากหน้า (ไม่สนบรรทัด)"""
    words = page.get_text("words")
    if len(words) < 5: 
        try:
            words = get_words_from_page_ocr(page)
        except: words = []
    
    # เรียงลำดับคำตามการอ่าน (บน->ล่าง, ซ้าย->ขวา)
    # y/10 เพื่อให้คำที่อยู่ในบรรทัดเดียวกัน (แม้ y ต่างกันนิดหน่อย) ถูกมองว่าบรรทัดเดียวกัน
    words.sort(key=lambda w: (round(w[1] / 10) * 10, w[0]))
    return words

def clean_text(text):
    """ลบอักขระพิเศษ แต่เก็บตัวเลขและภาษาไทย"""
    text = re.sub(r'[^\w\u0E00-\u0E7F]', '', text) 
    return text.lower()

def is_significant(text):
    clean = clean_text(text)
    if len(clean) < 2 and not clean.isdigit(): return False
    if clean in STOPWORDS: return False
    if any(char.isdigit() for char in clean): return True
    if len(clean) >= 3: return True
    return False

def is_similar_word(w1, w2):
    c1 = clean_text(w1)
    c2 = clean_text(w2)
    if not c1 or not c2: return False
    if not is_significant(w1) or not is_significant(w2): return False
    if c1 == c2: return True
    if len(c1) > 2 and len(c2) > 2:
        if c1 in c2 or c2 in c1: return True
    return difflib.SequenceMatcher(None, c1, c2).ratio() > 0.75

def highlight_text_differences(pdf1_path, pdf2_path, mode='diff'):
    if not os.path.exists(pdf1_path) or not os.path.exists(pdf2_path): return []

    try:
        doc1 = fitz.open(pdf1_path)
        doc2 = fitz.open(pdf2_path)
    except Exception as e:
        print(f"Error: {e}")
        return []

    images = []
    max_pages = max(len(doc1), len(doc2))

    for i in range(max_pages):
        page1 = doc1[i] if i < len(doc1) else None
        page2 = doc2[i] if i < len(doc2) else None
        
        # ดึงคำทั้งหมดออกมาเป็น List เดียว (Word Stream) ไม่สนบรรทัด
        words1 = get_words_all(page1) if page1 else []
        words2 = get_words_all(page2) if page2 else []

        # ==================== MODE: SAME (Fuzzy Match) ====================
        if mode == 'same':
            # (Logic เดิมที่ดีอยู่แล้ว)
            sig_words2 = [w for w in words2 if is_significant(w[4])]
            
            if page1 and sig_words2:
                for w1 in words1:
                    if not is_significant(w1[4]): continue
                    for w2 in sig_words2:
                        if is_similar_word(w1[4], w2[4]):
                            r = fitz.Rect(w1[:4])
                            page1.draw_rect(r, color=(0, 1, 0), fill=(0, 1, 0), fill_opacity=0.35, width=0)
                            break # เจอแล้วหยุด
            
            sig_words1 = [w for w in words1 if is_significant(w[4])]
            if page2 and sig_words1:
                for w2 in words2:
                    if not is_significant(w2[4]): continue
                    for w1 in sig_words1:
                        if is_similar_word(w2[4], w1[4]):
                            r = fitz.Rect(w2[:4])
                            page2.draw_rect(r, color=(0, 1, 0), fill=(0, 1, 0), fill_opacity=0.35, width=0)

        # ==================== MODE: DIFF (Content Sequence Match) ====================
        else:
            # 1. เตรียมข้อมูล String สำหรับเทียบ (ใช้ clean_text เพื่อลด noise)
            str1 = [clean_text(w[4]) for w in words1]
            str2 = [clean_text(w[4]) for w in words2]
            
            # 2. ใช้ SequenceMatcher เทียบ List ของคำทั้งหน้า
            matcher = difflib.SequenceMatcher(None, str1, str2, autojunk=False)
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                
                # REPLACE: คำเปลี่ยนไป (เช่น 2025 -> 2026)
                if tag == 'replace':
                    if page1:
                        for k in range(i1, i2):
                            # กรองไฮไลท์เฉพาะคำที่มีความหมาย
                            if is_significant(words1[k][4]): 
                                r = fitz.Rect(words1[k][:4])
                                page1.draw_rect(r, color=(1, 0.5, 0.5), fill=(1, 0.5, 0.5), fill_opacity=0.5, width=0)
                    if page2:
                        for k in range(j1, j2):
                            if is_significant(words2[k][4]):
                                r = fitz.Rect(words2[k][:4])
                                page2.draw_rect(r, color=(1, 0.8, 0.2), fill=(1, 0.8, 0.2), fill_opacity=0.5, width=0)
                
                # DELETE: คำหายไปจาก Doc 1 (มีใน Doc 1 แต่ไม่มีใน Doc 2)
                elif tag == 'delete':
                    if page1:
                        for k in range(i1, i2):
                            if is_significant(words1[k][4]):
                                r = fitz.Rect(words1[k][:4])
                                page1.draw_rect(r, color=(1, 0.2, 0.2), fill=(1, 0.2, 0.2), fill_opacity=0.5, width=0)

                # INSERT: คำเพิ่มเข้ามาใน Doc 2 (ไม่มีใน Doc 1 แต่มีใน Doc 2)
                elif tag == 'insert':
                    if page2:
                        for k in range(j1, j2):
                            if is_significant(words2[k][4]):
                                r = fitz.Rect(words2[k][:4])
                                page2.draw_rect(r, color=(0.2, 1, 0.2), fill=(0.2, 1, 0.2), fill_opacity=0.5, width=0)

        # Generate Output Images
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