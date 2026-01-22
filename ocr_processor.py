import easyocr
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import difflib
import re
import numpy as np
from collections import defaultdict

# --- STOPWORDS (ลดลงเหลือแค่คำที่ไม่สำคัญจริงๆ) ---
STOPWORDS = {
    "นาย", "นาง", "นางสาว",  # คำนำหน้าชื่อ
    "ถนน", "ซอย", "แขวง", "เขต", "หมู่",  # คำนำหน้าที่อยู่
    "วันที่", "เดือน", "พ.ศ.", "บาท",  # หน่วย
    "mr", "mrs", "miss", "ms",
    "road", "soi", "district", "province",
    "the", "a", "an", "is", "are", "of", "to", "in", "for", "on", "with"
}

print("Initializing EasyOCR with GPU...")
try:
    reader = easyocr.Reader(['th', 'en'], gpu=True)
    print("✅ EasyOCR initialized with GPU")
except:
    reader = easyocr.Reader(['th', 'en'], gpu=False)
    print("✅ EasyOCR initialized with CPU")


def extract_text_from_pdf(pdf_path):
    """แกะข้อความสำหรับ LLM"""
    if not os.path.exists(pdf_path): 
        return ""
    try:
        doc = fitz.open(pdf_path)
    except: 
        return ""

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
        except: 
            pass
    return extracted_text


def clean_text(text):
    """ลบอักขระพิเศษ แต่เก็บตัวเลขและภาษาไทย"""
    text = re.sub(r'[^\w\u0E00-\u0E7F]', '', text) 
    return text.lower().strip()


def normalize_for_compare(text):
    """Normalize text สำหรับเปรียบเทียบ - ลบ space ทั้งหมด"""
    return re.sub(r'\s+', '', clean_text(text))


def normalize_number(text):
    """Normalize ตัวเลข - ลบ comma, space, จุด"""
    return re.sub(r'[,.\s]', '', text)


def is_significant(text):
    """ตรวจสอบว่าคำนี้สำคัญพอที่จะไฮไลท์หรือไม่"""
    clean = clean_text(text)
    if not clean:
        return False
    # คำสั้นมาก (1 ตัวอักษร) ไม่สำคัญ ยกเว้นตัวเลข
    if len(clean) == 1 and not clean.isdigit(): 
        return False
    if clean in STOPWORDS: 
        return False
    # ตัวเลขสำคัญเสมอ
    if any(char.isdigit() for char in clean): 
        return True
    # คำยาว 2 ตัวขึ้นไปถือว่าสำคัญ
    if len(clean) >= 2: 
        return True
    return False


def get_words_from_page_ocr(page):
    """ดึงคำและพิกัดด้วย EasyOCR"""
    pix = page.get_pixmap(dpi=300) 
    img_np = np.array(Image.open(io.BytesIO(pix.tobytes("png"))))
    
    results = reader.readtext(img_np)
    words = []
    
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
    """ดึงคำทั้งหมดจากหน้า"""
    if page is None:
        return []
    
    words = page.get_text("words")
    
    # ถ้าได้คำน้อยเกินไป → ใช้ OCR
    if len(words) < 5:
        print(f"  → Native text too few ({len(words)} words), using OCR...")
        try:
            words = get_words_from_page_ocr(page)
        except Exception as e:
            print(f"  → OCR failed: {e}")
            words = []
    else:
        # ตรวจสอบ bounding box ว่าถูกต้องไหม
        valid_words = []
        page_width = page.rect.width
        page_height = page.rect.height
        
        for w in words:
            x0, y0, x1, y1 = w[0], w[1], w[2], w[3]
            # ตรวจสอบว่า bounding box อยู่ในหน้า
            if 0 <= x0 < page_width and 0 <= y0 < page_height and x1 > x0 and y1 > y0:
                valid_words.append(w)
        
        words = valid_words
    
    return list(words)


def build_word_index(doc):
    """สร้าง index ของคำทั้งเอกสาร"""
    index = {
        'by_page': {},
        'by_word': defaultdict(list)
    }
    
    total_words = 0
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        words = get_words_all(page)
        index['by_page'][page_num] = words
        total_words += len(words)
        
        for word in words:
            text = word[4]
            clean = clean_text(text)
            if clean and is_significant(text):
                index['by_word'][clean].append((page_num, word))
    
    print(f"  → Total words: {total_words}, Unique significant words: {len(index['by_word'])}")
    return index


def fuzzy_match_word(word, word_index, threshold=0.70):
    """หาคำที่คล้ายกันใน index - ปรับปรุงให้จับ match ได้มากขึ้น"""
    clean = clean_text(word)
    normalized = normalize_for_compare(word)
    
    if not clean or len(clean) < 2:
        return []
    
    matches = []
    
    # 1. Exact match
    if clean in word_index['by_word']:
        for page_num, w in word_index['by_word'][clean]:
            matches.append((page_num, w, 1.0))
        return matches
    
    # 2. Check all words
    for indexed_word, locations in word_index['by_word'].items():
        indexed_normalized = normalize_for_compare(indexed_word)
        
        # 2a. Normalized exact match
        if normalized == indexed_normalized:
            for page_num, w in locations:
                matches.append((page_num, w, 0.95))
            continue
        
        # 2b. Substring match (คำหนึ่งอยู่ในอีกคำ)
        if len(normalized) >= 3 and len(indexed_normalized) >= 3:
            if normalized in indexed_normalized or indexed_normalized in normalized:
                for page_num, w in locations:
                    matches.append((page_num, w, 0.85))
                continue
        
        # 2c. Fuzzy match
        if abs(len(indexed_word) - len(clean)) > max(len(clean) * 0.5, 3):
            continue
            
        ratio = difflib.SequenceMatcher(None, clean, indexed_word).ratio()
        if ratio >= threshold:
            for page_num, w in locations:
                matches.append((page_num, w, ratio))
    
    return matches


def highlight_text_differences(pdf1_path, pdf2_path, mode='diff'):
    """Main function สำหรับไฮไลท์"""
    if not os.path.exists(pdf1_path) or not os.path.exists(pdf2_path): 
        return []

    try:
        doc1 = fitz.open(pdf1_path)
        doc2 = fitz.open(pdf2_path)
    except Exception as e:
        print(f"Error opening PDFs: {e}")
        return []

    print(f"\n📄 Document 1: {os.path.basename(pdf1_path)} ({len(doc1)} pages)")
    index1 = build_word_index(doc1)
    
    print(f"\n📄 Document 2: {os.path.basename(pdf2_path)} ({len(doc2)} pages)")
    index2 = build_word_index(doc2)

    if mode == 'same':
        print(f"\n🟢 Mode: SAME - Highlighting matching words...")
        highlight_same_mode(doc1, doc2, index1, index2)
    else:
        print(f"\n🔴 Mode: DIFF - Highlighting different words...")
        highlight_diff_mode(doc1, doc2, index1, index2)

    # Generate Output Images
    images = []
    max_pages = max(len(doc1), len(doc2))
    
    for i in range(max_pages):
        img1 = None
        img2 = None
        
        if i < len(doc1):
            pix = doc1[i].get_pixmap(dpi=150)
            img1 = Image.open(io.BytesIO(pix.tobytes("png")))
        if i < len(doc2):
            pix = doc2[i].get_pixmap(dpi=150)
            img2 = Image.open(io.BytesIO(pix.tobytes("png")))
            
        images.append((img1, img2))

    doc1.close()
    doc2.close()
    
    return images


def highlight_same_mode(doc1, doc2, index1, index2):
    """โหมด SAME: ไฮไลท์คำที่เหมือนกัน"""
    GREEN = (0, 0.8, 0)
    
    matched_in_doc1 = set()
    matched_in_doc2 = set()
    
    for clean_word, locations1 in index1['by_word'].items():
        # ลด threshold เป็น 0.70 เพื่อจับ match ได้มากขึ้น
        matches_in_doc2 = fuzzy_match_word(clean_word, index2, threshold=0.70)
        
        if matches_in_doc2:
            # ไฮไลท์ใน doc1
            for page_num, word in locations1:
                word_id = (page_num, word[0], word[1], word[4])
                if word_id not in matched_in_doc1:
                    matched_in_doc1.add(word_id)
                    page = doc1[page_num]
                    rect = fitz.Rect(word[:4])
                    page.draw_rect(rect, color=GREEN, fill=GREEN, fill_opacity=0.35, width=0)
            
            # ไฮไลท์ใน doc2
            for page_num, word, score in matches_in_doc2:
                word_id = (page_num, word[0], word[1], word[4])
                if word_id not in matched_in_doc2:
                    matched_in_doc2.add(word_id)
                    page = doc2[page_num]
                    rect = fitz.Rect(word[:4])
                    page.draw_rect(rect, color=GREEN, fill=GREEN, fill_opacity=0.35, width=0)
    
    print(f"  → Highlighted {len(matched_in_doc1)} words in Doc1, {len(matched_in_doc2)} words in Doc2")


def highlight_diff_mode(doc1, doc2, index1, index2):
    """โหมด DIFF: ไฮไลท์คำที่แตกต่างกัน - แบบเรียบง่าย"""
    RED = (1, 0.3, 0.3)       # มีใน Doc1 แต่ไม่มีใน Doc2
    GREEN = (0.3, 0.8, 0.3)   # มีใน Doc2 แต่ไม่มีใน Doc1
    
    # สร้าง set ของคำที่ normalized แล้ว สำหรับทั้ง 2 doc
    def get_normalized_words(index):
        """สร้าง set ของคำที่ normalized แล้ว"""
        words = set()
        for clean_word in index['by_word'].keys():
            words.add(clean_word)
            # ถ้าเป็นตัวเลข เพิ่ม normalized version ด้วย
            if any(c.isdigit() for c in clean_word):
                # ลบ leading zeros
                normalized = clean_word.lstrip('0') or '0'
                words.add(normalized)
        return words
    
    normalized1 = get_normalized_words(index1)
    normalized2 = get_normalized_words(index2)
    
    words_only_in_doc1 = set()
    words_only_in_doc2 = set()
    
    # หาคำที่มีเฉพาะใน Doc1
    for clean_word in index1['by_word'].keys():
        # ตรวจสอบ exact match
        if clean_word in normalized2:
            continue
        
        # ตรวจสอบ normalized number match
        if any(c.isdigit() for c in clean_word):
            normalized = clean_word.lstrip('0') or '0'
            if normalized in normalized2:
                continue
        
        # ตรวจสอบ fuzzy match สำหรับคำที่ไม่ใช่ตัวเลข
        if not any(c.isdigit() for c in clean_word):
            matches = fuzzy_match_word(clean_word, index2, threshold=0.80)
            if matches:
                continue
        
        words_only_in_doc1.add(clean_word)
    
    # หาคำที่มีเฉพาะใน Doc2
    for clean_word in index2['by_word'].keys():
        if clean_word in normalized1:
            continue
        
        if any(c.isdigit() for c in clean_word):
            normalized = clean_word.lstrip('0') or '0'
            if normalized in normalized1:
                continue
        
        if not any(c.isdigit() for c in clean_word):
            matches = fuzzy_match_word(clean_word, index1, threshold=0.80)
            if matches:
                continue
        
        words_only_in_doc2.add(clean_word)
    
    print(f"  → Words only in Doc1: {len(words_only_in_doc1)}")
    print(f"  → Words only in Doc2: {len(words_only_in_doc2)}")
    
    # Debug: แสดงตัวอย่างคำที่ต่าง
    if words_only_in_doc1:
        sample1 = list(words_only_in_doc1)[:5]
        print(f"  → Sample Doc1: {sample1}")
    if words_only_in_doc2:
        sample2 = list(words_only_in_doc2)[:5]
        print(f"  → Sample Doc2: {sample2}")
    
    # ไฮไลท์ Doc1 (แดง)
    count1 = 0
    for clean_word in words_only_in_doc1:
        for page_num, word in index1['by_word'][clean_word]:
            page = doc1[page_num]
            rect = fitz.Rect(word[:4])
            page.draw_rect(rect, color=RED, fill=RED, fill_opacity=0.4, width=0)
            count1 += 1
    
    # ไฮไลท์ Doc2 (เขียว)
    count2 = 0
    for clean_word in words_only_in_doc2:
        for page_num, word in index2['by_word'][clean_word]:
            page = doc2[page_num]
            rect = fitz.Rect(word[:4])
            page.draw_rect(rect, color=GREEN, fill=GREEN, fill_opacity=0.4, width=0)
            count2 += 1
    
    print(f"  → Highlighted {count1} in Doc1 (red), {count2} in Doc2 (green)")