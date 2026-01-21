import easyocr
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import difflib
import re
import numpy as np
from collections import defaultdict

# --- STOPWORDS ---
STOPWORDS = {
    "นาย", "นาง", "นางสาว", "เด็กชาย", "เด็กหญิง", "บริษัท", "จํากัด", "มหาชน", 
    "ถนน", "ซอย", "แขวง", "เขต", "จังหวัด", "อำเภอ", "ตำบล", 
    "วันที่", "เดือน", "พ.ศ.", "เลขที่", "หมู่", "ราคา", "บาท", "สตางค์",
    "กรมธรรม์", "ประกันภัย", "ผู้เอาประกัน", "ที่อยู่", "เบอร์โทร", 
    "โทร", "แฟกซ์", "รายละเอียด", "จำนวน", "รวม", "ภาษี", "อากร", "หมายเหตุ",
    "mr", "mrs", "miss", "ms", "company", "ltd", "public", "limited",
    "road", "soi", "district", "province", "date", "month", "year", "no",
    "price", "baht", "policy", "insurance", "insured", "address", "tel", "fax",
    "detail", "amount", "total", "tax", "vat", "sum", "premium", "copy", "original",
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "for", "on", "with"
}

print("Initializing EasyOCR with GPU...")
try:
    reader = easyocr.Reader(['th', 'en'], gpu=True)
except:
    reader = easyocr.Reader(['th', 'en'], gpu=False)

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


def clean_text(text):
    """ลบอักขระพิเศษ แต่เก็บตัวเลขและภาษาไทย"""
    text = re.sub(r'[^\w\u0E00-\u0E7F]', '', text) 
    return text.lower().strip()


def normalize_number(text):
    """แปลงตัวเลขให้เป็นรูปแบบเดียวกัน (ลบ comma, space)"""
    return re.sub(r'[,\s]', '', text)


def is_significant(text):
    """ตรวจสอบว่าคำนี้สำคัญพอที่จะไฮไลท์หรือไม่"""
    clean = clean_text(text)
    if len(clean) < 2 and not clean.isdigit(): return False
    if clean in STOPWORDS: return False
    # ตัวเลขสำคัญเสมอ
    if any(char.isdigit() for char in clean): return True
    # คำยาวพอสมควร
    if len(clean) >= 3: return True
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
    if len(words) < 5: 
        try:
            words = get_words_from_page_ocr(page)
        except: 
            words = []
    return list(words)


# =============================================================================
# NEW: Global Word Index สำหรับ Cross-Page Matching
# =============================================================================

def build_word_index(doc):
    """
    สร้าง index ของคำทั้งเอกสาร
    Returns: {
        'by_page': {page_num: [(x0,y0,x1,y1,text,...), ...]},
        'by_word': {clean_text: [(page_num, word_tuple), ...]}
    }
    """
    index = {
        'by_page': {},
        'by_word': defaultdict(list)
    }
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        words = get_words_all(page)
        index['by_page'][page_num] = words
        
        for word in words:
            text = word[4]
            clean = clean_text(text)
            if clean and is_significant(text):
                index['by_word'][clean].append((page_num, word))
                
                # เพิ่ม normalized number ด้วย
                if any(c.isdigit() for c in clean):
                    norm = normalize_number(clean)
                    if norm != clean:
                        index['by_word'][norm].append((page_num, word))
    
    return index


def fuzzy_match_word(word, word_index, threshold=0.75):
    """
    หาคำที่คล้ายกันใน index
    Returns: list of (page_num, word_tuple, similarity_score)
    """
    clean = clean_text(word)
    if not clean:
        return []
    
    matches = []
    
    # Exact match
    if clean in word_index['by_word']:
        for page_num, w in word_index['by_word'][clean]:
            matches.append((page_num, w, 1.0))
    
    # Fuzzy match
    for indexed_word, locations in word_index['by_word'].items():
        if indexed_word == clean:
            continue
        
        # Skip if length difference is too big
        if abs(len(indexed_word) - len(clean)) > max(len(clean) * 0.5, 2):
            continue
            
        ratio = difflib.SequenceMatcher(None, clean, indexed_word).ratio()
        if ratio >= threshold:
            for page_num, w in locations:
                matches.append((page_num, w, ratio))
    
    return matches


# =============================================================================
# IMPROVED: Highlight Functions
# =============================================================================

def highlight_text_differences(pdf1_path, pdf2_path, mode='diff'):
    """
    Main function สำหรับไฮไลท์ความแตกต่าง/ความเหมือน
    
    mode='diff': ไฮไลท์จุดที่ต่างกัน
    mode='same': ไฮไลท์จุดที่เหมือนกัน (รองรับ cross-page)
    """
    if not os.path.exists(pdf1_path) or not os.path.exists(pdf2_path): 
        return []

    try:
        doc1 = fitz.open(pdf1_path)
        doc2 = fitz.open(pdf2_path)
    except Exception as e:
        print(f"Error: {e}")
        return []

    # สร้าง Global Word Index สำหรับทั้ง 2 เอกสาร
    print("Building word index for Document 1...")
    index1 = build_word_index(doc1)
    print("Building word index for Document 2...")
    index2 = build_word_index(doc2)

    if mode == 'same':
        highlight_same_mode(doc1, doc2, index1, index2)
    else:
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
    """
    โหมด SAME: ไฮไลท์คำที่เหมือนกันระหว่าง 2 เอกสาร
    รองรับ Cross-Page Matching
    """
    GREEN = (0, 0.8, 0)
    
    # Track คำที่ match แล้ว เพื่อไม่ให้ไฮไลท์ซ้ำ
    matched_in_doc1 = set()
    matched_in_doc2 = set()
    
    # หาคำที่ตรงกัน
    for clean_word, locations1 in index1['by_word'].items():
        matches_in_doc2 = fuzzy_match_word(clean_word, index2, threshold=0.80)
        
        if matches_in_doc2:
            # ไฮไลท์ทุกที่ที่เจอใน doc1
            for page_num, word in locations1:
                word_id = (page_num, word[0], word[1], word[4])
                if word_id not in matched_in_doc1:
                    matched_in_doc1.add(word_id)
                    page = doc1[page_num]
                    rect = fitz.Rect(word[:4])
                    page.draw_rect(rect, color=GREEN, fill=GREEN, fill_opacity=0.35, width=0)
            
            # ไฮไลท์ทุกที่ที่เจอใน doc2
            for page_num, word, score in matches_in_doc2:
                word_id = (page_num, word[0], word[1], word[4])
                if word_id not in matched_in_doc2:
                    matched_in_doc2.add(word_id)
                    page = doc2[page_num]
                    rect = fitz.Rect(word[:4])
                    page.draw_rect(rect, color=GREEN, fill=GREEN, fill_opacity=0.35, width=0)


def highlight_diff_mode(doc1, doc2, index1, index2):
    """
    โหมด DIFF: ไฮไลท์คำที่แตกต่างกันระหว่าง 2 เอกสาร
    
    สีที่ใช้:
    - แดง: คำที่มีเฉพาะใน Doc1 (หายไป)
    - เขียว: คำที่มีเฉพาะใน Doc2 (เพิ่มเข้ามา)
    - ส้ม/เหลือง: คำที่เปลี่ยนค่า (เช่น ตัวเลขต่างกัน)
    """
    RED = (1, 0.2, 0.2)       # มีใน Doc1 แต่ไม่มีใน Doc2
    GREEN = (0.2, 0.8, 0.2)   # มีใน Doc2 แต่ไม่มีใน Doc1
    ORANGE = (1, 0.6, 0.2)    # มีทั้งสองแต่ค่าต่างกัน (potential change)
    
    # หาคำที่อยู่ใน Doc1 แต่ไม่อยู่ใน Doc2
    words_only_in_doc1 = set()
    words_only_in_doc2 = set()
    words_in_both = set()
    
    for clean_word in index1['by_word'].keys():
        if clean_word in index2['by_word']:
            words_in_both.add(clean_word)
        else:
            # ลอง fuzzy match
            matches = fuzzy_match_word(clean_word, index2, threshold=0.85)
            if matches:
                words_in_both.add(clean_word)
            else:
                words_only_in_doc1.add(clean_word)
    
    for clean_word in index2['by_word'].keys():
        if clean_word not in index1['by_word']:
            matches = fuzzy_match_word(clean_word, index1, threshold=0.85)
            if not matches:
                words_only_in_doc2.add(clean_word)
    
    # ไฮไลท์คำที่อยู่เฉพาะใน Doc1 (แดง)
    for clean_word in words_only_in_doc1:
        for page_num, word in index1['by_word'][clean_word]:
            page = doc1[page_num]
            rect = fitz.Rect(word[:4])
            page.draw_rect(rect, color=RED, fill=RED, fill_opacity=0.4, width=0)
    
    # ไฮไลท์คำที่อยู่เฉพาะใน Doc2 (เขียว)
    for clean_word in words_only_in_doc2:
        for page_num, word in index2['by_word'][clean_word]:
            page = doc2[page_num]
            rect = fitz.Rect(word[:4])
            page.draw_rect(rect, color=GREEN, fill=GREEN, fill_opacity=0.4, width=0)
    
    # Optional: ไฮไลท์ตัวเลขที่มีทั้งสองฝั่งแต่ค่าอาจต่างกัน
    # (เช่น 1000 กับ 2000 - ทั้งคู่เป็นตัวเลขแต่ค่าต่างกัน)
    highlight_potential_number_changes(doc1, doc2, index1, index2, ORANGE)


def highlight_potential_number_changes(doc1, doc2, index1, index2, color):
    """
    ไฮไลท์ตัวเลขที่อาจเปลี่ยนค่า
    วิธี: เทียบตัวเลขที่อยู่ใกล้กัน (ตำแหน่งใกล้เคียง) แต่ค่าต่างกัน
    """
    # รวบรวมตัวเลขจากทั้ง 2 เอกสาร พร้อมตำแหน่ง relative
    def extract_numbers_with_context(index):
        numbers = []
        for clean_word, locations in index['by_word'].items():
            if any(c.isdigit() for c in clean_word):
                for page_num, word in locations:
                    # คำนวณตำแหน่ง relative (0-1) บนหน้า
                    rel_y = word[1] / 1000  # ประมาณการ
                    numbers.append({
                        'value': clean_word,
                        'page': page_num,
                        'word': word,
                        'rel_y': rel_y
                    })
        return numbers
    
    nums1 = extract_numbers_with_context(index1)
    nums2 = extract_numbers_with_context(index2)
    
    # เทียบตัวเลขที่อยู่ตำแหน่งใกล้กัน (same page, similar y position)
    for n1 in nums1:
        for n2 in nums2:
            # ถ้าหน้าเดียวกัน และ y ใกล้กัน (within 10%)
            if n1['page'] == n2['page'] and abs(n1['rel_y'] - n2['rel_y']) < 0.1:
                # ถ้าค่าต่างกัน
                if n1['value'] != n2['value']:
                    # ไฮไลท์ทั้งคู่
                    page1 = doc1[n1['page']]
                    rect1 = fitz.Rect(n1['word'][:4])
                    page1.draw_rect(rect1, color=color, fill=color, fill_opacity=0.35, width=0)
                    
                    page2 = doc2[n2['page']]
                    rect2 = fitz.Rect(n2['word'][:4])
                    page2.draw_rect(rect2, color=color, fill=color, fill_opacity=0.35, width=0)


# =============================================================================
# Alternative: Semantic Block Comparison (สำหรับเอกสารที่ layout ต่างกันมาก)
# =============================================================================

def extract_key_values(text):
    """
    แยก key-value pairs จากข้อความ
    เช่น "ชื่อ: สมชาย" -> {'ชื่อ': 'สมชาย'}
    """
    pairs = {}
    
    # Pattern สำหรับ Thai documents
    patterns = [
        r'(ชื่อ|นามสกุล|ที่อยู่|เลขที่|วันที่|จำนวน|ราคา|รวม)[:\s]+([^\n]+)',
        r'(Name|Address|Date|Amount|Total|No\.?)[:\s]+([^\n]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for key, value in matches:
            pairs[key.strip().lower()] = value.strip()
    
    return pairs


def compare_key_values(doc1_text, doc2_text):
    """
    เทียบ key-value pairs ระหว่าง 2 เอกสาร
    Returns: {
        'same': [(key, value), ...],
        'different': [(key, val1, val2), ...],
        'only_in_1': [(key, value), ...],
        'only_in_2': [(key, value), ...]
    }
    """
    kv1 = extract_key_values(doc1_text)
    kv2 = extract_key_values(doc2_text)
    
    result = {
        'same': [],
        'different': [],
        'only_in_1': [],
        'only_in_2': []
    }
    
    all_keys = set(kv1.keys()) | set(kv2.keys())
    
    for key in all_keys:
        if key in kv1 and key in kv2:
            if kv1[key] == kv2[key]:
                result['same'].append((key, kv1[key]))
            else:
                result['different'].append((key, kv1[key], kv2[key]))
        elif key in kv1:
            result['only_in_1'].append((key, kv1[key]))
        else:
            result['only_in_2'].append((key, kv2[key]))
    
    return result