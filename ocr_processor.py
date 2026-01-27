import easyocr
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import io
import os
import difflib
import re
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Lock สำหรับ thread-safe OCR
ocr_lock = threading.Lock()

# --- STOPWORDS (ลดลงเหลือแค่คำที่ไม่สำคัญจริงๆ) ---
STOPWORDS = {
    "นาย", "นาง", "นางสาว",  # คำนำหน้าชื่อ
    "ถนน", "ซอย", "แขวง", "เขต", "หมู่",  # คำนำหน้าที่อยู่
    "วันที่", "เดือน", "พ.ศ.", "บาท",  # หน่วย
    "mr", "mrs", "miss", "ms",
    "road", "soi", "district", "province",
    "the", "a", "an", "is", "are", "of", "to", "in", "for", "on", "with"
}
# หมายเหตุ: ลบ "คุณ" ออกจาก STOPWORDS เพราะมักติดกับชื่อคน

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


def normalize_thai_tones(text):
    """
    ลบวรรณยุกต์ไทยเพื่อเปรียบเทียบ
    เช่น "ค่าย" → "คาย", "บ้าน" → "บาน"
    """
    thai_tones = '\u0E48\u0E49\u0E4A\u0E4B'  # ่ ้ ๊ ๋
    for tone in thai_tones:
        text = text.replace(tone, '')
    return text


def normalize_for_compare(text):
    """Normalize text สำหรับเปรียบเทียบ - ลบ space ทั้งหมด (ไม่ลบวรรณยุกต์)"""
    return re.sub(r'\s+', '', clean_text(text))


def normalize_number(text):
    """Normalize ตัวเลข - ลบ comma, space, จุด"""
    return re.sub(r'[,.\s]', '', text)


def smart_normalize_number(text):
    """
    Smart Number Normalization - รองรับหลาย format:
    - "1,000.00" → "1000.00" → "1000"
    - "1,000,000" → "1000000"  
    - "40,000,000.00" → "40000000"
    - "7.0000" → "7"
    - "007" → "7"
    """
    # ลบ comma และ space
    normalized = re.sub(r'[,\s]', '', text)
    
    # ลอง parse เป็นตัวเลข
    try:
        # ถ้ามีจุด อาจเป็นทศนิยม
        if '.' in normalized:
            num = float(normalized)
            # ถ้าเป็นจำนวนเต็ม (เช่น 1000.00) → แปลงเป็น int
            if num == int(num):
                return str(int(num))
            else:
                # ตัดทศนิยมท้ายที่เป็น 0
                return f"{num:g}"
        else:
            # ลบ leading zeros
            num = int(normalized)
            return str(num)
    except:
        # ถ้า parse ไม่ได้ ใช้วิธีเดิม
        return normalized.lstrip('0') or '0'


def extract_number_value(text):
    """
    แยกค่าตัวเลขจาก text (รองรับ format ต่างๆ)
    Returns: (numeric_value, original_text) หรือ None ถ้าไม่ใช่ตัวเลข
    """
    # ลบ comma และ space
    clean = re.sub(r'[,\s]', '', text)
    
    # ลอง match ตัวเลข (รวมทศนิยม)
    match = re.match(r'^-?\d+\.?\d*$', clean)
    if match:
        try:
            if '.' in clean:
                return (float(clean), text)
            else:
                return (int(clean), text)
        except:
            pass
    return None


def numbers_are_equal(num1_text, num2_text):
    """
    เปรียบเทียบตัวเลข 2 ตัวว่าเท่ากันไหม (แม้ format ต่างกัน)
    เช่น "1,000.00" == "1000" → True
    """
    val1 = extract_number_value(num1_text)
    val2 = extract_number_value(num2_text)
    
    if val1 is None or val2 is None:
        return False
    
    # เปรียบเทียบค่า
    v1, v2 = val1[0], val2[0]
    
    # ถ้าเป็น float เปรียบเทียบด้วย tolerance
    if isinstance(v1, float) or isinstance(v2, float):
        return abs(float(v1) - float(v2)) < 0.001
    else:
        return v1 == v2


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


def ocr_single_image(img_np):
    """OCR รูปเดียว - thread-safe ด้วย lock"""
    with ocr_lock:
        return reader.readtext(img_np)


def get_words_from_page_ocr(page):
    """ดึงคำและพิกัดด้วย EasyOCR"""
    pix = page.get_pixmap(dpi=300) 
    img_np = np.array(Image.open(io.BytesIO(pix.tobytes("png"))))
    
    # ใช้ thread-safe OCR
    results = ocr_single_image(img_np)
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


def process_page_ocr(args):
    """Process single page สำหรับ parallel OCR"""
    page_num, pdf_path, page_rect_width, page_rect_height = args
    
    try:
        # เปิด PDF ใหม่ใน thread นี้
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        pix = page.get_pixmap(dpi=300)
        img_np = np.array(Image.open(io.BytesIO(pix.tobytes("png"))))
        
        # OCR with lock
        results = ocr_single_image(img_np)
        
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
        
        doc.close()
        return page_num, words
        
    except Exception as e:
        print(f"  → OCR failed for page {page_num}: {e}")
        return page_num, []


# Global settings
FORCE_OCR = True  # เปลี่ยนเป็น False ถ้าต้องการใช้ native text
PARALLEL_OCR = True  # เปิดใช้ parallel OCR
MAX_OCR_WORKERS = 2  # จำนวน workers (ไม่ควรเกิน 2-3 เพราะ OCR หนัก)


def get_words_all(page):
    """ดึงคำทั้งหมดจากหน้า"""
    if page is None:
        return []
    
    # ถ้าบังคับใช้ OCR
    if FORCE_OCR:
        try:
            return get_words_from_page_ocr(page)
        except Exception as e:
            print(f"  → OCR failed: {e}")
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


def build_word_index_parallel(pdf_path, progress_callback=None):
    """สร้าง index ด้วย parallel OCR"""
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    index = {
        'by_page': {},
        'by_word': defaultdict(list)
    }
    
    if progress_callback:
        progress_callback(f"  → Starting parallel OCR ({MAX_OCR_WORKERS} workers)...", 0)
    
    # เตรียม arguments สำหรับแต่ละหน้า
    page_args = []
    for page_num in range(total_pages):
        page = doc[page_num]
        page_args.append((page_num, pdf_path, page.rect.width, page.rect.height))
    
    doc.close()
    
    # ใช้ ThreadPoolExecutor สำหรับ parallel OCR
    total_words = 0
    completed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_OCR_WORKERS) as executor:
        futures = {executor.submit(process_page_ocr, args): args[0] for args in page_args}
        
        for future in as_completed(futures):
            page_num, words = future.result()
            index['by_page'][page_num] = words
            total_words += len(words)
            
            # Build word index
            for word in words:
                text = word[4]
                clean = clean_text(text)
                if clean and is_significant(text):
                    index['by_word'][clean].append((page_num, word))
            
            completed += 1
            if progress_callback:
                pct = completed / total_pages * 100
                progress_callback(f"  → OCR page {completed}/{total_pages} done", pct)
    
    print(f"  → Total words: {total_words}, Unique significant words: {len(index['by_word'])}")
    return index


def build_word_index(doc, progress_callback=None):
    """สร้าง index ของคำทั้งเอกสาร (sequential)"""
    index = {
        'by_page': {},
        'by_word': defaultdict(list)
    }
    
    total_words = 0
    total_pages = len(doc)
    
    for page_num in range(total_pages):
        page = doc[page_num]
        words = get_words_all(page)
        index['by_page'][page_num] = words
        total_words += len(words)
        
        for word in words:
            text = word[4]
            clean = clean_text(text)
            if clean and is_significant(text):
                index['by_word'][clean].append((page_num, word))
        
        # Update progress
        if progress_callback:
            pct = (page_num + 1) / total_pages * 100
            progress_callback(f"  → OCR page {page_num + 1}/{total_pages}", pct)
    
    print(f"  → Total words: {total_words}, Unique significant words: {len(index['by_word'])}")
    return index


def fuzzy_match_word(word, word_index, threshold=0.70):
    """หาคำที่คล้ายกันใน index - รองรับภาษาไทยที่ OCR อ่านวรรณยุกต์ต่างกัน"""
    clean = clean_text(word)
    normalized = normalize_for_compare(word)
    normalized_no_tone = normalize_thai_tones(normalized)  # ลบวรรณยุกต์
    
    if not clean or len(clean) < 2:
        return []
    
    matches = []
    
    # 1. Exact match
    if clean in word_index['by_word']:
        for page_num, w in word_index['by_word'][clean]:
            matches.append((page_num, w, 1.0))
        return matches
    
    # 2. Check all words for text matching
    for indexed_word, locations in word_index['by_word'].items():
        indexed_normalized = normalize_for_compare(indexed_word)
        indexed_no_tone = normalize_thai_tones(indexed_normalized)  # ลบวรรณยุกต์
        
        # 2a. Normalized exact match (รวมวรรณยุกต์)
        if normalized == indexed_normalized:
            for page_num, w in locations:
                matches.append((page_num, w, 0.95))
            continue
        
        # 2b. Match หลังลบวรรณยุกต์ (สำหรับ OCR ที่อ่านวรรณยุกต์ต่างกัน)
        if normalized_no_tone == indexed_no_tone:
            for page_num, w in locations:
                matches.append((page_num, w, 0.93))
            continue
        
        # 2c. Substring match (คำหนึ่งอยู่ในอีกคำ)
        if len(normalized) >= 3 and len(indexed_normalized) >= 3:
            if normalized in indexed_normalized or indexed_normalized in normalized:
                for page_num, w in locations:
                    matches.append((page_num, w, 0.85))
                continue
        
        # 2d. Substring match หลังลบวรรณยุกต์
        if len(normalized_no_tone) >= 3 and len(indexed_no_tone) >= 3:
            if normalized_no_tone in indexed_no_tone or indexed_no_tone in normalized_no_tone:
                for page_num, w in locations:
                    matches.append((page_num, w, 0.83))
                continue
        
        # 2e. Fuzzy match
        if abs(len(indexed_word) - len(clean)) > max(len(clean) * 0.5, 3):
            continue
            
        ratio = difflib.SequenceMatcher(None, clean, indexed_word).ratio()
        if ratio >= threshold:
            for page_num, w in locations:
                matches.append((page_num, w, ratio))
    
    return matches


def highlight_text_differences(pdf1_path, pdf2_path, mode='diff', progress_callback=None):
    """Main function สำหรับไฮไลท์"""
    def update_progress(msg, percent=None):
        if progress_callback:
            progress_callback(msg, percent)
        print(msg)
    
    if not os.path.exists(pdf1_path) or not os.path.exists(pdf2_path): 
        return [], {}

    try:
        doc1 = fitz.open(pdf1_path)
        doc2 = fitz.open(pdf2_path)
    except Exception as e:
        print(f"Error opening PDFs: {e}")
        return [], {}

    total_pages = len(doc1) + len(doc2)
    doc1_pages = len(doc1)
    doc2_pages = len(doc2)
    
    # Build word index (รองรับ parallel)
    update_progress(f"\n📄 Document 1: {os.path.basename(pdf1_path)} ({doc1_pages} pages)", 5)
    
    if PARALLEL_OCR and FORCE_OCR and doc1_pages > 1:
        doc1.close()
        index1 = build_word_index_parallel(pdf1_path, progress_callback=lambda msg, pct: update_progress(msg, 5 + pct * 0.25))
        doc1 = fitz.open(pdf1_path)  # เปิดใหม่สำหรับ highlight
    else:
        index1 = build_word_index(doc1, progress_callback=lambda msg, pct: update_progress(msg, 5 + pct * 0.25))
    
    update_progress(f"\n📄 Document 2: {os.path.basename(pdf2_path)} ({doc2_pages} pages)", 35)
    
    if PARALLEL_OCR and FORCE_OCR and doc2_pages > 1:
        doc2.close()
        index2 = build_word_index_parallel(pdf2_path, progress_callback=lambda msg, pct: update_progress(msg, 35 + pct * 0.25))
        doc2 = fitz.open(pdf2_path)  # เปิดใหม่สำหรับ highlight
    else:
        index2 = build_word_index(doc2, progress_callback=lambda msg, pct: update_progress(msg, 35 + pct * 0.25))

    # เก็บ summary statistics
    summary = {
        'mode': mode,
        'doc1_name': os.path.basename(pdf1_path),
        'doc2_name': os.path.basename(pdf2_path),
        'doc1_pages': len(doc1),
        'doc2_pages': len(doc2),
        'doc1_total_words': sum(len(words) for words in index1['by_page'].values()),
        'doc2_total_words': sum(len(words) for words in index2['by_page'].values()),
        'doc1_unique_words': len(index1['by_word']),
        'doc2_unique_words': len(index2['by_word']),
    }

    if mode == 'same':
        update_progress(f"\n🟢 Mode: SAME - Highlighting matching words...", 65)
        stats = highlight_same_mode(doc1, doc2, index1, index2)
        summary.update(stats)
    else:
        update_progress(f"\n🔴 Mode: DIFF - Highlighting different words...", 65)
        stats = highlight_diff_mode(doc1, doc2, index1, index2)
        summary.update(stats)

    # Generate Output Images พร้อม Legend
    update_progress("🖼️ Generating output images...", 75)
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
        
        # เพิ่ม Legend เฉพาะหน้าแรก
        if i == 0:
            if img1:
                img1 = add_legend_to_image(img1, mode, 'doc1')
            if img2:
                img2 = add_legend_to_image(img2, mode, 'doc2')
            
        images.append((img1, img2))
        
        # Update progress for each page
        img_progress = 75 + (i + 1) / max_pages * 20
        update_progress(f"  → Generated page {i+1}/{max_pages}", img_progress)

    doc1.close()
    doc2.close()
    
    # Print Summary
    print_summary(summary)
    update_progress("✅ Comparison complete!", 100)
    
    return images, summary


def highlight_text_differences_db(pdf_path, db_document, mode='diff', progress_callback=None):
    """
    Compare a PDF file against a document stored in database
    
    Args:
        pdf_path: Path to the PDF file to compare
        db_document: Document dict from database with 'word_index', 'extracted_text', etc.
        mode: 'diff' or 'same'
        progress_callback: Optional progress callback
    
    Returns:
        Tuple of (image_pairs, summary_dict)
    """
    def update_progress(msg, percent=None):
        if progress_callback:
            progress_callback(msg, percent)
        print(msg)
    
    if not os.path.exists(pdf_path):
        return [], {}
    
    try:
        doc1 = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return [], {}
    
    doc1_pages = len(doc1)
    
    # Get database word index
    db_index = db_document.get('word_index', {'by_page': {}, 'by_word': {}})
    db_page_count = db_document.get('page_count', 0)
    
    # Build word index for PDF file
    update_progress(f"\n📄 Input File: {os.path.basename(pdf_path)} ({doc1_pages} pages)", 5)
    
    if PARALLEL_OCR and FORCE_OCR and doc1_pages > 1:
        doc1.close()
        index1 = build_word_index_parallel(pdf_path, progress_callback=lambda msg, pct: update_progress(msg, 5 + pct * 0.45))
        doc1 = fitz.open(pdf_path)
    else:
        index1 = build_word_index(doc1, progress_callback=lambda msg, pct: update_progress(msg, 5 + pct * 0.45))
    
    update_progress(f"\n📄 Reference (DB): {db_document.get('document_id', 'Unknown')} ({db_page_count} pages)", 55)
    update_progress(f"  → Using cached word index from database", 60)
    
    # Summary statistics
    summary = {
        'mode': mode,
        'doc1_name': os.path.basename(pdf_path),
        'doc2_name': f"[DB] {db_document.get('document_id', 'Unknown')}",
        'doc1_pages': doc1_pages,
        'doc2_pages': db_page_count,
        'doc1_total_words': sum(len(words) for words in index1['by_page'].values()),
        'doc2_total_words': sum(len(words) for words in db_index.get('by_page', {}).values()),
        'doc1_unique_words': len(index1['by_word']),
        'doc2_unique_words': len(db_index.get('by_word', {})),
    }
    
    # We need a dummy doc2 for highlighting - create highlight overlays directly on images
    if mode == 'same':
        update_progress(f"\n🟢 Mode: SAME - Highlighting matching words...", 65)
        stats = highlight_same_mode_db(doc1, index1, db_index)
        summary.update(stats)
    else:
        update_progress(f"\n🔴 Mode: DIFF - Highlighting different words...", 65)
        stats = highlight_diff_mode_db(doc1, index1, db_index)
        summary.update(stats)
    
    # Generate Output Images
    update_progress("🖼️ Generating output images...", 75)
    images = []
    
    for i in range(doc1_pages):
        pix = doc1[i].get_pixmap(dpi=150)
        img1 = Image.open(io.BytesIO(pix.tobytes("png")))
        
        # Add Legend on first page
        if i == 0:
            img1 = add_legend_to_image(img1, mode, 'doc1')
        
        # For DB comparison, we only have one document to show
        # Could add reference text view later
        images.append((img1, None))
        
        img_progress = 75 + (i + 1) / doc1_pages * 20
        update_progress(f"  → Generated page {i+1}/{doc1_pages}", img_progress)
    
    doc1.close()
    
    print_summary(summary)
    update_progress("✅ Comparison complete!", 100)
    
    return images, summary


def highlight_same_mode_db(doc1, index1, db_index):
    """Mode SAME: Highlight matching words (file vs database)"""
    GREEN = (0, 0.8, 0)
    
    matched_in_doc1 = set()
    matched_words = []
    
    db_words = db_index.get('by_word', {})
    
    for clean_word, locations1 in index1['by_word'].items():
        # Exact match
        if clean_word in db_words:
            matched_words.append(clean_word)
            
            for page_num, word in locations1:
                word_id = (page_num, word[0], word[1], word[4])
                if word_id not in matched_in_doc1:
                    matched_in_doc1.add(word_id)
                    page = doc1[page_num]
                    rect = fitz.Rect(word[:4])
                    page.draw_rect(rect, color=GREEN, fill=GREEN, fill_opacity=0.35, width=0)
        else:
            # Fuzzy match
            matches_in_db = fuzzy_match_word_index(clean_word, db_words, threshold=0.60)
            
            if matches_in_db:
                matched_words.append(clean_word)
                
                for page_num, word in locations1:
                    word_id = (page_num, word[0], word[1], word[4])
                    if word_id not in matched_in_doc1:
                        matched_in_doc1.add(word_id)
                        page = doc1[page_num]
                        rect = fitz.Rect(word[:4])
                        page.draw_rect(rect, color=GREEN, fill=GREEN, fill_opacity=0.35, width=0)
    
    print(f"  → Matched unique words: {len(matched_words)}")
    print(f"  → Highlighted {len(matched_in_doc1)} words in input file")
    
    return {
        'matched_doc1': len(matched_in_doc1),
        'matched_doc2': 0,  # DB document not displayed
        'matched_unique_words': len(matched_words),
        'sample_matched': matched_words[:10]
    }


def highlight_diff_mode_db(doc1, index1, db_index):
    """Mode DIFF: Highlight different words (file vs database)"""
    RED = (1, 0.3, 0.3)
    
    words1 = set(index1['by_word'].keys())
    db_words = db_index.get('by_word', {})
    words2 = set(db_words.keys())
    
    words_only_in_doc1 = set()
    
    for clean_word in words1:
        if clean_word in words2:
            continue
        
        # Numbers use exact match only
        if any(c.isdigit() for c in clean_word):
            words_only_in_doc1.add(clean_word)
        else:
            matches = fuzzy_match_word_index(clean_word, db_words, threshold=0.80)
            if not matches:
                words_only_in_doc1.add(clean_word)
    
    print(f"  → Words only in input file: {len(words_only_in_doc1)}")
    
    # Highlight in doc1 (red)
    count1 = 0
    for clean_word in words_only_in_doc1:
        for page_num, word in index1['by_word'][clean_word]:
            page = doc1[page_num]
            rect = fitz.Rect(word[:4])
            page.draw_rect(rect, color=RED, fill=RED, fill_opacity=0.4, width=0)
            count1 += 1
    
    print(f"  → Highlighted {count1} words in input file (red)")
    
    return {
        'diff_doc1': len(words_only_in_doc1),
        'diff_doc2': 0,
        'highlighted_doc1': count1,
        'highlighted_doc2': 0,
        'sample_doc1': list(words_only_in_doc1)[:10],
        'sample_doc2': []
    }


def fuzzy_match_word_index(word, word_index_by_word, threshold=0.70):
    """Fuzzy match against a by_word index (dict of word -> locations)"""
    clean = clean_text(word)
    normalized = normalize_for_compare(word)
    normalized_no_tone = normalize_thai_tones(normalized)
    
    if not clean or len(clean) < 2:
        return []
    
    matches = []
    
    for indexed_word in word_index_by_word.keys():
        indexed_normalized = normalize_for_compare(indexed_word)
        indexed_no_tone = normalize_thai_tones(indexed_normalized)
        
        # Normalized exact match
        if normalized == indexed_normalized:
            matches.append(indexed_word)
            continue
        
        # Match after removing tones
        if normalized_no_tone == indexed_no_tone:
            matches.append(indexed_word)
            continue
        
        # Substring match
        if len(normalized) >= 3 and len(indexed_normalized) >= 3:
            if normalized in indexed_normalized or indexed_normalized in normalized:
                matches.append(indexed_word)
                continue
        
        # Fuzzy ratio
        if abs(len(indexed_word) - len(clean)) > max(len(clean) * 0.5, 3):
            continue
            
        ratio = difflib.SequenceMatcher(None, clean, indexed_word).ratio()
        if ratio >= threshold:
            matches.append(indexed_word)
    
    return matches


def add_legend_to_image(img, mode, doc_side):
    """เพิ่ม Legend ลงบนรูป"""
    # สร้าง copy ของรูป
    img = img.copy()
    draw = ImageDraw.Draw(img)
    
    # ขนาดและตำแหน่ง legend
    legend_height = 60
    legend_width = 280
    margin = 10
    x = margin
    y = margin
    
    # วาดพื้นหลัง legend
    draw.rectangle(
        [x, y, x + legend_width, y + legend_height],
        fill=(255, 255, 255, 230),
        outline=(100, 100, 100),
        width=2
    )
    
    # ใช้ font เริ่มต้น
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
        font_small = font
    
    # วาด Legend ตาม mode
    if mode == 'same':
        # สีเขียว = เหมือนกัน
        draw.rectangle([x + 10, y + 15, x + 30, y + 30], fill=(0, 204, 0))
        draw.text((x + 40, y + 14), "= Matching content", fill=(0, 0, 0), font=font_small)
        draw.text((x + 10, y + 38), f"Document: {doc_side.upper()}", fill=(80, 80, 80), font=font_small)
    else:
        if doc_side == 'doc1':
            # Doc1: แดง = มีเฉพาะใน Doc1
            draw.rectangle([x + 10, y + 15, x + 30, y + 30], fill=(255, 77, 77))
            draw.text((x + 40, y + 14), "= Only in THIS document", fill=(0, 0, 0), font=font_small)
        else:
            # Doc2: เขียว = มีเฉพาะใน Doc2
            draw.rectangle([x + 10, y + 15, x + 30, y + 30], fill=(77, 204, 77))
            draw.text((x + 40, y + 14), "= Only in THIS document", fill=(0, 0, 0), font=font_small)
        
        draw.text((x + 10, y + 38), f"Document: {doc_side.upper()}", fill=(80, 80, 80), font=font_small)
    
    return img


def print_summary(summary):
    """พิมพ์ Summary Report"""
    print("\n" + "="*60)
    print("📊 COMPARISON SUMMARY")
    print("="*60)
    
    mode_text = "Finding MATCHES" if summary['mode'] == 'same' else "Finding DIFFERENCES"
    print(f"Mode: {mode_text}")
    print("-"*60)
    
    print(f"📄 Document 1: {summary['doc1_name']}")
    print(f"   Pages: {summary['doc1_pages']}, Words: {summary['doc1_total_words']}, Unique: {summary['doc1_unique_words']}")
    
    print(f"📄 Document 2: {summary['doc2_name']}")
    print(f"   Pages: {summary['doc2_pages']}, Words: {summary['doc2_total_words']}, Unique: {summary['doc2_unique_words']}")
    
    print("-"*60)
    
    if summary['mode'] == 'same':
        print(f"✅ Matching words in Doc1: {summary.get('matched_doc1', 0)}")
        print(f"✅ Matching words in Doc2: {summary.get('matched_doc2', 0)}")
        
        # คำนวณ % match
        if summary['doc1_unique_words'] > 0:
            match_pct = (summary.get('matched_doc1', 0) / summary['doc1_unique_words']) * 100
            print(f"📈 Match rate: {match_pct:.1f}%")
    else:
        print(f"🔴 Words only in Doc1: {summary.get('diff_doc1', 0)}")
        print(f"🟢 Words only in Doc2: {summary.get('diff_doc2', 0)}")
        print(f"📝 Sample differences Doc1: {summary.get('sample_doc1', [])[:5]}")
        print(f"📝 Sample differences Doc2: {summary.get('sample_doc2', [])[:5]}")
        
        # คำนวณ % diff
        total_unique = summary['doc1_unique_words'] + summary['doc2_unique_words']
        if total_unique > 0:
            diff_count = summary.get('diff_doc1', 0) + summary.get('diff_doc2', 0)
            diff_pct = (diff_count / total_unique) * 100
            print(f"📈 Difference rate: {diff_pct:.1f}%")
    
    print("="*60 + "\n")


def highlight_same_mode(doc1, doc2, index1, index2):
    """โหมด SAME: ไฮไลท์คำที่เหมือนกัน"""
    GREEN = (0, 0.8, 0)
    
    matched_in_doc1 = set()
    matched_in_doc2 = set()
    matched_words = []
    
    # Debug: แสดงตัวอย่างคำใน index
    sample_words1 = list(index1['by_word'].keys())[:10]
    sample_words2 = list(index2['by_word'].keys())[:10]
    print(f"  → Sample words in Doc1: {sample_words1}")
    print(f"  → Sample words in Doc2: {sample_words2}")
    
    # Debug: หาคำที่มี "ประกาย" ในทั้ง 2 เอกสาร
    prakay_words1 = [w for w in index1['by_word'].keys() if 'ประกาย' in w]
    prakay_words2 = [w for w in index2['by_word'].keys() if 'ประกาย' in w]
    print(f"  → Words containing 'ประกาย' in Doc1: {prakay_words1}")
    print(f"  → Words containing 'ประกาย' in Doc2: {prakay_words2}")
    
    for clean_word, locations1 in index1['by_word'].items():
        # ลองหา exact match ก่อน
        if clean_word in index2['by_word']:
            matched_words.append(clean_word)
            
            # ไฮไลท์ใน doc1
            for page_num, word in locations1:
                word_id = (page_num, word[0], word[1], word[4])
                if word_id not in matched_in_doc1:
                    matched_in_doc1.add(word_id)
                    page = doc1[page_num]
                    rect = fitz.Rect(word[:4])
                    page.draw_rect(rect, color=GREEN, fill=GREEN, fill_opacity=0.35, width=0)
            
            # ไฮไลท์ใน doc2
            for page_num, word in index2['by_word'][clean_word]:
                word_id = (page_num, word[0], word[1], word[4])
                if word_id not in matched_in_doc2:
                    matched_in_doc2.add(word_id)
                    page = doc2[page_num]
                    rect = fitz.Rect(word[:4])
                    page.draw_rect(rect, color=GREEN, fill=GREEN, fill_opacity=0.35, width=0)
        else:
            # ลอง fuzzy match (ลด threshold เป็น 0.60 สำหรับภาษาไทย)
            matches_in_doc2 = fuzzy_match_word(clean_word, index2, threshold=0.60)
            
            if matches_in_doc2:
                matched_words.append(clean_word)
                
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
    
    print(f"  → Matched unique words: {len(matched_words)}")
    print(f"  → Sample matched: {matched_words[:10]}")
    print(f"  → Highlighted {len(matched_in_doc1)} words in Doc1, {len(matched_in_doc2)} words in Doc2")
    
    return {
        'matched_doc1': len(matched_in_doc1),
        'matched_doc2': len(matched_in_doc2),
        'matched_unique_words': len(matched_words),
        'sample_matched': matched_words[:10]
    }


def highlight_diff_mode(doc1, doc2, index1, index2):
    """โหมด DIFF: ไฮไลท์คำที่แตกต่างกัน - ใช้ Exact Match สำหรับตัวเลข"""
    RED = (1, 0.3, 0.3)       # มีใน Doc1 แต่ไม่มีใน Doc2
    GREEN = (0.3, 0.8, 0.3)   # มีใน Doc2 แต่ไม่มีใน Doc1
    
    # สร้าง set ของคำ
    words1 = set(index1['by_word'].keys())
    words2 = set(index2['by_word'].keys())
    
    words_only_in_doc1 = set()
    words_only_in_doc2 = set()
    
    # หาคำที่มีเฉพาะใน Doc1
    for clean_word in words1:
        # Exact match ก่อน
        if clean_word in words2:
            continue
        
        # ถ้าเป็นตัวเลข → ใช้ exact match เท่านั้น (ไม่ fuzzy)
        if any(c.isdigit() for c in clean_word):
            words_only_in_doc1.add(clean_word)
        else:
            # ถ้าไม่ใช่ตัวเลข → ลอง fuzzy match
            matches = fuzzy_match_word(clean_word, index2, threshold=0.80)
            if not matches:
                words_only_in_doc1.add(clean_word)
    
    # หาคำที่มีเฉพาะใน Doc2
    for clean_word in words2:
        if clean_word in words1:
            continue
        
        if any(c.isdigit() for c in clean_word):
            words_only_in_doc2.add(clean_word)
        else:
            matches = fuzzy_match_word(clean_word, index1, threshold=0.80)
            if not matches:
                words_only_in_doc2.add(clean_word)
    
    print(f"  → Words only in Doc1: {len(words_only_in_doc1)}")
    print(f"  → Words only in Doc2: {len(words_only_in_doc2)}")
    
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
    
    return {
        'diff_doc1': len(words_only_in_doc1),
        'diff_doc2': len(words_only_in_doc2),
        'highlighted_doc1': count1,
        'highlighted_doc2': count2,
        'sample_doc1': list(words_only_in_doc1)[:10],
        'sample_doc2': list(words_only_in_doc2)[:10]
    }