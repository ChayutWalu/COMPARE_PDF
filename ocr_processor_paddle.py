# =====================================================
# OCR Processor - PaddleOCR Version
# เปลี่ยนจาก EasyOCR เป็น PaddleOCR เพื่อความเร็วที่ดีขึ้น
# =====================================================

import os
# Set environment variable before importing paddle/paddleocr
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU

from paddleocr import PaddleOCR
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import io
import difflib
import re
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import platform

# =====================================================
# Platform Detection - รองรับ macOS, Windows, Linux
# =====================================================
IS_MAC = platform.system() == 'Darwin'
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'

print(f"🖥️  Platform: {platform.system()} ({platform.machine()})")

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

# --- FORM_LABELS: หัวข้อ/labels ของ form ที่ไม่ควร highlight ---
FORM_LABELS_EXACT = {
    "u", "hr", "น", "baht", "sq", "m2",
    "to", "from", "no", "of", "as", "at", "on", "per",
    "sum", "vat", "tax", "net", "rate", "area", "code",
    "และ", "หรือ", "and", "or", "ที่", "ณ",
}

FORM_LABELS_PARTIAL = {
    "เวลา", "time", "เริ่มต้น", "สิ้นสุด",
    "ตำแหน่ง", "จังหวัด", "อำเภอ", "ตำบล", "รหัสไปรษณีย์",
    "district", "province", "subdistrict", "block", "บล็อก",
    "ตารางที่", "รายการที่", "ลำดับที่", "itemno",
    "รายละเอียด", "description", "จำนวนเงิน", "amount",
    "ความเสียหาย", "deductible", "ส่วนแรก", "excess",
    "จำนวนชั้น", "จำนวนอาคาร", "จำนวนห้อง", "storey", "building",
    "พื้นที่", "ภายใน", "internal", "total",
    "สถานที่", "occupancy", "รหัสภัย", "riskcode",
    "ชั้นของ", "class", "สิ่งปลูกสร้าง", "construction",
    "เจ้าของ", "owner", "ผู้เช่า", "tenant",
    "เบี้ยประกัน", "premium", "อากรแสตมป์", "stamp", "duty",
    "ภาษี", "รวม", "สุทธิ",
    "อัตรา", "เงื่อนไข", "condition", "clause",
    "ข้อตกลง", "agreement", "วันที่ทำ", "issued", "made",
    "ตัวแทน", "agent", "นายหน้า", "broker", "ใบอนุญาต", "license",
    "โดยตรง", "direct", "กรมธรรม์", "policy",
}

FORM_LABELS = FORM_LABELS_EXACT | FORM_LABELS_PARTIAL
ALL_SKIP_WORDS = STOPWORDS | FORM_LABELS

# =====================================================
# PaddleOCR Initialization - CPU mode (stable)
# =====================================================
print("Initializing PaddleOCR...")

# PaddleOCR 2.x config - ใช้ CPU ก่อนเพื่อความเสถียร
# สำหรับ GPU ต้องติดตั้ง CUDA 11.8 และ cuDNN ที่ถูกต้อง

ocr_reader = PaddleOCR(
    use_angle_cls=True,
    lang='ch',  # Chinese model รองรับหลายภาษา
    use_gpu=False,
    show_log=False,
    enable_mkldnn=True,  # เร่งความเร็ว CPU
)
print("✅ PaddleOCR initialized with CPU (MKL-DNN enabled)")


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
            
            # PaddleOCR returns: [[box, (text, confidence)], ...]
            result = ocr_reader.ocr(img_np, cls=True)
            
            if result and result[0]:
                texts = [line[1][0] for line in result[0] if line[1][1] > 0.5]
                extracted_text += f"--- Page {i+1} (PaddleOCR) ---\n{chr(10).join(texts)}\n"
        except Exception as e:
            print(f"OCR Error on page {i+1}: {e}")
            pass
    return extracted_text


def clean_text(text):
    """ลบอักขระพิเศษ แต่เก็บตัวเลขและภาษาไทย"""
    text = re.sub(r'[^\w\u0E00-\u0E7F]', '', text) 
    return text.lower().strip()


def normalize_thai_tones(text):
    """ลบวรรณยุกต์ไทยเพื่อเปรียบเทียบ"""
    thai_tones = '\u0E48\u0E49\u0E4A\u0E4B'
    for tone in thai_tones:
        text = text.replace(tone, '')
    return text


def normalize_leading_zeros(text):
    """Normalize leading zeros สำหรับวันที่ไทย"""
    normalized = re.sub(r'^(\d)([\u0E00-\u0E7F])', r'0\1\2', text)
    return normalized


def normalize_for_compare(text):
    """Normalize text สำหรับเปรียบเทียบ"""
    normalized = re.sub(r'[\s/\-\.]+', '', clean_text(text))
    normalized = normalize_leading_zeros(normalized)
    return normalized


def normalize_date_text(text):
    """Normalize วันที่ format ต่างๆ"""
    return re.sub(r'[\s/\-]+', '', text)


def normalize_number(text):
    """Normalize ตัวเลข"""
    return re.sub(r'[,.\s]', '', text)


def smart_normalize_number(text):
    """Smart Number Normalization"""
    normalized = re.sub(r'[,\s]', '', text)
    try:
        if '.' in normalized:
            num = float(normalized)
            if num == int(num):
                return str(int(num))
            else:
                return f"{num:g}"
        else:
            num = int(normalized)
            return str(num)
    except:
        return normalized.lstrip('0') or '0'


def extract_number_value(text):
    """แยกค่าตัวเลขจาก text"""
    clean = re.sub(r'[,\s]', '', text)
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
    """เปรียบเทียบตัวเลข 2 ตัว"""
    val1 = extract_number_value(num1_text)
    val2 = extract_number_value(num2_text)
    
    if val1 is None or val2 is None:
        return False
    
    v1, v2 = val1[0], val2[0]
    if isinstance(v1, float) or isinstance(v2, float):
        return abs(float(v1) - float(v2)) < 0.001
    else:
        return v1 == v2


def is_thai_number_word(text):
    """ตรวจสอบจำนวนเงินตัวหนังสือภาษาไทย"""
    thai_digits = ['หนึ่ง', 'สอง', 'สาม', 'สี่', 'ห้า', 'หก', 'เจ็ด', 'แปด', 'เก้า', 'สิบ', 
                   'เอ็ด', 'ยี่', 'ศูนย์']
    thai_units = ['ร้อย', 'พัน', 'หมื่น', 'แสน', 'ล้าน', 'สิบ']
    money_words = ['บาท', 'สตางค์', 'ถ้วน']
    
    has_digit = any(digit in text for digit in thai_digits)
    has_unit = any(unit in text for unit in thai_units)
    has_money = any(word in text for word in money_words)
    
    return (has_digit and has_unit) or (has_money and has_digit)


# =====================================================
# Date/Time Extraction
# =====================================================
THAI_MONTHS = {
    'มกราคม': '01', 'กุมภาพันธ์': '02', 'มีนาคม': '03', 'เมษายน': '04',
    'พฤษภาคม': '05', 'มิถุนายน': '06', 'กรกฎาคม': '07', 'สิงหาคม': '08',
    'กันยายน': '09', 'ตุลาคม': '10', 'พฤศจิกายน': '11', 'ธันวาคม': '12',
    'ม.ค.': '01', 'ก.พ.': '02', 'มี.ค.': '03', 'เม.ย.': '04',
    'พ.ค.': '05', 'มิ.ย.': '06', 'ก.ค.': '07', 'ส.ค.': '08',
    'ก.ย.': '09', 'ต.ค.': '10', 'พ.ย.': '11', 'ธ.ค.': '12'
}

DATE_KEYWORDS = [
    'ระยะเวลาประกันภัย', 'วันที่', 'วันทำสัญญา', 'สิ้นสุดวันที่', 'เริ่มต้นวันที่',
    'ตั้งแต่วันที่', 'ถึงวันที่', 'วันทำสัญญาประกันภัย', 'วันที่ทำสัญญา',
    'period of insurance', 'from', 'agreement made on', 'policy issued',
    'เริ่มต้น', 'สิ้นสุด', 'เวลา'
]


def extract_thai_dates(text):
    """แยกวันที่ภาษาไทยจาก text"""
    dates = []
    for month_th, month_num in THAI_MONTHS.items():
        pattern = rf'(\d{{1,2}})\s*[/\-]?\s*{re.escape(month_th)}\s*[/\-]?\s*(\d{{4}})'
        matches = re.findall(pattern, text)
        for day, year in matches:
            dates.append(f"{int(day):02d}-{month_num}-{year}")
    
    pattern = r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})'
    matches = re.findall(pattern, text)
    for day, month, year in matches:
        dates.append(f"{int(day):02d}-{int(month):02d}-{year}")
    
    return dates


def extract_times(text):
    """แยกเวลาจาก text"""
    times = []
    pattern = r'(\d{1,2})[.:](\d{2})\s*(?:น\.?|u\.?)?'
    matches = re.findall(pattern, text)
    for hour, minute in matches:
        h = int(hour)
        if 0 <= h <= 24:
            times.append(f"{h:02d}:{minute}")
    return times


def is_date_related_word(text):
    """ตรวจสอบว่าเป็นส่วนหนึ่งของวันที่"""
    if text in THAI_MONTHS:
        return True
    if re.match(r'^25\d{2}$', text):
        return True
    if re.match(r'^0?[1-9]$|^[12]\d$|^3[01]$', text):
        return True
    for month in THAI_MONTHS.keys():
        if month in text and re.search(r'\d', text):
            return True
    if re.match(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{4}$', text):
        return True
    return False


def is_date_related_context(text):
    """ตรวจสอบ context วันที่"""
    text_lower = text.lower()
    return any(kw in text_lower or kw in text for kw in DATE_KEYWORDS)


def dates_match(text1, text2):
    """เปรียบเทียบวันที่"""
    dates1 = extract_thai_dates(text1)
    dates2 = extract_thai_dates(text2)
    if not dates1 and not dates2:
        return None
    if dates1 and dates2:
        return sorted(dates1) == sorted(dates2)
    return False


def times_match(text1, text2):
    """เปรียบเทียบเวลา"""
    times1 = extract_times(text1)
    times2 = extract_times(text2)
    if not times1 and not times2:
        return None
    if times1 and times2:
        return sorted(times1) == sorted(times2)
    return False


def is_form_label(text):
    """ตรวจสอบว่าเป็น form label"""
    clean = clean_text(text)
    original_lower = text.lower().strip()
    
    if not clean:
        return False
    
    if clean in FORM_LABELS_EXACT:
        return True
    if clean in FORM_LABELS_PARTIAL:
        return True
    
    for label in FORM_LABELS_PARTIAL:
        if len(label) >= 4:
            if label in clean or label in original_lower:
                return True
    
    label_prefixes = [
        'รายละเอียด', 'จำนวน', 'พื้นที่', 'สถานที่', 'ชั้นของ',
        'description', 'amount', 'number', 'location',
        'ความเสียหาย', 'อัตรา', 'เงื่อนไข', 'ข้อตกลง'
    ]
    for prefix in label_prefixes:
        if clean.startswith(prefix) or original_lower.startswith(prefix):
            return True
    
    return False


def is_significant(text):
    """ตรวจสอบคำสำคัญ"""
    clean = clean_text(text)
    if not clean:
        return False
    if len(clean) < 2:
        return False
    if clean in STOPWORDS:
        return False
    return True


# =====================================================
# PaddleOCR Specific Functions
# =====================================================
def ocr_single_image(img_np):
    """OCR รูปเดียว - thread-safe ด้วย lock (PaddleOCR version)"""
    with ocr_lock:
        result = ocr_reader.ocr(img_np, cls=True)
        
        # Convert PaddleOCR format to EasyOCR-like format for compatibility
        # PaddleOCR: [[box, (text, confidence)], ...]
        # EasyOCR:   [(box, text, confidence), ...]
        converted = []
        if result and result[0]:
            for line in result[0]:
                box = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text = line[1][0]
                confidence = line[1][1]
                
                # Convert box format: PaddleOCR uses 4 points, we need corners
                # box format: [top-left, top-right, bottom-right, bottom-left]
                converted.append((box, text, confidence))
        
        return converted


def get_words_from_page_ocr(page):
    """ดึงคำและพิกัดด้วย PaddleOCR"""
    pix = page.get_pixmap(dpi=300) 
    img_np = np.array(Image.open(io.BytesIO(pix.tobytes("png"))))
    
    results = ocr_single_image(img_np)
    words = []
    
    scale_x = page.rect.width / pix.width
    scale_y = page.rect.height / pix.height

    for (bbox, text, prob) in results:
        if prob > 0.2 and text.strip():
            # PaddleOCR bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # Format: [top-left, top-right, bottom-right, bottom-left]
            tl, tr, br, bl = bbox
            
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
    """Process single page สำหรับ parallel OCR (PaddleOCR version)"""
    page_num, pdf_path, page_rect_width, page_rect_height = args
    
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        pix = page.get_pixmap(dpi=300)
        img_np = np.array(Image.open(io.BytesIO(pix.tobytes("png"))))
        
        results = ocr_single_image(img_np)
        
        words = []
        scale_x = page.rect.width / pix.width
        scale_y = page.rect.height / pix.height
        
        for (bbox, text, prob) in results:
            if prob > 0.2 and text.strip():
                tl, tr, br, bl = bbox
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


# =====================================================
# Global Settings
# =====================================================
FORCE_OCR = True

if IS_MAC:
    PARALLEL_OCR = False
    MAX_OCR_WORKERS = 1
    print("⚙️  Parallel OCR: Disabled (macOS)")
else:
    PARALLEL_OCR = True
    MAX_OCR_WORKERS = 2
    print(f"⚙️  Parallel OCR: Enabled ({MAX_OCR_WORKERS} workers)")


def get_words_all(page):
    """ดึงคำทั้งหมดจากหน้า"""
    if page is None:
        return []
    
    if FORCE_OCR:
        try:
            return get_words_from_page_ocr(page)
        except Exception as e:
            print(f"  → OCR failed: {e}")
            return []
    
    words = page.get_text("words")
    
    if len(words) < 5:
        print(f"  → Native text too few ({len(words)} words), using OCR...")
        try:
            words = get_words_from_page_ocr(page)
        except Exception as e:
            print(f"  → OCR failed: {e}")
            words = []
    else:
        valid_words = []
        page_width = page.rect.width
        page_height = page.rect.height
        
        for w in words:
            x0, y0, x1, y1 = w[0], w[1], w[2], w[3]
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
    
    page_args = []
    for page_num in range(total_pages):
        page = doc[page_num]
        page_args.append((page_num, pdf_path, page.rect.width, page.rect.height))
    
    doc.close()
    
    total_words = 0
    completed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_OCR_WORKERS) as executor:
        futures = {executor.submit(process_page_ocr, args): args[0] for args in page_args}
        
        for future in as_completed(futures):
            page_num, words = future.result()
            index['by_page'][page_num] = words
            total_words += len(words)
            
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
        
        if progress_callback:
            pct = (page_num + 1) / total_pages * 100
            progress_callback(f"  → OCR page {page_num + 1}/{total_pages}", pct)
    
    print(f"  → Total words: {total_words}, Unique significant words: {len(index['by_word'])}")
    return index


def fuzzy_match_word(word, word_index, threshold=0.70):
    """หาคำที่คล้ายกันใน index"""
    clean = clean_text(word)
    normalized = normalize_for_compare(word)
    normalized_no_tone = normalize_thai_tones(normalized)
    
    if not clean or len(clean) < 2:
        return []
    
    matches = []
    
    if clean in word_index['by_word']:
        for page_num, w in word_index['by_word'][clean]:
            matches.append((page_num, w, 1.0))
        return matches
    
    for indexed_word, locations in word_index['by_word'].items():
        indexed_normalized = normalize_for_compare(indexed_word)
        indexed_no_tone = normalize_thai_tones(indexed_normalized)
        
        if normalized == indexed_normalized:
            for page_num, w in locations:
                matches.append((page_num, w, 0.95))
            continue
        
        if normalized_no_tone == indexed_no_tone:
            for page_num, w in locations:
                matches.append((page_num, w, 0.93))
            continue
        
        if len(normalized) >= 3 and len(indexed_normalized) >= 3:
            if normalized in indexed_normalized or indexed_normalized in normalized:
                for page_num, w in locations:
                    matches.append((page_num, w, 0.85))
                continue
        
        if len(normalized_no_tone) >= 3 and len(indexed_no_tone) >= 3:
            if normalized_no_tone in indexed_no_tone or indexed_no_tone in normalized_no_tone:
                for page_num, w in locations:
                    matches.append((page_num, w, 0.83))
                continue
        
        if abs(len(indexed_word) - len(clean)) > max(len(clean) * 0.5, 3):
            continue
            
        ratio = difflib.SequenceMatcher(None, clean, indexed_word).ratio()
        if ratio >= threshold:
            for page_num, w in locations:
                matches.append((page_num, w, ratio))
    
    return matches


def fuzzy_match_word_index(word, word_index_by_word, threshold=0.70):
    """Fuzzy match against a by_word index"""
    clean = clean_text(word)
    normalized = normalize_for_compare(word)
    normalized_no_tone = normalize_thai_tones(normalized)
    
    if not clean or len(clean) < 2:
        return []
    
    matches = []
    
    for indexed_word in word_index_by_word.keys():
        indexed_normalized = normalize_for_compare(indexed_word)
        indexed_no_tone = normalize_thai_tones(indexed_normalized)
        
        if normalized == indexed_normalized:
            matches.append(indexed_word)
            continue
        
        if normalized_no_tone == indexed_no_tone:
            matches.append(indexed_word)
            continue
        
        if len(normalized) >= 3 and len(indexed_normalized) >= 3:
            if normalized in indexed_normalized or indexed_normalized in normalized:
                matches.append(indexed_word)
                continue
        
        if abs(len(indexed_word) - len(clean)) > max(len(clean) * 0.5, 3):
            continue
            
        ratio = difflib.SequenceMatcher(None, clean, indexed_word).ratio()
        if ratio >= threshold:
            matches.append(indexed_word)
    
    return matches


def add_legend_to_image(img, mode, doc_side):
    """เพิ่ม Legend ลงบนรูป"""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    
    legend_height = 60
    legend_width = 280
    margin = 10
    x = margin
    y = margin
    
    draw.rectangle(
        [x, y, x + legend_width, y + legend_height],
        fill=(255, 255, 255, 230),
        outline=(100, 100, 100),
        width=2
    )
    
    font = None
    font_small = None
    
    try:
        if IS_MAC:
            mac_fonts = [
                "/System/Library/Fonts/Supplemental/Thonburi.ttc",
                "/System/Library/Fonts/Thonburi.ttc",
                "/System/Library/Fonts/Helvetica.ttc",
            ]
            for font_path in mac_fonts:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, 14)
                    font_small = ImageFont.truetype(font_path, 12)
                    break
        else:
            font = ImageFont.truetype("arial.ttf", 14)
            font_small = ImageFont.truetype("arial.ttf", 12)
    except:
        pass
    
    if font is None:
        font = ImageFont.load_default()
        font_small = font
    
    if mode == 'same':
        draw.rectangle([x + 10, y + 15, x + 30, y + 30], fill=(0, 204, 0))
        draw.text((x + 40, y + 14), "= Matching content", fill=(0, 0, 0), font=font_small)
        draw.text((x + 10, y + 38), f"Document: {doc_side.upper()}", fill=(80, 80, 80), font=font_small)
    else:
        if doc_side == 'doc1':
            draw.rectangle([x + 10, y + 15, x + 30, y + 30], fill=(255, 77, 77))
            draw.text((x + 40, y + 14), "= Only in THIS document", fill=(0, 0, 0), font=font_small)
        else:
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
        
        if summary['doc1_unique_words'] > 0:
            match_pct = (summary.get('matched_doc1', 0) / summary['doc1_unique_words']) * 100
            print(f"📈 Match rate: {match_pct:.1f}%")
    else:
        print(f"🔴 Words only in Doc1: {summary.get('diff_doc1', 0)}")
        print(f"🟢 Words only in Doc2: {summary.get('diff_doc2', 0)}")
        print(f"📝 Sample differences Doc1: {summary.get('sample_doc1', [])[:5]}")
        print(f"📝 Sample differences Doc2: {summary.get('sample_doc2', [])[:5]}")
        
        total_unique = summary['doc1_unique_words'] + summary['doc2_unique_words']
        if total_unique > 0:
            diff_count = summary.get('diff_doc1', 0) + summary.get('diff_doc2', 0)
            diff_pct = (diff_count / total_unique) * 100
            print(f"📈 Difference rate: {diff_pct:.1f}%")
    
    print("="*60 + "\n")


# =====================================================
# Main Highlight Functions (เหมือนเดิม)
# =====================================================
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
    
    update_progress(f"\n📄 Document 1: {os.path.basename(pdf1_path)} ({doc1_pages} pages)", 5)
    
    if PARALLEL_OCR and FORCE_OCR and doc1_pages > 1:
        doc1.close()
        index1 = build_word_index_parallel(pdf1_path, progress_callback=lambda msg, pct: update_progress(msg, 5 + pct * 0.25))
        doc1 = fitz.open(pdf1_path)
    else:
        index1 = build_word_index(doc1, progress_callback=lambda msg, pct: update_progress(msg, 5 + pct * 0.25))
    
    update_progress(f"\n📄 Document 2: {os.path.basename(pdf2_path)} ({doc2_pages} pages)", 35)
    
    if PARALLEL_OCR and FORCE_OCR and doc2_pages > 1:
        doc2.close()
        index2 = build_word_index_parallel(pdf2_path, progress_callback=lambda msg, pct: update_progress(msg, 35 + pct * 0.25))
        doc2 = fitz.open(pdf2_path)
    else:
        index2 = build_word_index(doc2, progress_callback=lambda msg, pct: update_progress(msg, 35 + pct * 0.25))

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
        
        if i == 0:
            if img1:
                img1 = add_legend_to_image(img1, mode, 'doc1')
            if img2:
                img2 = add_legend_to_image(img2, mode, 'doc2')
            
        images.append((img1, img2))
        
        img_progress = 75 + (i + 1) / max_pages * 20
        update_progress(f"  → Generated page {i+1}/{max_pages}", img_progress)

    doc1.close()
    doc2.close()
    
    print_summary(summary)
    update_progress("✅ Comparison complete!", 100)
    
    return images, summary


def highlight_same_mode(doc1, doc2, index1, index2):
    """โหมด SAME: ไฮไลท์คำที่เหมือนกัน"""
    GREEN = (0, 0.8, 0)
    
    matched_in_doc1 = set()
    matched_in_doc2 = set()
    matched_words = []
    
    words2 = set(index2['by_word'].keys())
    normalized2 = {normalize_for_compare(w): w for w in words2}
    
    for clean_word, locations1 in index1['by_word'].items():
        matched_doc2_word = None
        
        if clean_word in index2['by_word']:
            matched_doc2_word = clean_word
        else:
            normalized_clean = normalize_for_compare(clean_word)
            if normalized_clean in normalized2:
                matched_doc2_word = normalized2[normalized_clean]
        
        if matched_doc2_word:
            matched_words.append(clean_word)
            
            for page_num, word in locations1:
                word_id = (page_num, word[0], word[1], word[4])
                if word_id not in matched_in_doc1:
                    matched_in_doc1.add(word_id)
                    page = doc1[page_num]
                    rect = fitz.Rect(word[:4])
                    page.draw_rect(rect, color=GREEN, fill=GREEN, fill_opacity=0.35, width=0)
            
            for page_num, word in index2['by_word'][matched_doc2_word]:
                word_id = (page_num, word[0], word[1], word[4])
                if word_id not in matched_in_doc2:
                    matched_in_doc2.add(word_id)
                    page = doc2[page_num]
                    rect = fitz.Rect(word[:4])
                    page.draw_rect(rect, color=GREEN, fill=GREEN, fill_opacity=0.35, width=0)
        else:
            matches_in_doc2 = fuzzy_match_word(clean_word, index2, threshold=0.60)
            
            if matches_in_doc2:
                matched_words.append(clean_word)
                
                for page_num, word in locations1:
                    word_id = (page_num, word[0], word[1], word[4])
                    if word_id not in matched_in_doc1:
                        matched_in_doc1.add(word_id)
                        page = doc1[page_num]
                        rect = fitz.Rect(word[:4])
                        page.draw_rect(rect, color=GREEN, fill=GREEN, fill_opacity=0.35, width=0)
                
                for page_num, word, score in matches_in_doc2:
                    word_id = (page_num, word[0], word[1], word[4])
                    if word_id not in matched_in_doc2:
                        matched_in_doc2.add(word_id)
                        page = doc2[page_num]
                        rect = fitz.Rect(word[:4])
                        page.draw_rect(rect, color=GREEN, fill=GREEN, fill_opacity=0.35, width=0)
    
    print(f"  → Matched unique words: {len(matched_words)}")
    print(f"  → Highlighted {len(matched_in_doc1)} words in Doc1, {len(matched_in_doc2)} words in Doc2")
    
    return {
        'matched_doc1': len(matched_in_doc1),
        'matched_doc2': len(matched_in_doc2),
        'matched_unique_words': len(matched_words),
        'sample_matched': matched_words[:10]
    }


def highlight_diff_mode(doc1, doc2, index1, index2):
    """โหมด DIFF: ไฮไลท์คำที่แตกต่างกัน"""
    RED = (1, 0.3, 0.3)
    GREEN = (0.3, 0.8, 0.3)
    
    words1 = set(index1['by_word'].keys())
    words2 = set(index2['by_word'].keys())
    
    normalized2 = {normalize_for_compare(w): w for w in words2}
    normalized1 = {normalize_for_compare(w): w for w in words1}
    
    def get_full_text(index):
        texts = []
        for page_num, words in index.get('by_page', {}).items():
            page_text = ' '.join(w[4] for w in words)
            texts.append(page_text)
        return ' '.join(texts)
    
    full_text1 = get_full_text(index1)
    full_text2 = get_full_text(index2)
    
    all_dates1 = extract_thai_dates(full_text1)
    all_dates2 = extract_thai_dates(full_text2)
    all_times1 = extract_times(full_text1)
    all_times2 = extract_times(full_text2)
    
    dates_are_same = sorted(all_dates1) == sorted(all_dates2) if all_dates1 and all_dates2 else True
    times_are_same = sorted(all_times1) == sorted(all_times2) if all_times1 and all_times2 else True
    
    print(f"  → [Date Check] Dates match: {dates_are_same}, Times match: {times_are_same}")
    
    words_only_in_doc1 = set()
    words_only_in_doc2 = set()
    
    for clean_word in words1:
        if clean_word in words2:
            continue
        
        normalized_clean = normalize_for_compare(clean_word)
        if normalized_clean in normalized2:
            continue
        
        original_text = ""
        if clean_word in index1['by_word'] and index1['by_word'][clean_word]:
            original_text = index1['by_word'][clean_word][0][1][4]
        
        if is_form_label(original_text) or is_form_label(clean_word):
            continue
        
        if dates_are_same and is_date_related_word(original_text):
            continue
        
        if times_are_same:
            word_times = extract_times(original_text)
            if word_times:
                continue
        
        if any(c.isdigit() for c in clean_word) or is_thai_number_word(original_text):
            if dates_are_same:
                if re.match(r'^25\d{2}$', clean_word):
                    continue
                if re.match(r'^0?[1-9]$|^[12]\d$|^3[01]$', clean_word):
                    if any(clean_word in d for d in all_dates1):
                        continue
            
            val1 = extract_number_value(clean_word)
            if val1:
                found_match = False
                for w2 in words2:
                    if numbers_are_equal(clean_word, w2):
                        found_match = True
                        break
                if not found_match:
                    words_only_in_doc1.add(clean_word)
            else:
                words_only_in_doc1.add(clean_word)
        else:
            matches = fuzzy_match_word(clean_word, index2, threshold=0.80)
            if not matches:
                words_only_in_doc1.add(clean_word)
    
    for clean_word in words2:
        if clean_word in words1:
            continue
        
        normalized_clean = normalize_for_compare(clean_word)
        if normalized_clean in normalized1:
            continue
        
        original_text = ""
        if clean_word in index2['by_word'] and index2['by_word'][clean_word]:
            original_text = index2['by_word'][clean_word][0][1][4]
        
        if is_form_label(original_text) or is_form_label(clean_word):
            continue
        
        if dates_are_same and is_date_related_word(original_text):
            continue
        
        if times_are_same:
            word_times = extract_times(original_text)
            if word_times:
                continue
        
        if any(c.isdigit() for c in clean_word) or is_thai_number_word(original_text):
            if dates_are_same:
                if re.match(r'^25\d{2}$', clean_word):
                    continue
                if re.match(r'^0?[1-9]$|^[12]\d$|^3[01]$', clean_word):
                    if any(clean_word in d for d in all_dates2):
                        continue
            
            val2 = extract_number_value(clean_word)
            if val2:
                found_match = False
                for w1 in words1:
                    if numbers_are_equal(clean_word, w1):
                        found_match = True
                        break
                if not found_match:
                    words_only_in_doc2.add(clean_word)
            else:
                words_only_in_doc2.add(clean_word)
        else:
            matches = fuzzy_match_word(clean_word, index1, threshold=0.80)
            if not matches:
                words_only_in_doc2.add(clean_word)
    
    print(f"  → Words only in Doc1: {len(words_only_in_doc1)}")
    print(f"  → Words only in Doc2: {len(words_only_in_doc2)}")
    
    count1 = 0
    for clean_word in words_only_in_doc1:
        for page_num, word in index1['by_word'][clean_word]:
            page = doc1[page_num]
            rect = fitz.Rect(word[:4])
            page.draw_rect(rect, color=RED, fill=RED, fill_opacity=0.4, width=0)
            count1 += 1
    
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


# =====================================================
# Database Mode Functions
# =====================================================
def highlight_text_differences_db(pdf_path, db_document, mode='diff', progress_callback=None):
    """Compare a PDF file against a document stored in database"""
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
    
    db_index = db_document.get('word_index', {'by_page': {}, 'by_word': {}})
    db_page_count = db_document.get('page_count', 0)
    
    update_progress(f"\n📄 Input File: {os.path.basename(pdf_path)} ({doc1_pages} pages)", 5)
    
    if PARALLEL_OCR and FORCE_OCR and doc1_pages > 1:
        doc1.close()
        index1 = build_word_index_parallel(pdf_path, progress_callback=lambda msg, pct: update_progress(msg, 5 + pct * 0.45))
        doc1 = fitz.open(pdf_path)
    else:
        index1 = build_word_index(doc1, progress_callback=lambda msg, pct: update_progress(msg, 5 + pct * 0.45))
    
    update_progress(f"\n📄 Reference (DB): {db_document.get('document_id', 'Unknown')} ({db_page_count} pages)", 55)
    update_progress(f"  → Using cached word index from database", 60)
    
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
    
    if mode == 'same':
        update_progress(f"\n🟢 Mode: SAME - Highlighting matching words...", 65)
        stats = highlight_same_mode_db(doc1, index1, db_index)
        summary.update(stats)
    else:
        update_progress(f"\n🔴 Mode: DIFF - Highlighting different words...", 65)
        stats = highlight_diff_mode_db(doc1, index1, db_index)
        summary.update(stats)
    
    update_progress("🖼️ Generating output images...", 75)
    images = []
    
    for i in range(doc1_pages):
        pix = doc1[i].get_pixmap(dpi=150)
        img1 = Image.open(io.BytesIO(pix.tobytes("png")))
        
        if i == 0:
            img1 = add_legend_to_image(img1, mode, 'doc1')
        
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
        'matched_doc2': 0,
        'matched_unique_words': len(matched_words),
        'sample_matched': matched_words[:10]
    }


def highlight_diff_mode_db(doc1, index1, db_index):
    """Mode DIFF: Highlight different words (file vs database)"""
    RED = (1, 0.3, 0.3)
    
    words1 = set(index1['by_word'].keys())
    db_words = db_index.get('by_word', {})
    words2 = set(db_words.keys())
    
    def get_full_text(index):
        texts = []
        for page_num, words in index.get('by_page', {}).items():
            page_text = ' '.join(w[4] for w in words)
            texts.append(page_text)
        return ' '.join(texts)
    
    full_text1 = get_full_text(index1)
    full_text2 = get_full_text(db_index)
    
    all_dates1 = extract_thai_dates(full_text1)
    all_dates2 = extract_thai_dates(full_text2)
    all_times1 = extract_times(full_text1)
    all_times2 = extract_times(full_text2)
    
    dates_are_same = sorted(all_dates1) == sorted(all_dates2) if all_dates1 and all_dates2 else True
    times_are_same = sorted(all_times1) == sorted(all_times2) if all_times1 and all_times2 else True
    
    print(f"  → [Date Check] Dates match: {dates_are_same}, Times match: {times_are_same}")
    
    words_only_in_doc1 = set()
    
    for clean_word in words1:
        if clean_word in words2:
            continue
        
        original_text = ""
        if clean_word in index1['by_word'] and index1['by_word'][clean_word]:
            original_text = index1['by_word'][clean_word][0][1][4]
        
        if dates_are_same and is_date_related_word(original_text):
            continue
        
        if times_are_same:
            word_times = extract_times(original_text)
            if word_times:
                continue
        
        if any(c.isdigit() for c in clean_word) or is_thai_number_word(original_text):
            if dates_are_same:
                if re.match(r'^25\d{2}$', clean_word):
                    continue
                if re.match(r'^0?[1-9]$|^[12]\d$|^3[01]$', clean_word):
                    continue
            words_only_in_doc1.add(clean_word)
        else:
            matches = fuzzy_match_word_index(clean_word, db_words, threshold=0.80)
            if not matches:
                words_only_in_doc1.add(clean_word)
    
    print(f"  → Words only in input file: {len(words_only_in_doc1)}")
    
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