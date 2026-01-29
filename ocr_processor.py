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
import platform
import logging
from decimal import Decimal, InvalidOperation

# =====================================================
# Logging Configuration
# =====================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================
# Constants - Magic Numbers Centralized
# =====================================================
OCR_CONFIDENCE_THRESHOLD = 0.2      # ค่าความมั่นใจขั้นต่ำของ OCR
OCR_DPI = 300                        # DPI สำหรับ OCR processing
OUTPUT_DPI = 150                     # DPI สำหรับ output images
MIN_WORD_LENGTH = 2                  # ความยาวขั้นต่ำของคำที่จะ index
FUZZY_THRESHOLD_DEFAULT = 0.70       # threshold สำหรับ fuzzy matching
FUZZY_THRESHOLD_STRICT = 0.80        # threshold สำหรับ strict matching
FUZZY_THRESHOLD_LOOSE = 0.60         # threshold สำหรับ loose matching

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
# เพราะเป็นส่วนของ template ไม่ใช่ข้อมูลที่ต้องเปรียบเทียบ
# แบ่งเป็น 2 กลุ่ม: exact match และ partial match

# คำที่ต้อง exact match เท่านั้น (คำสั้นที่อาจไปซ้อนกับคำอื่น)
FORM_LABELS_EXACT = {
    # หน่วย (ต้อง exact match เท่านั้น)
    "u", "hr", "น", "baht", "sq", "m2",
    # คำสั้น
    "to", "from", "no", "of", "as", "at", "on", "per",
    "sum", "vat", "tax", "net", "rate", "area", "code",
    "และ", "หรือ", "and", "or", "ที่", "ณ",
}

# คำที่ใช้ partial match ได้ (คำยาวที่ไม่ค่อยซ้อนกับข้อมูลจริง)
FORM_LABELS_PARTIAL = {
    # หัวข้อทั่วไป
    "เวลา", "time", "เริ่มต้น", "สิ้นสุด",
    "ตำแหน่ง", "จังหวัด", "อำเภอ", "ตำบล", "รหัสไปรษณีย์",
    "district", "province", "subdistrict", "block", "บล็อก",
    
    # หัวข้อกรมธรรม์ประกันภัย
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

# รวมทั้งหมดสำหรับ backward compatibility
FORM_LABELS = FORM_LABELS_EXACT | FORM_LABELS_PARTIAL

# รวม FORM_LABELS เข้ากับ STOPWORDS สำหรับการ filter
ALL_SKIP_WORDS = STOPWORDS | FORM_LABELS
# หมายเหตุ: ลบ "คุณ" ออกจาก STOPWORDS เพราะมักติดกับชื่อคน

# =====================================================
# EasyOCR Initialization - Lazy Loading for faster imports
# =====================================================
_reader = None
_reader_lock = threading.Lock()


def get_ocr_reader():
    """Lazy initialization of EasyOCR reader - thread-safe"""
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:  # Double-check locking
                logger.info("Initializing EasyOCR (lazy load)...")
                if IS_MAC:
                    logger.info("🍎 macOS detected - Using CPU mode")
                    _reader = easyocr.Reader(['th', 'en'], gpu=False)
                    logger.info("✅ EasyOCR initialized with CPU")
                else:
                    logger.info("Checking for CUDA GPU...")
                    try:
                        _reader = easyocr.Reader(['th', 'en'], gpu=True)
                        logger.info("✅ EasyOCR initialized with GPU (CUDA)")
                    except Exception as e:
                        logger.warning(f"⚠️ GPU not available: {e}")
                        _reader = easyocr.Reader(['th', 'en'], gpu=False)
                        logger.info("✅ EasyOCR initialized with CPU (fallback)")
    return _reader


# Backward compatibility - สำหรับ code ที่ยังใช้ reader ตรงๆ
class LazyReader:
    """Proxy class for lazy loading EasyOCR reader"""
    def __getattr__(self, name):
        return getattr(get_ocr_reader(), name)

reader = LazyReader()


def extract_text_from_pdf(pdf_path):
    """แกะข้อความสำหรับ LLM"""
    if not os.path.exists(pdf_path): 
        logger.warning(f"PDF file not found: {pdf_path}")
        return ""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path}: {e}")
        return ""

    extracted_text = ""
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            extracted_text += f"--- Page {i+1} (Native) ---\n{text}\n"
            continue
            
        try:
            pix = page.get_pixmap(dpi=OCR_DPI)
            img_np = np.array(Image.open(io.BytesIO(pix.tobytes("png"))))
            ocr_reader = get_ocr_reader()
            result_list = ocr_reader.readtext(img_np, detail=0, paragraph=True)
            extracted_text += f"--- Page {i+1} (EasyOCR) ---\n{chr(10).join(result_list)}\n"
        except Exception as e:
            logger.warning(f"OCR failed for page {i+1} of {pdf_path}: {e}")
    
    doc.close()
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


def normalize_leading_zeros(text):
    """
    Normalize leading zeros สำหรับวันที่ไทย - ทำงานทั้ง string
    "9กันยายน2568" → "09กันยายน2568"
    "1มกราคม2567" → "01มกราคม2567"
    "กันยายน9" → "กันยายน09" (กรณีวันอยู่หลังเดือน)
    """
    # Pattern 1: ตัวเลข 1 หลักที่อยู่ต้น string ตามด้วยตัวอักษรไทย (เดือน)
    normalized = re.sub(r'^(\d)([\u0E00-\u0E7F])', r'0\1\2', text)
    
    # Pattern 2: ตัวเลข 1 หลักที่อยู่หลังตัวอักษรไทย (กรณีวันอยู่หลังเดือน)
    normalized = re.sub(r'([\u0E00-\u0E7F])(\d)$', r'\g<1>0\2', normalized)
    
    # Pattern 3: ตัวเลข 1 หลักระหว่างตัวอักษรไทย (กลาง string)
    normalized = re.sub(r'([\u0E00-\u0E7F])(\d)([\u0E00-\u0E7F])', r'\g<1>0\2\3', normalized)
    
    return normalized


def normalize_for_compare(text):
    """Normalize text สำหรับเปรียบเทียบ - ลบ space, separators และ normalize leading zeros"""
    # ลบ space, /, -, . สำหรับ format ต่างๆ เช่น วันที่
    normalized = re.sub(r'[\s/\-\.]+', '', clean_text(text))
    
    # Normalize leading zeros สำหรับวันที่ไทย
    normalized = normalize_leading_zeros(normalized)
    
    return normalized


def normalize_date_text(text):
    """
    Normalize วันที่ format ต่างๆ ให้เหมือนกัน
    เช่น "09/กันยายน/2568" → "09กันยายน2568"
         "09 กันยายน 2568" → "09กันยายน2568"
    """
    # ลบ separators ทั้งหมด (/, -, space)
    return re.sub(r'[\s/\-]+', '', text)


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
    
    ใช้ Decimal สำหรับความแม่นยำในการเปรียบเทียบจำนวนเงิน
    """
    val1 = extract_number_value(num1_text)
    val2 = extract_number_value(num2_text)
    
    if val1 is None or val2 is None:
        return False
    
    v1, v2 = val1[0], val2[0]
    
    # ใช้ Decimal สำหรับความแม่นยำสูง (จำนวนเงินประกัน)
    try:
        d1 = Decimal(str(v1))
        d2 = Decimal(str(v2))
        
        # เปรียบเทียบ exact match สำหรับจำนวนเต็ม
        if d1 == d1.to_integral_value() and d2 == d2.to_integral_value():
            return d1.to_integral_value() == d2.to_integral_value()
        
        # สำหรับทศนิยม ใช้ relative tolerance 0.0001% (สำหรับ rounding errors)
        if d2 != 0:
            relative_diff = abs((d1 - d2) / d2)
            return relative_diff < Decimal('0.000001')  # 0.0001%
        else:
            return d1 == 0
    except (InvalidOperation, ZeroDivisionError):
        # Fallback to original comparison
        if isinstance(v1, float) or isinstance(v2, float):
            return abs(float(v1) - float(v2)) < 0.001
        else:
            return v1 == v2


def is_thai_number_word(text):
    """
    ตรวจสอบว่าข้อความมีจำนวนเงินที่เป็นตัวหนังสือภาษาไทยหรือไม่
    เช่น "หนึ่งหมื่นบาท", "สองแสนห้าหมื่น", "สามล้านบาทถ้วน"
    """
    # ตัวเลขภาษาไทย
    thai_digits = ['หนึ่ง', 'สอง', 'สาม', 'สี่', 'ห้า', 'หก', 'เจ็ด', 'แปด', 'เก้า', 'สิบ', 
                   'เอ็ด', 'ยี่', 'ศูนย์']
    # หน่วยนับภาษาไทย
    thai_units = ['ร้อย', 'พัน', 'หมื่น', 'แสน', 'ล้าน', 'สิบ']
    # คำที่เกี่ยวกับเงิน
    money_words = ['บาท', 'สตางค์', 'ถ้วน']
    
    text_lower = text.lower()
    
    # ตรวจสอบว่ามีตัวเลขภาษาไทยและหน่วยนับรวมกัน
    has_digit = any(digit in text for digit in thai_digits)
    has_unit = any(unit in text for unit in thai_units)
    has_money = any(word in text for word in money_words)
    
    # ถ้ามีตัวเลข+หน่วย หรือ มีคำเกี่ยวกับเงิน+ตัวเลข → เป็นจำนวนเงินตัวหนังสือ
    if (has_digit and has_unit) or (has_money and has_digit):
        return True
    
    # ตรวจสอบ pattern ที่เป็นจำนวนเงิน เช่น "หนึ่งหมื่น", "สองแสน"
    return False


# =====================================================
# [NEW] Date/Time Extraction and Comparison
# =====================================================
THAI_MONTHS = {
    'มกราคม': '01', 'กุมภาพันธ์': '02', 'มีนาคม': '03', 'เมษายน': '04',
    'พฤษภาคม': '05', 'มิถุนายน': '06', 'กรกฎาคม': '07', 'สิงหาคม': '08',
    'กันยายน': '09', 'ตุลาคม': '10', 'พฤศจิกายน': '11', 'ธันวาคม': '12',
    # Short forms with dots
    'ม.ค.': '01', 'ก.พ.': '02', 'มี.ค.': '03', 'เม.ย.': '04',
    'พ.ค.': '05', 'มิ.ย.': '06', 'ก.ค.': '07', 'ส.ค.': '08',
    'ก.ย.': '09', 'ต.ค.': '10', 'พ.ย.': '11', 'ธ.ค.': '12',
    # Short forms without dots (OCR อาจอ่านไม่มีจุด)
    'มค': '01', 'กพ': '02', 'มีค': '03', 'เมย': '04',
    'พค': '05', 'มิย': '06', 'กค': '07', 'สค': '08',
    'กย': '09', 'ตค': '10', 'พย': '11', 'ธค': '12'
}

# คำ keywords ที่บอกว่าเป็นบรรทัดเกี่ยวกับวันที่
DATE_KEYWORDS = [
    'ระยะเวลาประกันภัย', 'วันที่', 'วันทำสัญญา', 'สิ้นสุดวันที่', 'เริ่มต้นวันที่',
    'ตั้งแต่วันที่', 'ถึงวันที่', 'วันทำสัญญาประกันภัย', 'วันที่ทำสัญญา',
    'period of insurance', 'from', 'agreement made on', 'policy issued',
    'เริ่มต้น', 'สิ้นสุด', 'เวลา'
]


def extract_thai_dates(text):
    """
    แยกวันที่ภาษาไทยจาก text
    รองรับ: "09 กันยายน 2568", "09กันยายน2568", "09/กันยายน/2568"
    Returns: list of normalized date strings (DD-MM-YYYY)
    """
    dates = []
    
    for month_th, month_num in THAI_MONTHS.items():
        # Pattern: วัน เดือน ปี (with or without spaces/separators)
        pattern = rf'(\d{{1,2}})\s*[/\-]?\s*{re.escape(month_th)}\s*[/\-]?\s*(\d{{4}})'
        matches = re.findall(pattern, text)
        for day, year in matches:
            dates.append(f"{int(day):02d}-{month_num}-{year}")
    
    # Pattern: DD/MM/YYYY or DD-MM-YYYY
    pattern = r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})'
    matches = re.findall(pattern, text)
    for day, month, year in matches:
        dates.append(f"{int(day):02d}-{int(month):02d}-{year}")
    
    return dates


def extract_times(text):
    """
    แยกเวลาจาก text
    รองรับ: "16.30 น.", "16:30", "16.30"
    Returns: list of normalized time strings (HH:MM)
    """
    times = []
    
    # Pattern: HH.MM หรือ HH:MM (with optional น. or u.)
    pattern = r'(\d{1,2})[.:](\d{2})\s*(?:น\.?|u\.?)?'
    matches = re.findall(pattern, text)
    for hour, minute in matches:
        h = int(hour)
        if 0 <= h <= 24:  # Valid hour
            times.append(f"{h:02d}:{minute}")
    
    return times


def is_date_related_word(text):
    """
    ตรวจสอบว่าคำนี้เป็นส่วนหนึ่งของวันที่หรือไม่
    รองรับทั้งคำแยก เช่น "กันยายน", "2568", "09"
    และ full date string เช่น "09/กันยายน/2568", "09กันยายน2568"
    """
    # เป็นชื่อเดือนไทย
    if text in THAI_MONTHS:
        return True
    
    # เป็นปี พ.ศ. (2500-2600)
    if re.match(r'^25\d{2}$', text):
        return True
    
    # เป็นวันที่ (01-31)
    if re.match(r'^0?[1-9]$|^[12]\d$|^3[01]$', text):
        return True
    
    # [NEW] ตรวจสอบว่ามีชื่อเดือนไทยอยู่ใน text หรือไม่ (full date string)
    for month in THAI_MONTHS.keys():
        if month in text:
            # ถ้ามีเดือน + ตัวเลข (ปีหรือวัน) → เป็น date string
            if re.search(r'\d', text):
                return True
    
    # [NEW] Pattern: DD/MM/YYYY หรือ DD-MM-YYYY
    if re.match(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{4}$', text):
        return True
    
    return False


def is_date_related_context(text):
    """
    ตรวจสอบว่า text มี context เกี่ยวกับวันที่หรือไม่
    ใช้สำหรับตรวจสอบทั้งบรรทัด
    """
    text_lower = text.lower()
    return any(kw in text_lower or kw in text for kw in DATE_KEYWORDS)


def dates_match(text1, text2):
    """
    เปรียบเทียบว่าวันที่ใน 2 texts เหมือนกันหรือไม่
    Returns: True ถ้าวันที่เหมือนกัน, False ถ้าต่างกัน, None ถ้าไม่มีวันที่
    """
    dates1 = extract_thai_dates(text1)
    dates2 = extract_thai_dates(text2)
    
    if not dates1 and not dates2:
        return None  # ไม่มีวันที่ทั้งคู่
    
    if dates1 and dates2:
        return sorted(dates1) == sorted(dates2)
    
    return False  # มีฝั่งเดียว


def times_match(text1, text2):
    """
    เปรียบเทียบว่าเวลาใน 2 texts เหมือนกันหรือไม่
    """
    times1 = extract_times(text1)
    times2 = extract_times(text2)
    
    if not times1 and not times2:
        return None  # ไม่มีเวลาทั้งคู่
    
    if times1 and times2:
        return sorted(times1) == sorted(times2)
    
    return False  # มีฝั่งเดียว


def is_form_label(text):
    """
    ตรวจสอบว่าคำนี้เป็น form label/header หรือไม่
    ใช้สำหรับไม่ highlight คำที่เป็นส่วนของ template
    
    ใช้ 2 วิธี:
    1. EXACT match สำหรับคำสั้น (เพื่อไม่ให้ไปซ้อนกับคำอื่น)
    2. PARTIAL match สำหรับคำยาว (หัวข้อ form ที่ชัดเจน)
    """
    clean = clean_text(text)
    original_lower = text.lower().strip()
    
    if not clean:
        return False
    
    # 1. Exact match กับ FORM_LABELS_EXACT
    if clean in FORM_LABELS_EXACT:
        return True
    
    # 2. Exact match กับ FORM_LABELS_PARTIAL (ทั้งคำ)
    if clean in FORM_LABELS_PARTIAL:
        return True
    
    # 3. Partial match เฉพาะกับ FORM_LABELS_PARTIAL (คำยาวที่ปลอดภัย)
    # เฉพาะคำที่ยาว >= 4 ตัวอักษร เพื่อหลีกเลี่ยง false positive
    for label in FORM_LABELS_PARTIAL:
        if len(label) >= 4:  # เฉพาะ label ที่ยาวพอ
            if label in clean or label in original_lower:
                return True
    
    # 4. ตรวจสอบ pattern ของ form labels
    # เฉพาะคำยาวที่เริ่มต้นด้วย prefix ที่ชัดเจน
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
    """
    ตรวจสอบว่าคำนี้สำคัญพอที่จะ index และเปรียบเทียบหรือไม่
    index ทุกคำที่มีความยาว >= MIN_WORD_LENGTH ตัวอักษร
    """
    clean = clean_text(text)
    if not clean:
        return False
    
    # คำที่สั้นเกินไป (< MIN_WORD_LENGTH) ไม่ index
    if len(clean) < MIN_WORD_LENGTH:
        return False
    
    # ข้ามคำที่อยู่ใน stopwords (ไม่รวม FORM_LABELS เพราะต้อง index ไว้เพื่อ matching)
    if clean in STOPWORDS:
        return False
    
    # index ทุกคำที่เหลือ (รวมถึงประเภทกรมธรรม์, ชื่อ, ที่อยู่, วันที่ ฯลฯ)
    return True


def get_full_text_from_index(index):
    """
    สร้าง full text จาก word index
    ใช้สำหรับ date/time extraction
    """
    texts = []
    by_page = index.get('by_page', {})
    
    # Sort by page number to maintain order
    for page_num in sorted(by_page.keys()):
        words = by_page[page_num]
        page_text = ' '.join(w[4] for w in words)
        texts.append(page_text)
    
    return ' '.join(texts)


def ocr_single_image(img_np):
    """OCR รูปเดียว - thread-safe ด้วย lock"""
    with ocr_lock:
        return get_ocr_reader().readtext(img_np)


def _process_ocr_bbox(bbox, text, prob, scale_x, scale_y):
    """
    Helper function สำหรับ process OCR bounding box
    ลด code duplication ระหว่าง get_words_from_page_ocr และ process_page_ocr
    """
    if prob <= OCR_CONFIDENCE_THRESHOLD or not text.strip():
        return None
    
    (tl, tr, br, bl) = bbox
    x_min = min(tl[0], bl[0])
    y_min = min(tl[1], tr[1])
    x_max = max(tr[0], br[0])
    y_max = max(bl[1], br[1])
    
    x0 = x_min * scale_x
    y0 = y_min * scale_y
    x1 = x_max * scale_x
    y1 = y_max * scale_y
    
    return (x0, y0, x1, y1, text, 0, 0, 0)


def get_words_from_page_ocr(page):
    """ดึงคำและพิกัดด้วย EasyOCR"""
    pix = page.get_pixmap(dpi=OCR_DPI) 
    img_np = np.array(Image.open(io.BytesIO(pix.tobytes("png"))))
    
    # ใช้ thread-safe OCR
    results = ocr_single_image(img_np)
    words = []
    
    scale_x = page.rect.width / pix.width
    scale_y = page.rect.height / pix.height

    for (bbox, text, prob) in results:
        word_tuple = _process_ocr_bbox(bbox, text, prob, scale_x, scale_y)
        if word_tuple:
            words.append(word_tuple)
    return words


def process_page_ocr(args):
    """Process single page สำหรับ parallel OCR"""
    page_num, pdf_path, page_rect_width, page_rect_height = args
    
    try:
        # เปิด PDF ใหม่ใน thread นี้
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        pix = page.get_pixmap(dpi=OCR_DPI)
        img_np = np.array(Image.open(io.BytesIO(pix.tobytes("png"))))
        
        # OCR with lock
        results = ocr_single_image(img_np)
        
        words = []
        scale_x = page.rect.width / pix.width
        scale_y = page.rect.height / pix.height
        
        for (bbox, text, prob) in results:
            word_tuple = _process_ocr_bbox(bbox, text, prob, scale_x, scale_y)
            if word_tuple:
                words.append(word_tuple)
        
        doc.close()
        return page_num, words
        
    except Exception as e:
        logger.error(f"OCR failed for page {page_num}: {e}")
        return page_num, []


# =====================================================
# Global Settings - Auto-configure based on platform
# =====================================================
FORCE_OCR = True  # เปลี่ยนเป็น False ถ้าต้องการใช้ native text

# Parallel OCR settings - ปรับตาม platform
if IS_MAC:
    # macOS: ปิด parallel เพราะอาจมี threading issues
    PARALLEL_OCR = False
    MAX_OCR_WORKERS = 1
    print("⚙️  Parallel OCR: Disabled (macOS)")
else:
    # Windows/Linux: เปิด parallel ได้
    PARALLEL_OCR = True
    MAX_OCR_WORKERS = 2
    print(f"⚙️  Parallel OCR: Enabled ({MAX_OCR_WORKERS} workers)")


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


def fuzzy_match_word(word, word_index, threshold=FUZZY_THRESHOLD_DEFAULT):
    """หาคำที่คล้ายกันใน index - รองรับภาษาไทยที่ OCR อ่านวรรณยุกต์ต่างกัน"""
    clean = clean_text(word)
    normalized = normalize_for_compare(word)
    normalized_no_tone = normalize_thai_tones(normalized)  # ลบวรรณยุกต์
    
    if not clean or len(clean) < MIN_WORD_LENGTH:
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
            pix = doc1[i].get_pixmap(dpi=OUTPUT_DPI)
            img1 = Image.open(io.BytesIO(pix.tobytes("png")))
        if i < len(doc2):
            pix = doc2[i].get_pixmap(dpi=OUTPUT_DPI)
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
        pix = doc1[i].get_pixmap(dpi=OUTPUT_DPI)
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
    
    # สร้าง normalized lookup สำหรับ O(1) matching
    normalized2 = {normalize_for_compare(w): w for w in words2}
    
    # สร้าง full text สำหรับ date comparison
    full_text1 = get_full_text_from_index(index1)
    full_text2 = get_full_text_from_index(db_index)
    
    # Extract all dates from both documents
    all_dates1 = extract_thai_dates(full_text1)
    all_dates2 = extract_thai_dates(full_text2)
    all_times1 = extract_times(full_text1)
    all_times2 = extract_times(full_text2)
    
    # Check if dates match globally
    dates_are_same = sorted(all_dates1) == sorted(all_dates2) if all_dates1 and all_dates2 else True
    times_are_same = sorted(all_times1) == sorted(all_times2) if all_times1 and all_times2 else True
    
    print(f"  → [Date Check] Dates match: {dates_are_same}, Times match: {times_are_same}")
    
    # Pre-compute normalized number values สำหรับ O(1) lookup
    number_values2 = {}
    for w2 in words2:
        val = extract_number_value(w2)
        if val:
            # เก็บ normalized value สำหรับ lookup
            normalized_val = smart_normalize_number(w2)
            number_values2[normalized_val] = w2
    
    words_only_in_doc1 = set()
    
    for clean_word in words1:
        if clean_word in words2:
            continue
        
        # ลอง normalized match
        normalized_clean = normalize_for_compare(clean_word)
        if normalized_clean in normalized2:
            continue
        
        # ดึง original text จาก index เพื่อตรวจสอบ
        original_text = ""
        if clean_word in index1['by_word'] and index1['by_word'][clean_word]:
            original_text = index1['by_word'][clean_word][0][1][4]
        
        # [เพิ่ม] ถ้าเป็น form label → ไม่ highlight
        if is_form_label(original_text) or is_form_label(clean_word):
            continue
        
        # ถ้าเป็นคำที่เกี่ยวกับวันที่ และวันที่ทั้งสองเอกสารเหมือนกัน → ข้าม
        if dates_are_same and is_date_related_word(original_text):
            continue
        
        # ถ้าเป็นเวลา และเวลาทั้งสองเอกสารเหมือนกัน → ข้าม
        if times_are_same:
            word_times = extract_times(original_text)
            if word_times:
                continue
        
        # Numbers and Thai number words
        if any(c.isdigit() for c in clean_word) or is_thai_number_word(original_text):
            if dates_are_same:
                if re.match(r'^25\d{2}$', clean_word):
                    continue
                if re.match(r'^0?[1-9]$|^[12]\d$|^3[01]$', clean_word):
                    continue
            
            # ใช้ pre-computed lookup แทน O(n) loop
            normalized_num = smart_normalize_number(clean_word)
            if normalized_num not in number_values2:
                words_only_in_doc1.add(clean_word)
        else:
            matches = fuzzy_match_word_index(clean_word, db_words, threshold=FUZZY_THRESHOLD_STRICT)
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


def fuzzy_match_word_index(word, word_index_by_word, threshold=FUZZY_THRESHOLD_DEFAULT):
    """Fuzzy match against a by_word index (dict of word -> locations)"""
    clean = clean_text(word)
    normalized = normalize_for_compare(word)
    normalized_no_tone = normalize_thai_tones(normalized)
    
    if not clean or len(clean) < MIN_WORD_LENGTH:
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
    
    # ใช้ font ตาม platform
    font = None
    font_small = None
    
    try:
        if IS_MAC:
            # macOS: ใช้ Thai font ที่มีในระบบ
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
            # Windows/Linux: ใช้ Arial
            font = ImageFont.truetype("arial.ttf", 14)
            font_small = ImageFont.truetype("arial.ttf", 12)
    except:
        pass
    
    # Fallback to default font
    if font is None:
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
    """โหมด SAME: ไฮไลท์คำที่เหมือนกัน (รวม normalized match สำหรับวันที่/ตัวเลข)"""
    GREEN = (0, 0.8, 0)
    
    matched_in_doc1 = set()
    matched_in_doc2 = set()
    matched_words = []
    
    # สร้าง normalized lookup
    words2 = set(index2['by_word'].keys())
    normalized2 = {normalize_for_compare(w): w for w in words2}
    
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
        matched_doc2_word = None
        
        # 1. ลองหา exact match ก่อน
        if clean_word in index2['by_word']:
            matched_doc2_word = clean_word
        else:
            # 2. [NEW] ลอง normalized match (สำหรับวันที่/ตัวเลขที่ต่าง format)
            normalized_clean = normalize_for_compare(clean_word)
            if normalized_clean in normalized2:
                matched_doc2_word = normalized2[normalized_clean]
        
        if matched_doc2_word:
            matched_words.append(clean_word)
            
            # ไฮไลท์ใน doc1
            for page_num, word in locations1:
                word_id = (page_num, word[0], word[1], word[4])
                if word_id not in matched_in_doc1:
                    matched_in_doc1.add(word_id)
                    page = doc1[page_num]
                    rect = fitz.Rect(word[:4])
                    page.draw_rect(rect, color=GREEN, fill=GREEN, fill_opacity=0.35, width=0)
            
            # ไฮไลท์ใน doc2 (ใช้ matched_doc2_word)
            for page_num, word in index2['by_word'][matched_doc2_word]:
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
    """โหมด DIFF: ไฮไลท์คำที่แตกต่างกัน - ใช้ Normalized Match สำหรับตัวเลข/วันที่"""
    RED = (1, 0.3, 0.3)       # มีใน Doc1 แต่ไม่มีใน Doc2
    GREEN = (0.3, 0.8, 0.3)   # มีใน Doc2 แต่ไม่มีใน Doc1
    
    # สร้าง set ของคำ
    words1 = set(index1['by_word'].keys())
    words2 = set(index2['by_word'].keys())
    
    # สร้าง normalized lookup สำหรับเปรียบเทียบวันที่/ตัวเลขที่ต่าง format
    normalized2 = {normalize_for_compare(w): w for w in words2}
    normalized1 = {normalize_for_compare(w): w for w in words1}
    
    # Pre-compute normalized number values สำหรับ O(1) lookup แทน O(n) loop
    number_values1 = {}
    number_values2 = {}
    for w1 in words1:
        val = extract_number_value(w1)
        if val:
            number_values1[smart_normalize_number(w1)] = w1
    for w2 in words2:
        val = extract_number_value(w2)
        if val:
            number_values2[smart_normalize_number(w2)] = w2
    
    # ใช้ module-level function แทน local function
    full_text1 = get_full_text_from_index(index1)
    full_text2 = get_full_text_from_index(index2)
    
    # Extract all dates from both documents
    all_dates1 = extract_thai_dates(full_text1)
    all_dates2 = extract_thai_dates(full_text2)
    all_times1 = extract_times(full_text1)
    all_times2 = extract_times(full_text2)
    
    # Check if dates match globally
    dates_are_same = sorted(all_dates1) == sorted(all_dates2) if all_dates1 and all_dates2 else True
    times_are_same = sorted(all_times1) == sorted(all_times2) if all_times1 and all_times2 else True
    
    print(f"  → [Date Check] Doc1 dates: {all_dates1[:10]}")
    print(f"  → [Date Check] Doc2 dates: {all_dates2[:10]}")
    print(f"  → [Date Check] Dates match: {dates_are_same}, Times match: {times_are_same}")
    
    # [DEBUG] แสดงตัวอย่าง normalized words
    sample_normalized1 = list(normalized1.items())[:5]
    sample_normalized2 = list(normalized2.items())[:5]
    print(f"  → [Normalize Check] Sample Doc1: {sample_normalized1}")
    print(f"  → [Normalize Check] Sample Doc2: {sample_normalized2}")
    
    words_only_in_doc1 = set()
    words_only_in_doc2 = set()
    
    # หาคำที่มีเฉพาะใน Doc1
    for clean_word in words1:
        # Exact match ก่อน
        if clean_word in words2:
            continue
        
        # ลอง normalized match (สำหรับวันที่/ตัวเลขที่ต่าง format)
        normalized_clean = normalize_for_compare(clean_word)
        if normalized_clean in normalized2:
            continue  # มี match ใน doc2 หลัง normalize → ไม่ใช่ความต่าง
        
        # ดึง original text จาก index เพื่อตรวจสอบ
        original_text = ""
        if clean_word in index1['by_word'] and index1['by_word'][clean_word]:
            original_text = index1['by_word'][clean_word][0][1][4]  # เอา text จาก word tuple
        
        # ถ้าเป็น form label → ไม่ highlight
        if is_form_label(original_text) or is_form_label(clean_word):
            continue
        
        # ถ้าเป็นคำที่เกี่ยวกับวันที่ และวันที่ทั้งสองเอกสารเหมือนกัน → ข้าม
        if dates_are_same and is_date_related_word(original_text):
            continue
        
        # ถ้าเป็นเวลา และเวลาทั้งสองเอกสารเหมือนกัน → ข้าม
        if times_are_same:
            word_times = extract_times(original_text)
            if word_times:
                continue
        
        # ถ้าเป็นตัวเลข หรือ จำนวนเงินตัวหนังสือ → ลองเทียบเป็นตัวเลข
        if any(c.isdigit() for c in clean_word) or is_thai_number_word(original_text):
            # ตรวจสอบว่าเป็นส่วนของวันที่หรือไม่
            if dates_are_same:
                # ถ้าตัวเลขนี้เป็นปี (2500-2600) หรือ วัน (1-31) → ข้าม
                if re.match(r'^25\d{2}$', clean_word):
                    continue
                if re.match(r'^0?[1-9]$|^[12]\d$|^3[01]$', clean_word):
                    # เช็คว่าเลขนี้อยู่ในวันที่หรือไม่
                    if any(clean_word in d for d in all_dates1):
                        continue
            
            # ใช้ pre-computed lookup O(1) แทน O(n) loop
            normalized_num = smart_normalize_number(clean_word)
            if normalized_num not in number_values2:
                words_only_in_doc1.add(clean_word)
        else:
            # ถ้าไม่ใช่ตัวเลข → ลอง fuzzy match
            matches = fuzzy_match_word(clean_word, index2, threshold=FUZZY_THRESHOLD_STRICT)
            if not matches:
                words_only_in_doc1.add(clean_word)
    
    # หาคำที่มีเฉพาะใน Doc2
    for clean_word in words2:
        if clean_word in words1:
            continue
        
        # ลอง normalized match
        normalized_clean = normalize_for_compare(clean_word)
        if normalized_clean in normalized1:
            continue
        
        # ดึง original text จาก index เพื่อตรวจสอบ
        original_text = ""
        if clean_word in index2['by_word'] and index2['by_word'][clean_word]:
            original_text = index2['by_word'][clean_word][0][1][4]  # เอา text จาก word tuple
        
        # ถ้าเป็น form label → ไม่ highlight
        if is_form_label(original_text) or is_form_label(clean_word):
            continue
        
        # ถ้าเป็นคำที่เกี่ยวกับวันที่ และวันที่ทั้งสองเอกสารเหมือนกัน → ข้าม
        if dates_are_same and is_date_related_word(original_text):
            continue
        
        # ถ้าเป็นเวลา และเวลาทั้งสองเอกสารเหมือนกัน → ข้าม
        if times_are_same:
            word_times = extract_times(original_text)
            if word_times:
                continue
        
        if any(c.isdigit() for c in clean_word) or is_thai_number_word(original_text):
            # ตรวจสอบว่าเป็นส่วนของวันที่หรือไม่
            if dates_are_same:
                if re.match(r'^25\d{2}$', clean_word):
                    continue
                if re.match(r'^0?[1-9]$|^[12]\d$|^3[01]$', clean_word):
                    if any(clean_word in d for d in all_dates2):
                        continue
            
            # ใช้ pre-computed lookup O(1) แทน O(n) loop
            normalized_num = smart_normalize_number(clean_word)
            if normalized_num not in number_values1:
                words_only_in_doc2.add(clean_word)
        else:
            matches = fuzzy_match_word(clean_word, index1, threshold=FUZZY_THRESHOLD_STRICT)
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