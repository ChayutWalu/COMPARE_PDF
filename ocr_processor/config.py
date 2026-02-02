"""Configuration และ constants สำหรับ PDF comparison"""

from dataclasses import dataclass
import platform
import threading
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class OcrCompareConfig:
    """Configuration สำหรับ PDF comparison - แก้ไขได้เพื่อปรับพฤติกรรม"""
    ocr_confidence_threshold: float = 0.2
    ocr_dpi: int = 300
    output_dpi: int = 150
    min_word_length: int = 2
    fuzzy_threshold_default: float = 0.70
    fuzzy_threshold_strict: float = 0.80
    fuzzy_threshold_loose: float = 0.60
    debug_mode: bool = False
    force_ocr: bool = True
    use_process_pool: bool = False


config = OcrCompareConfig()

# Aliases
OCR_CONFIDENCE_THRESHOLD = config.ocr_confidence_threshold
OCR_DPI = config.ocr_dpi
OUTPUT_DPI = config.output_dpi
MIN_WORD_LENGTH = config.min_word_length
FUZZY_THRESHOLD_DEFAULT = config.fuzzy_threshold_default
FUZZY_THRESHOLD_STRICT = config.fuzzy_threshold_strict
FUZZY_THRESHOLD_LOOSE = config.fuzzy_threshold_loose
DEBUG_MODE = config.debug_mode
FORCE_OCR = config.force_ocr

IS_MAC = platform.system() == 'Darwin'
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'

ocr_lock = threading.Lock()

STOPWORDS = {
    "นาย", "นาง", "นางสาว",
    "ถนน", "ซอย", "แขวง", "เขต", "หมู่",
    "วันที่", "เดือน", "พ.ศ.", "บาท",
    "mr", "mrs", "miss", "ms",
    "road", "soi", "district", "province",
    "the", "a", "an", "is", "are", "of", "to", "in", "for", "on", "with"
}

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

USE_PROCESS_POOL = config.use_process_pool
PARALLEL_OCR = True
MAX_OCR_WORKERS = 2
if IS_MAC:
    MAX_OCR_WORKERS = 2
else:
    MAX_OCR_WORKERS = 2

# Print on first import
print(f"🖥️  Platform: {platform.system()} ({platform.machine()})")
print(f"⚙️  Parallel OCR: ThreadPool, {MAX_OCR_WORKERS} workers" if not USE_PROCESS_POOL else f"⚙️  Parallel OCR: ProcessPool, {MAX_OCR_WORKERS} workers")
