"""
OCR Processor - PDF Comparison (Typhoon OCR เป็นตัวอ่านหลัก)
=============================================================
ใช้ Typhoon OCR 1.5 อ่านข้อความจริง ๆ ทั้งระบบ (เปรียบเทียบ + ไฮไลท์ + สรุป)

- อ่านข้อความ: Typhoon OCR 1.5 via Ollama (LOCAL) ← ตัวอ่านหลัก
- ตำแหน่งคำ:   EasyOCR (bbox สำหรับวาดไฮไลท์ เพราะ Typhoon ไม่คืน bbox)
- สรุป/เปรียบเทียบ: Typhoon Qwen v2.5-30b (API) — llm_client.py

Modules:
- config: Configuration, constants
- text_utils: clean_text, normalize_*
- number_utils: ตัวเลข, จำนวนเงิน
- date_utils: วันที่ไทย
- word_analysis: is_form_label, is_significant
- ocr_engine: Typhoon OCR (local) + EasyOCR, extract_text_from_pdf
- indexing: build_word_index
- fuzzy_matching: fuzzy_match_word
- highlighting: highlight_same_mode, highlight_diff_mode
- comparison: highlight_text_differences
- ui_helpers: add_legend_to_image, print_summary
"""

# Config & constants
from .config import (
    config, OcrCompareConfig,
    OCR_CONFIDENCE_THRESHOLD, OCR_DPI, OUTPUT_DPI, MIN_WORD_LENGTH,
    FUZZY_THRESHOLD_DEFAULT, FUZZY_THRESHOLD_STRICT, FUZZY_THRESHOLD_LOOSE,
    DEBUG_MODE, FORCE_OCR, PARALLEL_OCR, MAX_OCR_WORKERS, USE_PROCESS_POOL,
    STOPWORDS, FORM_LABELS, FORM_LABELS_EXACT, FORM_LABELS_PARTIAL,
    IS_MAC, IS_WINDOWS, IS_LINUX,
    OLLAMA_BASE_URL, OLLAMA_OCR_MODEL,
)

# OCR Engine
from .ocr_engine import (
    get_ocr_reader, reader, LazyReader,
    extract_text_from_pdf, get_words_from_page_ocr, get_words_all,
    ocr_single_image, process_page_ocr,
)

# Indexing
from .indexing import build_word_index, build_word_index_parallel

# Main API - สิ่งที่ gui, main, upload_reference ใช้
from .comparison import highlight_text_differences, highlight_text_differences_db

# Utils (สำหรับ advanced usage)
from .text_utils import clean_text, normalize_for_compare, normalize_thai_tones
from .number_utils import smart_normalize_number, extract_number_value, numbers_are_equal, is_thai_number_word
from .date_utils import extract_thai_dates, extract_times, is_date_related_word, dates_match, times_match
from .word_analysis import is_form_label, is_significant, get_full_text_from_index, get_first_original_text
from .fuzzy_matching import fuzzy_match_word, fuzzy_match_word_index
from .ui_helpers import add_legend_to_image, print_summary

__all__ = [
    'config', 'OcrCompareConfig',
    'highlight_text_differences', 'highlight_text_differences_db',
    'extract_text_from_pdf', 'build_word_index', 'build_word_index_parallel',
    'PARALLEL_OCR', 'FORCE_OCR',
    'clean_text', 'normalize_for_compare', 'get_ocr_reader', 'reader',
    'add_legend_to_image', 'print_summary',
]
