"""OCR processing ด้วย EasyOCR"""

import io
import os
import easyocr
import fitz
import numpy as np
from PIL import Image

from .config import (
    config, logger, ocr_lock, IS_MAC,
    OCR_DPI, OCR_CONFIDENCE_THRESHOLD, FORCE_OCR,
    USE_PROCESS_POOL,
)

_reader = None
_reader_lock = __import__('threading').Lock()


def get_ocr_reader():
    """Lazy initialization of EasyOCR reader - thread-safe"""
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:
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


class LazyReader:
    """Proxy class for lazy loading EasyOCR reader"""

    def __getattr__(self, name):
        return getattr(get_ocr_reader(), name)


reader = LazyReader()


def _process_ocr_bbox(bbox, text, prob, scale_x, scale_y):
    """Helper สำหรับ process OCR bounding box"""
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


def ocr_single_image(img_np):
    """OCR รูปเดียว - thread-safe ด้วย lock"""
    with ocr_lock:
        return get_ocr_reader().readtext(img_np)


def get_words_from_page_ocr(page):
    """ดึงคำและพิกัดด้วย EasyOCR"""
    pix = page.get_pixmap(dpi=OCR_DPI)
    img_np = np.array(Image.open(io.BytesIO(pix.tobytes("png"))))
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
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        pix = page.get_pixmap(dpi=OCR_DPI)
        img_np = np.array(Image.open(io.BytesIO(pix.tobytes("png"))))
        if USE_PROCESS_POOL:
            results = get_ocr_reader().readtext(img_np)
        else:
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
