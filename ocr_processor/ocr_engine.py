"""
OCR Engine – PaddleOCR เป็นตัวอ่านหลักทั้งระบบ
=================================================
PaddleOCR อ่านข้อความ + ให้ bounding box (bbox) ในตัว
→ ใช้ทั้งเปรียบเทียบ + วาดไฮไลท์

Pipeline ต่อหน้า PDF:
  1. PaddleOCR (LOCAL) → ข้อความ + bounding boxes
  2. คืน word tuples (x0, y0, x1, y1, text, 0, 0, 0)

ไม่ต้องการ Ollama / Typhoon OCR / EasyOCR
"""

import io
import os
import threading

# PaddleOCR 3.x: skip slow model-source connectivity check on startup
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import fitz
import numpy as np
from PIL import Image

from .config import (
    config, logger, ocr_lock, IS_MAC,
    OCR_DPI, OCR_CONFIDENCE_THRESHOLD, FORCE_OCR,
    USE_PROCESS_POOL,
)

# ══════════════════════════════════════════════════════════
#  PaddleOCR – Lazy singleton (thread-safe)
# ══════════════════════════════════════════════════════════
_reader = None
_reader_lock = threading.Lock()


def get_ocr_reader():
    """Lazy initialization of PaddleOCR 3.x reader – thread-safe"""
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                from paddleocr import PaddleOCR

                logger.info("Initializing PaddleOCR 3.x (lazy load)...")
                try:
                    _reader = PaddleOCR(
                        lang='th',
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                    )
                    logger.info("✅ PaddleOCR 3.x initialized (Thai PP-OCRv5)")
                except Exception as e:
                    logger.error(f"PaddleOCR init error: {e}")
                    raise
    return _reader


# ══════════════════════════════════════════════════════════
#  Low-level OCR helpers
# ══════════════════════════════════════════════════════════

def ocr_single_image(img_np):
    """
    OCR a single numpy image – thread-safe.  (PaddleOCR 3.x API)

    Returns list of (bbox, text, confidence) tuples where
    bbox = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (4-point polygon).
    """
    with ocr_lock:
        results = list(get_ocr_reader().predict(img_np))

        converted = []
        if results:
            r = results[0]  # OCRResult (dict-like)
            polys = r.get('dt_polys', [])
            texts = r.get('rec_texts', [])
            scores = r.get('rec_scores', [])
            for poly, text, conf in zip(polys, texts, scores):
                # poly is a (4, 2) numpy array → convert to list of lists
                box = poly.tolist() if hasattr(poly, 'tolist') else poly
                converted.append((box, text, conf))
        return converted


def get_words_from_page_ocr(page):
    """
    ดึงคำและพิกัดจากหน้า PDF ด้วย PaddleOCR

    Returns:
        list of (x0, y0, x1, y1, text, 0, 0, 0) tuples
    """
    pix = page.get_pixmap(dpi=OCR_DPI)
    img_np = np.array(Image.open(io.BytesIO(pix.tobytes("png"))))

    results = ocr_single_image(img_np)

    scale_x = page.rect.width / pix.width
    scale_y = page.rect.height / pix.height

    words = []
    for (bbox, text, prob) in results:
        if prob <= OCR_CONFIDENCE_THRESHOLD or not text.strip():
            continue
        # bbox: [top-left, top-right, bottom-right, bottom-left]
        tl, tr, br, bl = bbox
        x0 = min(tl[0], bl[0]) * scale_x
        y0 = min(tl[1], tr[1]) * scale_y
        x1 = max(tr[0], br[0]) * scale_x
        y1 = max(bl[1], br[1]) * scale_y
        words.append((x0, y0, x1, y1, text, 0, 0, 0))
    return words


# ══════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════

def get_words_all(page, page_num=0, total_pages=1):
    """
    ดึงคำทั้งหมดจากหน้า PDF

    ลำดับ:
      1. PaddleOCR (ถ้า FORCE_OCR)
      2. Native text จาก PyMuPDF (ถ้า PDF มี text layer)
      3. PaddleOCR fallback (ถ้า native text น้อยเกินไป)
    """
    if page is None:
        return []

    # ── 1. Force OCR → ใช้ PaddleOCR โดยตรง ──
    if FORCE_OCR:
        try:
            return get_words_from_page_ocr(page)
        except Exception as e:
            print(f"  → PaddleOCR failed: {e}")
            return []

    # ── 2. Native text จาก PyMuPDF ──
    words = page.get_text("words")
    if len(words) < 5:
        try:
            words = get_words_from_page_ocr(page)
        except Exception as e:
            print(f"  → PaddleOCR failed: {e}")
            words = []
    else:
        page_width = page.rect.width
        page_height = page.rect.height
        words = [
            w for w in words
            if 0 <= w[0] < page_width and 0 <= w[1] < page_height
            and w[2] > w[0] and w[3] > w[1]
        ]
    return list(words)


def process_page_ocr(args):
    """
    Process single page สำหรับ parallel OCR (PaddleOCR)
    """
    page_num, pdf_path, page_rect_width, page_rect_height = args
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        words = get_words_from_page_ocr(page)
        doc.close()
        return page_num, words
    except Exception as e:
        logger.error(f"PaddleOCR failed for page {page_num}: {e}")
        return page_num, []


def extract_text_from_pdf(pdf_path):
    """
    แกะข้อความจาก PDF สำหรับส่งให้ LLM วิเคราะห์

    Returns:
        tuple: (ข้อความ, engine_name)
    """
    if not os.path.exists(pdf_path):
        logger.warning(f"PDF file not found: {pdf_path}")
        return "", ""

    doc = fitz.open(pdf_path)
    extracted_text = ""

    for i, page in enumerate(doc):
        # Try native text first
        text = page.get_text()
        if text.strip():
            extracted_text += f"--- Page {i+1} (Native) ---\n{text}\n"
            continue

        # Fall back to PaddleOCR 3.x
        try:
            pix = page.get_pixmap(dpi=OCR_DPI)
            img_np = np.array(Image.open(io.BytesIO(pix.tobytes("png"))))
            ocr_results = ocr_single_image(img_np)
            if ocr_results:
                texts = [text for (_, text, conf) in ocr_results if conf > OCR_CONFIDENCE_THRESHOLD]
                extracted_text += f"--- Page {i+1} (PaddleOCR) ---\n{chr(10).join(texts)}\n"
        except Exception as e:
            logger.warning(f"PaddleOCR failed page {i+1}: {e}")

    doc.close()
    return extracted_text, "paddleocr"
