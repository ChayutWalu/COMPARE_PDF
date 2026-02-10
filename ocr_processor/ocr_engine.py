"""
OCR Engine – Typhoon OCR 1.5 เป็นตัวอ่านหลักทั้งระบบ
=====================================================
Typhoon OCR อ่านข้อความ (คุณภาพภาษาไทยดีกว่า)
+ EasyOCR ให้ตำแหน่งคำ (bbox) สำหรับวาดไฮไลท์
= Hybrid words ที่ใช้ทั้งเปรียบเทียบ + ไฮไลท์

Pipeline ต่อหน้า PDF:
  1. Typhoon OCR (Ollama local) → ข้อความคุณภาพสูง
  2. EasyOCR → bounding boxes (ตำแหน่งคำบนหน้า)
  3. จับคู่ข้อความ Typhoon กับตำแหน่ง EasyOCR
  4. คืน word tuples (x0, y0, x1, y1, text, ...) ที่ใช้ TEXT จาก Typhoon

ถ้า Ollama ไม่พร้อม → fallback เป็น EasyOCR ล้วนเหมือนเดิม
"""

import io
import os
import re
import base64
import easyocr
import fitz
import numpy as np
from PIL import Image
from openai import OpenAI
from difflib import SequenceMatcher

from .config import (
    config, logger, ocr_lock, IS_MAC,
    OCR_DPI, OCR_CONFIDENCE_THRESHOLD, FORCE_OCR,
    USE_PROCESS_POOL,
    OLLAMA_BASE_URL, OLLAMA_OCR_MODEL,
)

# ══════════════════════════════════════════════════════════
#  Ollama (Typhoon OCR) – ตรวจสอบครั้งเดียวตอน import
# ══════════════════════════════════════════════════════════
_ollama_available = None  # None = ยังไม่ได้เช็ค


def _check_ollama():
    """เช็คว่า Ollama + Typhoon OCR พร้อมใช้งานหรือไม่ (เช็คครั้งเดียว)"""
    global _ollama_available
    if _ollama_available is not None:
        return _ollama_available
    try:
        client = _get_ollama_client()
        models = client.models.list()
        model_ids = [m.id for m in models.data] if hasattr(models, 'data') else []
        _ollama_available = True
        logger.info(f"✅ Ollama พร้อมใช้งาน (models: {len(model_ids)})")
    except Exception as e:
        _ollama_available = False
        logger.warning(f"⚠️ Ollama ไม่พร้อม ({e}) → จะใช้ EasyOCR แทน")
    return _ollama_available


def _get_ollama_client():
    """สร้าง OpenAI client ที่ชี้ไป Ollama localhost (timeout 120s ต่อ request)"""
    import httpx
    return OpenAI(
        api_key="ollama",
        base_url=OLLAMA_BASE_URL,
        timeout=httpx.Timeout(120.0, connect=10.0),
    )


def _typhoon_ocr_page(page, page_num=0, total_pages=1):
    """
    อ่านข้อความจากหน้า PDF ด้วย Typhoon OCR 1.5 (Ollama local)
    คืนค่า: ข้อความ (str) หรือ None ถ้าล้มเหลว
    """
    try:
        client = _get_ollama_client()
        pix = page.get_pixmap(dpi=OCR_DPI)
        img_base64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

        print(f"  → [TYPHOON] OCR page {page_num}/{total_pages} (กำลังอ่าน อาจใช้เวลา 10-30 วินาที)...")
        logger.info(f"  → [TYPHOON] OCR page {page_num}/{total_pages}...")
        response = client.chat.completions.create(
            model=OLLAMA_OCR_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": "Extract all text from this document image. Return the text in structured markdown format.",
                        },
                    ],
                }
            ],
            max_tokens=4096,
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.warning(f"  → Typhoon OCR failed page {page_num}: {e}")
        return None


# ══════════════════════════════════════════════════════════
#  EasyOCR – ใช้สำหรับ bounding boxes
# ══════════════════════════════════════════════════════════
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
                    _reader = easyocr.Reader(['th', 'en'], gpu=False)
                    logger.info("✅ EasyOCR initialized (CPU)")
                else:
                    try:
                        _reader = easyocr.Reader(['th', 'en'], gpu=True)
                        logger.info("✅ EasyOCR initialized (GPU)")
                    except Exception:
                        _reader = easyocr.Reader(['th', 'en'], gpu=False)
                        logger.info("✅ EasyOCR initialized (CPU fallback)")
    return _reader


class LazyReader:
    """Proxy class for lazy loading EasyOCR reader"""
    def __getattr__(self, name):
        return getattr(get_ocr_reader(), name)


reader = LazyReader()


def ocr_single_image(img_np):
    """OCR รูปเดียว - thread-safe"""
    with ocr_lock:
        return get_ocr_reader().readtext(img_np)


def _easyocr_bboxes(page):
    """ดึง bounding boxes + ข้อความจาก EasyOCR สำหรับหน้าเดียว"""
    pix = page.get_pixmap(dpi=OCR_DPI)
    img_np = np.array(Image.open(io.BytesIO(pix.tobytes("png"))))
    results = ocr_single_image(img_np)
    scale_x = page.rect.width / pix.width
    scale_y = page.rect.height / pix.height
    bboxes = []
    for (bbox, text, prob) in results:
        if prob <= OCR_CONFIDENCE_THRESHOLD or not text.strip():
            continue
        (tl, tr, br, bl) = bbox
        x0 = min(tl[0], bl[0]) * scale_x
        y0 = min(tl[1], tr[1]) * scale_y
        x1 = max(tr[0], br[0]) * scale_x
        y1 = max(bl[1], br[1]) * scale_y
        bboxes.append((x0, y0, x1, y1, text))
    return bboxes


# ══════════════════════════════════════════════════════════
#  Hybrid: Typhoon text + EasyOCR bbox = word tuples
# ══════════════════════════════════════════════════════════

def _clean_markdown(text):
    """ลบ markdown formatting ออกจากข้อความ Typhoon OCR"""
    text = re.sub(r'#+\s*', '', text)          # headers
    text = re.sub(r'\*{1,3}', '', text)        # bold/italic
    text = re.sub(r'_{1,3}', ' ', text)        # underscores
    text = re.sub(r'`+', '', text)             # code
    text = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', text)  # links
    text = re.sub(r'[|>]', ' ', text)          # table/blockquote
    text = re.sub(r'---+', ' ', text)          # hr
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _word_similarity(s1, s2):
    """คำนวณความคล้ายกัน 0.0–1.0"""
    if not s1 or not s2:
        return 0.0
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def _find_typhoon_match(easyocr_word, typhoon_words):
    """
    หาคำจาก Typhoon ที่ตรงกับคำ EasyOCR มากที่สุด
    คืน Typhoon word ถ้าเจอ (similarity > 0.6) หรือ None
    """
    easyocr_clean = easyocr_word.strip()
    if not easyocr_clean or len(easyocr_clean) < 2:
        return None

    best_match = None
    best_score = 0.6  # threshold

    for tw in typhoon_words:
        if tw.lower() == easyocr_clean.lower():
            return tw  # exact match

        score = _word_similarity(easyocr_clean, tw)
        if score > best_score:
            best_score = score
            best_match = tw

    return best_match


def _build_hybrid_words(page, page_num=0, total_pages=1):
    """
    อ่านด้วย Typhoon OCR + ตำแหน่งจาก EasyOCR → word tuples

    Returns:
        list of (x0, y0, x1, y1, text, 0, 0, 0) หรือ None ถ้า Typhoon ไม่ได้
    """
    # 1. ── Typhoon OCR อ่านข้อความ ──
    typhoon_text = _typhoon_ocr_page(page, page_num, total_pages)
    if not typhoon_text or not typhoon_text.strip():
        return None

    # 2. ── แยกคำจาก Typhoon text ──
    clean_text = _clean_markdown(typhoon_text)
    typhoon_words = [w for w in re.split(r'\s+', clean_text) if w.strip()]
    if not typhoon_words:
        return None

    # 3. ── EasyOCR ให้ตำแหน่ง (bboxes) ──
    logger.info(f"  → [EASYOCR] Getting bboxes page {page_num}/{total_pages}...")
    easyocr_bboxes = _easyocr_bboxes(page)
    if not easyocr_bboxes:
        return None

    # 4. ── จับคู่: ใช้ Typhoon text + EasyOCR position ──
    typhoon_set = set(typhoon_words)
    words = []
    matched_typhoon = 0

    for (x0, y0, x1, y1, easyocr_text) in easyocr_bboxes:
        better_text = _find_typhoon_match(easyocr_text, typhoon_set)
        if better_text:
            text = better_text
            matched_typhoon += 1
        else:
            text = easyocr_text  # ใช้ EasyOCR text ถ้าจับคู่ไม่ได้

        words.append((x0, y0, x1, y1, text, 0, 0, 0))

    total = len(words)
    pct = (matched_typhoon / total * 100) if total > 0 else 0
    logger.info(
        f"  → [HYBRID] page {page_num}: "
        f"{matched_typhoon}/{total} words matched Typhoon ({pct:.0f}%)"
    )
    return words


# ══════════════════════════════════════════════════════════
#  Public API: get_words_all / process_page_ocr / extract_text_from_pdf
# ══════════════════════════════════════════════════════════

def get_words_from_page_ocr(page):
    """ดึงคำและพิกัดด้วย EasyOCR ล้วน (fallback)"""
    bboxes = _easyocr_bboxes(page)
    return [(x0, y0, x1, y1, text, 0, 0, 0) for (x0, y0, x1, y1, text) in bboxes]


def get_words_all(page, page_num=0, total_pages=1):
    """
    ดึงคำทั้งหมดจากหน้า PDF

    ลำดับ:
      1. Typhoon OCR (อ่านข้อความ) + EasyOCR (ตำแหน่ง) → hybrid words
      2. EasyOCR ล้วน (fallback ถ้า Ollama ไม่พร้อม)
      3. Native text จาก PyMuPDF (ถ้า PDF มี text layer)
    """
    if page is None:
        return []

    # ── 1. ลอง Hybrid: Typhoon text + EasyOCR bbox ──
    if _check_ollama():
        try:
            hybrid = _build_hybrid_words(page, page_num, total_pages)
            if hybrid:
                return hybrid
        except Exception as e:
            logger.warning(f"Hybrid OCR failed page {page_num}: {e}")

    # ── 2. Fallback: EasyOCR ล้วน ──
    if FORCE_OCR:
        try:
            logger.info(f"  → [EASYOCR fallback] page {page_num}/{total_pages}")
            return get_words_from_page_ocr(page)
        except Exception as e:
            print(f"  → OCR failed: {e}")
            return []

    # ── 3. Native text จาก PyMuPDF ──
    words = page.get_text("words")
    if len(words) < 5:
        try:
            words = get_words_from_page_ocr(page)
        except Exception as e:
            print(f"  → OCR failed: {e}")
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
    Process single page สำหรับ parallel OCR
    หมายเหตุ: Typhoon OCR ทำ parallel ไม่ดี (Ollama ทำทีละ request)
              ดังนั้นถ้ามี Typhoon → ไปใช้ sequential path แทน (build_word_index)
              ฟังก์ชันนี้จะใช้ EasyOCR ล้วนสำหรับ parallel
    """
    page_num, pdf_path, page_rect_width, page_rect_height = args
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        words = get_words_from_page_ocr(page)
        doc.close()
        return page_num, words
    except Exception as e:
        logger.error(f"OCR failed for page {page_num}: {e}")
        return page_num, []


def extract_text_from_pdf(pdf_path):
    """
    แกะข้อความจาก PDF สำหรับส่งให้ LLM วิเคราะห์

    Returns:
        tuple: (ข้อความ, engine) — engine เป็น "typhoon" หรือ "easyocr"
    """
    if not os.path.exists(pdf_path):
        logger.warning(f"PDF file not found: {pdf_path}")
        return "", ""

    # ── 1. Typhoon OCR (Ollama local) ──
    if _check_ollama():
        try:
            client = _get_ollama_client()
            logger.info(f"📄 [TYPHOON] → {os.path.basename(pdf_path)}")
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            extracted_text = ""

            for i, page in enumerate(doc):
                page_text = _typhoon_ocr_page(page, i + 1, total_pages)
                if page_text:
                    extracted_text += f"--- Page {i+1} (Typhoon OCR 1.5 Local) ---\n{page_text}\n"
                else:
                    extracted_text += f"--- Page {i+1} (Typhoon OCR Error) ---\n\n"

            doc.close()
            if extracted_text.strip():
                return extracted_text, "typhoon"
        except Exception as e:
            logger.warning(f"Typhoon extract failed: {e}")

    # ── 2. Fallback: EasyOCR ──
    logger.info(f"📄 [EASYOCR fallback] → {os.path.basename(pdf_path)}")
    doc = fitz.open(pdf_path)
    extracted_text = ""
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            extracted_text += f"--- Page {i+1} (Native) ---\n{text}\n"
            continue
        try:
            pix = page.get_pixmap(dpi=OCR_DPI)
            img_np = np.array(Image.open(io.BytesIO(pix.tobytes("png"))))
            result_list = get_ocr_reader().readtext(img_np, detail=0, paragraph=True)
            extracted_text += f"--- Page {i+1} (EasyOCR) ---\n{chr(10).join(result_list)}\n"
        except Exception as e:
            logger.warning(f"EasyOCR failed page {i+1}: {e}")
    doc.close()
    return extracted_text, "easyocr"
