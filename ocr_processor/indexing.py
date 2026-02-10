"""Word indexing สำหรับ comparison"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import fitz

from .config import logger, USE_PROCESS_POOL, MAX_OCR_WORKERS
from .ocr_engine import process_page_ocr, get_words_all
from .text_utils import clean_text
from .word_analysis import is_significant


def build_word_index_parallel(pdf_path, progress_callback=None):
    """สร้าง index ด้วย parallel OCR"""
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    index = {'by_page': {}, 'by_word': defaultdict(list)}
    if progress_callback:
        progress_callback(f"  → Starting parallel OCR ({MAX_OCR_WORKERS} workers)...", 0)
    page_args = []
    for page_num in range(total_pages):
        page = doc[page_num]
        page_args.append((page_num, pdf_path, page.rect.width, page.rect.height))
    doc.close()

    total_words = 0
    completed = 0
    executor_class = ProcessPoolExecutor if USE_PROCESS_POOL else ThreadPoolExecutor
    try:
        with executor_class(max_workers=MAX_OCR_WORKERS) as executor:
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
    except Exception as e:
        logger.error(f"Parallel OCR failed: {e}")
        total_words = 0
        index['by_page'] = {}
        index['by_word'] = defaultdict(list)

    if total_words == 0 and total_pages > 0:
        logger.warning("Parallel OCR returned 0 words, falling back to sequential...")
        if progress_callback:
            progress_callback("  → Retrying with sequential OCR...", 0)
        doc = fitz.open(pdf_path)
        index = build_word_index(doc, progress_callback)
        doc.close()
        return index

    print(f"  → Total words: {total_words}, Unique significant words: {len(index['by_word'])}")
    return index


def build_word_index(doc, progress_callback=None):
    """สร้าง index ของคำทั้งเอกสาร (sequential)"""
    index = {'by_page': {}, 'by_word': defaultdict(list)}
    total_words = 0
    total_pages = len(doc)
    for page_num in range(total_pages):
        page = doc[page_num]
        words = get_words_all(page, page_num=page_num + 1, total_pages=total_pages)
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
