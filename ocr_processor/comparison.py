"""Main comparison functions"""

import io
import os
from typing import Optional, Callable

import fitz
from PIL import Image

from .config import PARALLEL_OCR, FORCE_OCR, OUTPUT_DPI
from .indexing import build_word_index, build_word_index_parallel
from .ocr_engine import _check_ollama
from .highlighting import highlight_same_mode, highlight_diff_mode, highlight_same_mode_db, highlight_diff_mode_db
from .ui_helpers import add_legend_to_image, print_summary


def highlight_text_differences(
    pdf1_path: str, pdf2_path: str, mode: str = 'diff',
    progress_callback: Optional[Callable[[str, Optional[float]], None]] = None
) -> tuple:
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

    doc1_pages = len(doc1)
    doc2_pages = len(doc2)
    # ถ้า Ollama พร้อม → ใช้ sequential (Typhoon ทำ parallel ไม่ได้)
    # ถ้า Ollama ไม่พร้อม → ใช้ parallel ได้ (EasyOCR ล้วน)
    use_parallel = PARALLEL_OCR and FORCE_OCR and not _check_ollama()

    update_progress(f"\n📄 Document 1: {os.path.basename(pdf1_path)} ({doc1_pages} pages)", 5)

    if use_parallel and doc1_pages > 1:
        doc1.close()
        index1 = build_word_index_parallel(pdf1_path, progress_callback=lambda msg, pct: update_progress(msg, 5 + pct * 0.25))
        doc1 = fitz.open(pdf1_path)
    else:
        index1 = build_word_index(doc1, progress_callback=lambda msg, pct: update_progress(msg, 5 + pct * 0.25))

    update_progress(f"\n📄 Document 2: {os.path.basename(pdf2_path)} ({doc2_pages} pages)", 35)

    if use_parallel and doc2_pages > 1:
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
            pix = doc1[i].get_pixmap(dpi=OUTPUT_DPI)
            img1 = Image.open(io.BytesIO(pix.tobytes("png")))
        if i < len(doc2):
            pix = doc2[i].get_pixmap(dpi=OUTPUT_DPI)
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


def highlight_text_differences_db(
    pdf_path: str, db_document: dict, mode: str = 'diff',
    progress_callback: Optional[Callable[[str, Optional[float]], None]] = None
) -> tuple:
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

    use_parallel = PARALLEL_OCR and FORCE_OCR and not _check_ollama()

    update_progress(f"\n📄 Input File: {os.path.basename(pdf_path)} ({doc1_pages} pages)", 5)
    if use_parallel and doc1_pages > 1:
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
        pix = doc1[i].get_pixmap(dpi=OUTPUT_DPI)
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
