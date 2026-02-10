"""UI helper functions — legend overlay and summary printing"""

import os
from PIL import ImageDraw, ImageFont

from .config import IS_MAC


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
        width=2,
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
    except Exception:
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
    print("\n" + "=" * 60)
    print("📊 COMPARISON SUMMARY")
    print("=" * 60)

    mode_text = "Finding MATCHES" if summary['mode'] == 'same' else "Finding DIFFERENCES"
    print(f"Mode: {mode_text}")
    print("-" * 60)

    print(f"📄 Document 1: {summary['doc1_name']}")
    print(f"   Pages: {summary['doc1_pages']}, Words: {summary['doc1_total_words']}, Unique: {summary['doc1_unique_words']}")

    print(f"📄 Document 2: {summary['doc2_name']}")
    print(f"   Pages: {summary['doc2_pages']}, Words: {summary['doc2_total_words']}, Unique: {summary['doc2_unique_words']}")

    print("-" * 60)

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

    print("=" * 60 + "\n")
