"""Word significance และ form label analysis"""

from .config import FORM_LABELS_EXACT, FORM_LABELS_PARTIAL, STOPWORDS, MIN_WORD_LENGTH
from .text_utils import clean_text


def is_form_label(text):
    """ตรวจสอบว่าคำนี้เป็น form label/header หรือไม่"""
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
    """ตรวจสอบว่าคำนี้สำคัญพอที่จะ index และเปรียบเทียบหรือไม่"""
    clean = clean_text(text)
    if not clean:
        return False
    if len(clean) < MIN_WORD_LENGTH:
        return False
    if clean in STOPWORDS:
        return False
    return True


def get_full_text_from_index(index):
    """สร้าง full text จาก word index"""
    texts = []
    by_page = index.get('by_page', {})
    for page_num in sorted(by_page.keys()):
        words = by_page[page_num]
        page_text = ' '.join(w[4] for w in words)
        texts.append(page_text)
    return ' '.join(texts)


def get_first_original_text(index: dict, clean_word: str) -> str:
    """ดึง original text ตัวแรกจาก word index"""
    locs = index.get('by_word', {}).get(clean_word, [])
    if not locs:
        return ""
    return locs[0][1][4]
