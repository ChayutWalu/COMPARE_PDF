"""Word analysis utilities — form label detection, significance checking"""

from .config import FORM_LABELS_EXACT, FORM_LABELS_PARTIAL, STOPWORDS
from .text_utils import clean_text


def is_form_label(text):
    """ตรวจสอบว่าเป็น form label"""
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
        'ความเสียหาย', 'อัตรา', 'เงื่อนไข', 'ข้อตกลง',
    ]
    for prefix in label_prefixes:
        if clean.startswith(prefix) or original_lower.startswith(prefix):
            return True

    return False


def is_significant(text):
    """ตรวจสอบคำสำคัญ (ไม่ใช่ stopword / สั้นเกินไป)"""
    clean = clean_text(text)
    if not clean:
        return False
    if len(clean) < 2:
        return False
    if clean in STOPWORDS:
        return False
    return True


def get_full_text_from_index(index):
    """รวมข้อความทั้งหมดจาก word index"""
    texts = []
    for _page_num, words in index.get('by_page', {}).items():
        page_text = ' '.join(w[4] for w in words)
        texts.append(page_text)
    return ' '.join(texts)


def get_first_original_text(index, clean_word):
    """ดึงข้อความต้นฉบับแรกสุดของคำที่ clean แล้ว"""
    by_word = index.get('by_word', {})
    if clean_word in by_word and by_word[clean_word]:
        return by_word[clean_word][0][1][4]
    return ""
