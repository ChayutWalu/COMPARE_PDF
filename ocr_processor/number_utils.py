"""Number utility functions for comparison"""

import re


def smart_normalize_number(text):
    """Smart Number Normalization"""
    normalized = re.sub(r'[,\s]', '', text)
    try:
        if '.' in normalized:
            num = float(normalized)
            if num == int(num):
                return str(int(num))
            else:
                return f"{num:g}"
        else:
            num = int(normalized)
            return str(num)
    except Exception:
        return normalized.lstrip('0') or '0'


def extract_number_value(text):
    """แยกค่าตัวเลขจาก text"""
    clean = re.sub(r'[,\s]', '', text)
    match = re.match(r'^-?\d+\.?\d*$', clean)
    if match:
        try:
            if '.' in clean:
                return (float(clean), text)
            else:
                return (int(clean), text)
        except Exception:
            pass
    return None


def numbers_are_equal(num1_text, num2_text):
    """เปรียบเทียบตัวเลข 2 ตัว"""
    val1 = extract_number_value(num1_text)
    val2 = extract_number_value(num2_text)

    if val1 is None or val2 is None:
        return False

    v1, v2 = val1[0], val2[0]
    if isinstance(v1, float) or isinstance(v2, float):
        return abs(float(v1) - float(v2)) < 0.001
    else:
        return v1 == v2


def is_thai_number_word(text):
    """ตรวจสอบจำนวนเงินตัวหนังสือภาษาไทย"""
    thai_digits = [
        'หนึ่ง', 'สอง', 'สาม', 'สี่', 'ห้า', 'หก', 'เจ็ด', 'แปด', 'เก้า', 'สิบ',
        'เอ็ด', 'ยี่', 'ศูนย์',
    ]
    thai_units = ['ร้อย', 'พัน', 'หมื่น', 'แสน', 'ล้าน', 'สิบ']
    money_words = ['บาท', 'สตางค์', 'ถ้วน']

    has_digit = any(digit in text for digit in thai_digits)
    has_unit = any(unit in text for unit in thai_units)
    has_money = any(word in text for word in money_words)

    return (has_digit and has_unit) or (has_money and has_digit)
