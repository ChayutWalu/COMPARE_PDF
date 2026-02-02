"""Number extraction และ comparison"""

import re
from decimal import Decimal, InvalidOperation


def smart_normalize_number(text):
    """Smart Number Normalization - รองรับหลาย format"""
    normalized = re.sub(r'[,\s]', '', text)
    try:
        if '.' in normalized:
            num = float(normalized)
            if num == int(num):
                return str(int(num))
            return f"{num:g}"
        else:
            num = int(normalized)
            return str(num)
    except (ValueError, TypeError, InvalidOperation):
        return normalized.lstrip('0') or '0'


def extract_number_value(text):
    """แยกค่าตัวเลขจาก text Returns: (numeric_value, original_text) หรือ None"""
    clean = re.sub(r'[,\s]', '', text)
    match = re.match(r'^-?\d+\.?\d*$', clean)
    if match:
        try:
            if '.' in clean:
                return (float(clean), text)
            else:
                return (int(clean), text)
        except (ValueError, TypeError):
            pass
    return None


def numbers_are_equal(num1_text, num2_text):
    """เปรียบเทียบตัวเลข 2 ตัวว่าเท่ากันไหม"""
    val1 = extract_number_value(num1_text)
    val2 = extract_number_value(num2_text)
    if val1 is None or val2 is None:
        return False
    v1, v2 = val1[0], val2[0]
    try:
        d1 = Decimal(str(v1))
        d2 = Decimal(str(v2))
        if d1 == d1.to_integral_value() and d2 == d2.to_integral_value():
            return d1.to_integral_value() == d2.to_integral_value()
        if d2 != 0:
            relative_diff = abs((d1 - d2) / d2)
            return relative_diff < Decimal('0.000001')
        return d1 == 0
    except (InvalidOperation, ZeroDivisionError):
        if isinstance(v1, float) or isinstance(v2, float):
            return abs(float(v1) - float(v2)) < 0.001
        return v1 == v2


def is_thai_number_word(text):
    """ตรวจสอบว่าข้อความมีจำนวนเงินที่เป็นตัวหนังสือภาษาไทยหรือไม่"""
    thai_digits = ['หนึ่ง', 'สอง', 'สาม', 'สี่', 'ห้า', 'หก', 'เจ็ด', 'แปด', 'เก้า', 'สิบ',
                   'เอ็ด', 'ยี่', 'ศูนย์']
    thai_units = ['ร้อย', 'พัน', 'หมื่น', 'แสน', 'ล้าน', 'สิบ']
    money_words = ['บาท', 'สตางค์', 'ถ้วน']
    has_digit = any(digit in text for digit in thai_digits)
    has_unit = any(unit in text for unit in thai_units)
    has_money = any(word in text for word in money_words)
    return (has_digit and has_unit) or (has_money and has_digit)
