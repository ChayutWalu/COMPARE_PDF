"""Text cleaning และ normalization"""

import re


def clean_text(text: str) -> str:
    """ลบอักขระพิเศษ แต่เก็บตัวเลขและภาษาไทย"""
    text = re.sub(r'[^\w\u0E00-\u0E7F]', '', text)
    return text.lower().strip()


def normalize_thai_tones(text: str) -> str:
    """ลบวรรณยุกต์ไทยเพื่อเปรียบเทียบ"""
    thai_tones = '\u0E48\u0E49\u0E4A\u0E4B'
    for tone in thai_tones:
        text = text.replace(tone, '')
    return text


def normalize_leading_zeros(text):
    """Normalize leading zeros สำหรับวันที่ไทย"""
    normalized = re.sub(r'^(\d)([\u0E00-\u0E7F])', r'0\1\2', text)
    normalized = re.sub(r'([\u0E00-\u0E7F])(\d)$', r'\g<1>0\2', normalized)
    normalized = re.sub(r'([\u0E00-\u0E7F])(\d)([\u0E00-\u0E7F])', r'\g<1>0\2\3', normalized)
    return normalized


def normalize_for_compare(text):
    """Normalize text สำหรับเปรียบเทียบ"""
    normalized = re.sub(r'[\s/\-\.]+', '', clean_text(text))
    normalized = normalize_leading_zeros(normalized)
    return normalized


def normalize_date_text(text):
    """Normalize วันที่ format ต่างๆ ให้เหมือนกัน"""
    return re.sub(r'[\s/\-]+', '', text)


def normalize_number(text):
    """Normalize ตัวเลข - ลบ comma, space, จุด"""
    return re.sub(r'[,.\s]', '', text)
