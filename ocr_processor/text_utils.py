"""Text utility functions for normalization and cleaning"""

import re


def clean_text(text):
    """ลบอักขระพิเศษ แต่เก็บตัวเลขและภาษาไทย"""
    text = re.sub(r'[^\w\u0E00-\u0E7F]', '', text)
    return text.lower().strip()


def normalize_thai_tones(text):
    """ลบวรรณยุกต์ไทยเพื่อเปรียบเทียบ"""
    thai_tones = '\u0E48\u0E49\u0E4A\u0E4B'
    for tone in thai_tones:
        text = text.replace(tone, '')
    return text


def normalize_leading_zeros(text):
    """Normalize leading zeros สำหรับวันที่ไทย"""
    return re.sub(r'^(\d)([\u0E00-\u0E7F])', r'0\1\2', text)


def normalize_for_compare(text):
    """Normalize text สำหรับเปรียบเทียบ"""
    normalized = re.sub(r'[\s/\-\.]+', '', clean_text(text))
    normalized = normalize_leading_zeros(normalized)
    return normalized
