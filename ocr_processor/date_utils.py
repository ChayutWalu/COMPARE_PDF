"""Date and time utility functions"""

import re

THAI_MONTHS = {
    'มกราคม': '01', 'กุมภาพันธ์': '02', 'มีนาคม': '03', 'เมษายน': '04',
    'พฤษภาคม': '05', 'มิถุนายน': '06', 'กรกฎาคม': '07', 'สิงหาคม': '08',
    'กันยายน': '09', 'ตุลาคม': '10', 'พฤศจิกายน': '11', 'ธันวาคม': '12',
    'ม.ค.': '01', 'ก.พ.': '02', 'มี.ค.': '03', 'เม.ย.': '04',
    'พ.ค.': '05', 'มิ.ย.': '06', 'ก.ค.': '07', 'ส.ค.': '08',
    'ก.ย.': '09', 'ต.ค.': '10', 'พ.ย.': '11', 'ธ.ค.': '12',
}

DATE_KEYWORDS = [
    'ระยะเวลาประกันภัย', 'วันที่', 'วันทำสัญญา', 'สิ้นสุดวันที่', 'เริ่มต้นวันที่',
    'ตั้งแต่วันที่', 'ถึงวันที่', 'วันทำสัญญาประกันภัย', 'วันที่ทำสัญญา',
    'period of insurance', 'from', 'agreement made on', 'policy issued',
    'เริ่มต้น', 'สิ้นสุด', 'เวลา',
]


def extract_thai_dates(text):
    """แยกวันที่ภาษาไทยจาก text"""
    dates = []
    for month_th, month_num in THAI_MONTHS.items():
        pattern = rf'(\d{{1,2}})\s*[/\-]?\s*{re.escape(month_th)}\s*[/\-]?\s*(\d{{4}})'
        matches = re.findall(pattern, text)
        for day, year in matches:
            dates.append(f"{int(day):02d}-{month_num}-{year}")

    pattern = r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})'
    matches = re.findall(pattern, text)
    for day, month, year in matches:
        dates.append(f"{int(day):02d}-{int(month):02d}-{year}")

    return dates


def extract_times(text):
    """แยกเวลาจาก text"""
    times = []
    pattern = r'(\d{1,2})[.:](\d{2})\s*(?:น\.?|u\.?)?'
    matches = re.findall(pattern, text)
    for hour, minute in matches:
        h = int(hour)
        if 0 <= h <= 24:
            times.append(f"{h:02d}:{minute}")
    return times


def is_date_related_word(text):
    """ตรวจสอบว่าเป็นส่วนหนึ่งของวันที่"""
    if text in THAI_MONTHS:
        return True
    if re.match(r'^25\d{2}$', text):
        return True
    if re.match(r'^0?[1-9]$|^[12]\d$|^3[01]$', text):
        return True
    for month in THAI_MONTHS.keys():
        if month in text and re.search(r'\d', text):
            return True
    if re.match(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{4}$', text):
        return True
    return False


def dates_match(text1, text2):
    """เปรียบเทียบวันที่"""
    dates1 = extract_thai_dates(text1)
    dates2 = extract_thai_dates(text2)
    if not dates1 and not dates2:
        return None
    if dates1 and dates2:
        return sorted(dates1) == sorted(dates2)
    return False


def times_match(text1, text2):
    """เปรียบเทียบเวลา"""
    times1 = extract_times(text1)
    times2 = extract_times(text2)
    if not times1 and not times2:
        return None
    if times1 and times2:
        return sorted(times1) == sorted(times2)
    return False
