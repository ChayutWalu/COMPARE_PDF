"""Highlighting modes สำหรับ SAME และ DIFF"""

import re
from typing import Callable

import fitz

from .config import DEBUG_MODE, FUZZY_THRESHOLD_STRICT, logger
from .text_utils import normalize_for_compare
from .number_utils import smart_normalize_number, extract_number_value, is_thai_number_word
from .date_utils import extract_thai_dates, extract_times, is_date_related_word
from .word_analysis import get_first_original_text, get_full_text_from_index, is_form_label
from .fuzzy_matching import fuzzy_match_word, fuzzy_match_word_index


def _collect_words_only_in_doc(
    words_this: set, words_other: set, index_this: dict,
    normalized_other: dict, number_values_other: dict,
    dates_are_same: bool, times_are_same: bool, all_dates_this: list,
    fuzzy_has_match: Callable[[str], bool],
    skip_day_numbers_without_date_check: bool = False,
) -> set:
    """Helper: รวบรวมคำที่มีเฉพาะใน doc นี้"""
    words_only = set()
    for clean_word in words_this:
        if clean_word in words_other:
            continue
        normalized_clean = normalize_for_compare(clean_word)
        if normalized_clean in normalized_other:
            continue
        original_text = get_first_original_text(index_this, clean_word)
        if is_form_label(original_text) or is_form_label(clean_word):
            continue
        if dates_are_same and is_date_related_word(original_text):
            continue
        if times_are_same and extract_times(original_text):
            continue
        if any(c.isdigit() for c in clean_word) or is_thai_number_word(original_text):
            if dates_are_same:
                if re.match(r'^25\d{2}$', clean_word):
                    continue
                if re.match(r'^0?[1-9]$|^[12]\d$|^3[01]$', clean_word):
                    if skip_day_numbers_without_date_check or any(clean_word in d for d in all_dates_this):
                        continue
            normalized_num = smart_normalize_number(clean_word)
            if normalized_num not in number_values_other:
                words_only.add(clean_word)
        else:
            if not fuzzy_has_match(clean_word):
                words_only.add(clean_word)
    return words_only


def highlight_same_mode_db(doc1, index1, db_index):
    """Mode SAME: Highlight matching words (file vs database)"""
    GREEN = (0, 0.8, 0)
    matched_in_doc1 = set()
    matched_words = []
    db_words = db_index.get('by_word', {})
    for clean_word, locations1 in index1['by_word'].items():
        if clean_word in db_words:
            matched_words.append(clean_word)
            for page_num, word in locations1:
                word_id = (page_num, word[0], word[1], word[4])
                if word_id not in matched_in_doc1:
                    matched_in_doc1.add(word_id)
                    page = doc1[page_num]
                    rect = fitz.Rect(word[:4])
                    page.draw_rect(rect, color=GREEN, fill=GREEN, fill_opacity=0.35, width=0)
        else:
            matches_in_db = fuzzy_match_word_index(clean_word, db_words, threshold=0.60)
            if matches_in_db:
                matched_words.append(clean_word)
                for page_num, word in locations1:
                    word_id = (page_num, word[0], word[1], word[4])
                    if word_id not in matched_in_doc1:
                        matched_in_doc1.add(word_id)
                        page = doc1[page_num]
                        rect = fitz.Rect(word[:4])
                        page.draw_rect(rect, color=GREEN, fill=GREEN, fill_opacity=0.35, width=0)
    print(f"  → Matched unique words: {len(matched_words)}")
    print(f"  → Highlighted {len(matched_in_doc1)} words in input file")
    return {
        'matched_doc1': len(matched_in_doc1),
        'matched_doc2': 0,
        'matched_unique_words': len(matched_words),
        'sample_matched': matched_words[:10]
    }


def highlight_diff_mode_db(doc1, index1, db_index):
    """Mode DIFF: Highlight different words (file vs database)"""
    RED = (1, 0.3, 0.3)
    words1 = set(index1['by_word'].keys())
    db_words = db_index.get('by_word', {})
    words2 = set(db_words.keys())
    normalized2 = {normalize_for_compare(w): w for w in words2}
    full_text1 = get_full_text_from_index(index1)
    full_text2 = get_full_text_from_index(db_index)
    all_dates1 = extract_thai_dates(full_text1)
    all_dates2 = extract_thai_dates(full_text2)
    all_times1 = extract_times(full_text1)
    all_times2 = extract_times(full_text2)
    dates_are_same = sorted(all_dates1) == sorted(all_dates2) if all_dates1 and all_dates2 else True
    times_are_same = sorted(all_times1) == sorted(all_times2) if all_times1 and all_times2 else True
    if DEBUG_MODE:
        logger.debug(f"Date Check: Dates match={dates_are_same}, Times match={times_are_same}")
    number_values2 = {}
    for w2 in words2:
        val = extract_number_value(w2)
        if val:
            number_values2[smart_normalize_number(w2)] = w2

    def _fuzzy_has_match_db(w: str) -> bool:
        return bool(fuzzy_match_word_index(w, db_words, threshold=FUZZY_THRESHOLD_STRICT))

    words_only_in_doc1 = _collect_words_only_in_doc(
        words1, words2, index1, normalized2, number_values2,
        dates_are_same, times_are_same, all_dates1, _fuzzy_has_match_db,
        skip_day_numbers_without_date_check=True,
    )
    print(f"  → Words only in input file: {len(words_only_in_doc1)}")
    count1 = 0
    for clean_word in words_only_in_doc1:
        for page_num, word in index1['by_word'][clean_word]:
            page = doc1[page_num]
            rect = fitz.Rect(word[:4])
            page.draw_rect(rect, color=RED, fill=RED, fill_opacity=0.4, width=0)
            count1 += 1
    print(f"  → Highlighted {count1} words in input file (red)")
    return {
        'diff_doc1': len(words_only_in_doc1),
        'diff_doc2': 0,
        'highlighted_doc1': count1,
        'highlighted_doc2': 0,
        'sample_doc1': list(words_only_in_doc1)[:10],
        'sample_doc2': []
    }


def highlight_same_mode(doc1, doc2, index1, index2):
    """โหมด SAME: ไฮไลท์คำที่เหมือนกัน"""
    GREEN = (0, 0.8, 0)
    matched_in_doc1 = set()
    matched_in_doc2 = set()
    matched_words = []
    words2 = set(index2['by_word'].keys())
    normalized2 = {normalize_for_compare(w): w for w in words2}
    if DEBUG_MODE:
        sample_words1 = list(index1['by_word'].keys())[:10]
        sample_words2 = list(index2['by_word'].keys())[:10]
        logger.debug(f"Sample words in Doc1: {sample_words1}")
        logger.debug(f"Sample words in Doc2: {sample_words2}")
    for clean_word, locations1 in index1['by_word'].items():
        matched_doc2_word = None
        if clean_word in index2['by_word']:
            matched_doc2_word = clean_word
        else:
            normalized_clean = normalize_for_compare(clean_word)
            if normalized_clean in normalized2:
                matched_doc2_word = normalized2[normalized_clean]
        if matched_doc2_word:
            matched_words.append(clean_word)
            for page_num, word in locations1:
                word_id = (page_num, word[0], word[1], word[4])
                if word_id not in matched_in_doc1:
                    matched_in_doc1.add(word_id)
                    page = doc1[page_num]
                    rect = fitz.Rect(word[:4])
                    page.draw_rect(rect, color=GREEN, fill=GREEN, fill_opacity=0.35, width=0)
            for page_num, word in index2['by_word'][matched_doc2_word]:
                word_id = (page_num, word[0], word[1], word[4])
                if word_id not in matched_in_doc2:
                    matched_in_doc2.add(word_id)
                    page = doc2[page_num]
                    rect = fitz.Rect(word[:4])
                    page.draw_rect(rect, color=GREEN, fill=GREEN, fill_opacity=0.35, width=0)
        else:
            matches_in_doc2 = fuzzy_match_word(clean_word, index2, threshold=0.60)
            if matches_in_doc2:
                matched_words.append(clean_word)
                for page_num, word in locations1:
                    word_id = (page_num, word[0], word[1], word[4])
                    if word_id not in matched_in_doc1:
                        matched_in_doc1.add(word_id)
                        page = doc1[page_num]
                        rect = fitz.Rect(word[:4])
                        page.draw_rect(rect, color=GREEN, fill=GREEN, fill_opacity=0.35, width=0)
                for page_num, word, score in matches_in_doc2:
                    word_id = (page_num, word[0], word[1], word[4])
                    if word_id not in matched_in_doc2:
                        matched_in_doc2.add(word_id)
                        page = doc2[page_num]
                        rect = fitz.Rect(word[:4])
                        page.draw_rect(rect, color=GREEN, fill=GREEN, fill_opacity=0.35, width=0)
    print(f"  → Matched unique words: {len(matched_words)}")
    print(f"  → Sample matched: {matched_words[:10]}")
    print(f"  → Highlighted {len(matched_in_doc1)} words in Doc1, {len(matched_in_doc2)} words in Doc2")
    return {
        'matched_doc1': len(matched_in_doc1),
        'matched_doc2': len(matched_in_doc2),
        'matched_unique_words': len(matched_words),
        'sample_matched': matched_words[:10]
    }


def highlight_diff_mode(doc1, doc2, index1, index2):
    """โหมด DIFF: ไฮไลท์คำที่แตกต่างกัน"""
    RED = (1, 0.3, 0.3)
    GREEN = (0.3, 0.8, 0.3)
    words1 = set(index1['by_word'].keys())
    words2 = set(index2['by_word'].keys())
    normalized2 = {normalize_for_compare(w): w for w in words2}
    normalized1 = {normalize_for_compare(w): w for w in words1}
    number_values1 = {}
    number_values2 = {}
    for w1 in words1:
        val = extract_number_value(w1)
        if val:
            number_values1[smart_normalize_number(w1)] = w1
    for w2 in words2:
        val = extract_number_value(w2)
        if val:
            number_values2[smart_normalize_number(w2)] = w2
    full_text1 = get_full_text_from_index(index1)
    full_text2 = get_full_text_from_index(index2)
    all_dates1 = extract_thai_dates(full_text1)
    all_dates2 = extract_thai_dates(full_text2)
    all_times1 = extract_times(full_text1)
    all_times2 = extract_times(full_text2)
    dates_are_same = sorted(all_dates1) == sorted(all_dates2) if all_dates1 and all_dates2 else True
    times_are_same = sorted(all_times1) == sorted(all_times2) if all_times1 and all_times2 else True
    if DEBUG_MODE:
        logger.debug(f"Date Check: Doc1 dates={all_dates1[:10]}, Doc2 dates={all_dates2[:10]}")
        logger.debug(f"Dates match={dates_are_same}, Times match={times_are_same}")
        logger.debug(f"Normalize Check Doc1: {list(normalized1.items())[:5]}, Doc2: {list(normalized2.items())[:5]}")

    def _fuzzy_has_match_doc1(w: str) -> bool:
        return bool(fuzzy_match_word(w, index2, threshold=FUZZY_THRESHOLD_STRICT))

    def _fuzzy_has_match_doc2(w: str) -> bool:
        return bool(fuzzy_match_word(w, index1, threshold=FUZZY_THRESHOLD_STRICT))

    words_only_in_doc1 = _collect_words_only_in_doc(
        words1, words2, index1, normalized2, number_values2,
        dates_are_same, times_are_same, all_dates1, _fuzzy_has_match_doc1,
        skip_day_numbers_without_date_check=False,
    )
    words_only_in_doc2 = _collect_words_only_in_doc(
        words2, words1, index2, normalized1, number_values1,
        dates_are_same, times_are_same, all_dates2, _fuzzy_has_match_doc2,
        skip_day_numbers_without_date_check=False,
    )
    print(f"  → Words only in Doc1: {len(words_only_in_doc1)}")
    print(f"  → Words only in Doc2: {len(words_only_in_doc2)}")
    count1 = 0
    for clean_word in words_only_in_doc1:
        for page_num, word in index1['by_word'][clean_word]:
            page = doc1[page_num]
            rect = fitz.Rect(word[:4])
            page.draw_rect(rect, color=RED, fill=RED, fill_opacity=0.4, width=0)
            count1 += 1
    count2 = 0
    for clean_word in words_only_in_doc2:
        for page_num, word in index2['by_word'][clean_word]:
            page = doc2[page_num]
            rect = fitz.Rect(word[:4])
            page.draw_rect(rect, color=GREEN, fill=GREEN, fill_opacity=0.4, width=0)
            count2 += 1
    print(f"  → Highlighted {count1} in Doc1 (red), {count2} in Doc2 (green)")
    return {
        'diff_doc1': len(words_only_in_doc1),
        'diff_doc2': len(words_only_in_doc2),
        'highlighted_doc1': count1,
        'highlighted_doc2': count2,
        'sample_doc1': list(words_only_in_doc1)[:10],
        'sample_doc2': list(words_only_in_doc2)[:10]
    }
