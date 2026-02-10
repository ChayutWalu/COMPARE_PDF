"""Fuzzy matching functions for word comparison"""

import difflib

from .text_utils import clean_text, normalize_for_compare, normalize_thai_tones


def fuzzy_match_word(word, word_index, threshold=0.70):
    """หาคำที่คล้ายกันใน index (by_word dict inside full index)"""
    clean = clean_text(word)
    normalized = normalize_for_compare(word)
    normalized_no_tone = normalize_thai_tones(normalized)

    if not clean or len(clean) < 2:
        return []

    matches = []

    # Exact match
    if clean in word_index['by_word']:
        for page_num, w in word_index['by_word'][clean]:
            matches.append((page_num, w, 1.0))
        return matches

    for indexed_word, locations in word_index['by_word'].items():
        indexed_normalized = normalize_for_compare(indexed_word)
        indexed_no_tone = normalize_thai_tones(indexed_normalized)

        if normalized == indexed_normalized:
            for page_num, w in locations:
                matches.append((page_num, w, 0.95))
            continue

        if normalized_no_tone == indexed_no_tone:
            for page_num, w in locations:
                matches.append((page_num, w, 0.93))
            continue

        if len(normalized) >= 3 and len(indexed_normalized) >= 3:
            if normalized in indexed_normalized or indexed_normalized in normalized:
                for page_num, w in locations:
                    matches.append((page_num, w, 0.85))
                continue

        if len(normalized_no_tone) >= 3 and len(indexed_no_tone) >= 3:
            if normalized_no_tone in indexed_no_tone or indexed_no_tone in normalized_no_tone:
                for page_num, w in locations:
                    matches.append((page_num, w, 0.83))
                continue

        if abs(len(indexed_word) - len(clean)) > max(len(clean) * 0.5, 3):
            continue

        ratio = difflib.SequenceMatcher(None, clean, indexed_word).ratio()
        if ratio >= threshold:
            for page_num, w in locations:
                matches.append((page_num, w, ratio))

    return matches


def fuzzy_match_word_index(word, word_index_by_word, threshold=0.70):
    """Fuzzy match against a by_word dict (keys only, no locations)"""
    clean = clean_text(word)
    normalized = normalize_for_compare(word)
    normalized_no_tone = normalize_thai_tones(normalized)

    if not clean or len(clean) < 2:
        return []

    matches = []

    for indexed_word in word_index_by_word.keys():
        indexed_normalized = normalize_for_compare(indexed_word)
        indexed_no_tone = normalize_thai_tones(indexed_normalized)

        if normalized == indexed_normalized:
            matches.append(indexed_word)
            continue

        if normalized_no_tone == indexed_no_tone:
            matches.append(indexed_word)
            continue

        if len(normalized) >= 3 and len(indexed_normalized) >= 3:
            if normalized in indexed_normalized or indexed_normalized in normalized:
                matches.append(indexed_word)
                continue

        if abs(len(indexed_word) - len(clean)) > max(len(clean) * 0.5, 3):
            continue

        ratio = difflib.SequenceMatcher(None, clean, indexed_word).ratio()
        if ratio >= threshold:
            matches.append(indexed_word)

    return matches
