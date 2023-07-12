from typing import List

import re
import string
import unidecode

def normalize_section_name(text: str, stop_words: List[str]) -> str:
    text = text.strip().lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[0-9]+', '', text)
    text = unidecode.unidecode(text)
    return ' '.join([token for token in text.split() if token not in stop_words])


def _get_words(text: str) -> str:
    text = re.sub('[%s]+' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'[0-9]+', ' ', text)
    return text.strip().split()


def perc_isupper(text: str) -> float:
    words = _get_words(text)
    if not words:
        return 0.0
    num_upper = sum(word.isupper() for word in words)
    return float(num_upper) / len(words)


def perc_istitle(text: str) -> float:
    words = _get_words(text)
    if not words:
        return 0.0
    num_title = sum(word.istitle() for word in words)
    return float(num_title) / len(words)


def perc_islower(text: str) -> float:
    words = _get_words(text)
    if not words:
        return 0.0
    num_lower = sum(word.islower() for word in words)
    return float(num_lower) / len(words)


def has_year(text: str) -> bool:
    matches = re.findall(r'\b(\d{4}|/\d{2})\b', text)
    return len(matches) > 0


def has_punct(text: str) -> bool:
    matches = re.findall('[' + re.escape(string.punctuation) + '—]', text)
    return len(matches) > 0


def has_punct_end(text: str) -> bool:
    matches = re.findall('[' + re.escape(string.punctuation) + r'—]\*$', text)
    return len(matches) > 0


def has_colon(text: str) -> bool:
    return ':' in text


def has_colon_end(text: str) -> bool:
    matches = re.findall(r':\s*$', text)
    return len(matches) > 0


def has_number(text: str) -> bool:
    return any(c.isdigit() for c in text)


def get_num_tokens(text: str, stop_words: List[str]) -> int:
    tokens = text.lower().strip().split()
    tokens = [token for token in tokens if token not in stop_words]
    return len(tokens)


def is_sparse_section_name(text: str, section_names: List[str]) -> bool:
    text = re.sub(r'\s', '', text)
    for section_name in section_names:
        section_name = re.sub(r'\s', '', section_name)
        if text == section_name:
            return True
    return False


def has_section_name(text: str, section_names: List[str]) -> bool:
    if not text:
        return 0
    elif is_sparse_section_name(text, section_names):
        return 1
    return any(section_name in text for section_name in section_names)


def perc_section_name(text: str, section_names: List[str]) -> float:
    if not text:
        return 0
    elif is_sparse_section_name(text, section_names):
        return 1
    max_tokens_used = 0
    for section_name in section_names:
        if section_name in text:
            num_tokens = len(section_name.split())
            max_tokens_used = max(max_tokens_used, num_tokens)
    return max_tokens_used / len(text.split())


def perc_section_name_all(text: str, section_names: List[str]) -> float:
    text_tokens = text.split()
    if not text_tokens:
        return 0
    elif is_sparse_section_name(text, section_names):
        return 1
    tokens_seen = [0] * len(text_tokens)
    for section_name in section_names:
        section_tokens = section_name.split()
        # Speed-up search
        is_candidate = all(token not in text_tokens for token in section_tokens)
        if not is_candidate:
            continue

        # Check whether section name is in text
        for i in range(len(text_tokens) - len(section_tokens) + 1):
            if section_tokens != text_tokens[i:i+len(section_tokens)]:
                continue
            # Mark tokens as seen
            for j in range(len(section_tokens)):
                tokens_seen[i+j] = 1
            break
    return sum(tokens_seen) / len(text_tokens)
