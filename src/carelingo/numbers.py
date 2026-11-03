"""Locale-aware extraction of numbers from dictation transcripts and notes."""

import re

# German and Polish write decimals with a comma and group thousands with a dot
# ("1.500 ml"); English is the reverse. Grouped forms are only accepted with
# exact three-digit groups so that "36.9" is never read as a thousands separator.
# Digits glued to letters ("SpO2") are not numbers.
_PATTERNS = {
    "comma_decimal": re.compile(r"(?<![\w.,])(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?(?!\d|,\d)"),
    "point_decimal": re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?!\d|\.\d)"),
}

_DECIMAL_STYLE = {"de": "comma_decimal", "pl": "comma_decimal", "en": "point_decimal"}

# Calendar dates ("12.06.2024"); the number patterns skip digits after a dot.
_DATE = re.compile(r"\b\d{1,2}\.\d{1,2}\.(?:\d{4}|\d{2})\b")

# "03:00 Uhr" should not produce a stray 0 that the dictation ("3 o'clock") lacks.
_FULL_HOUR = re.compile(r"\b(\d{1,2}):00\b")


def extract_numbers(text: str, language: str) -> list[float]:
    pattern = _PATTERNS[_DECIMAL_STYLE[language]]
    numbers = []
    for integer_part, fraction in pattern.findall(text):
        digits = re.sub(r"[.,]", "", integer_part)
        numbers.append(float(f"{digits}.{fraction}" if fraction else digits))
    return numbers


def contains_number(text: str, language: str, value: float) -> bool:
    return any(abs(found - value) < 1e-9 for found in extract_numbers(text, language))


def unsupported_numbers(source: str, source_language: str, german: str) -> tuple[float, ...]:
    """Numbers in a German note that never occur in the source dictation."""
    known = set(extract_numbers(source, source_language))
    found = extract_numbers(_FULL_HOUR.sub(r"\1", german), "de")
    return tuple(value for value in found if value not in known)


def unsupported_dates(source: str, german: str) -> tuple[str, ...]:
    """Calendar dates in a German note that the dictation does not contain."""
    return tuple(date for date in _DATE.findall(german) if date not in source)
