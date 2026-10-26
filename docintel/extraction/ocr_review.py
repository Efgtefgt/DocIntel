"""Flag suspicious OCR output for human review. Nothing here changes the recognised text.

Each flag names the token, the reasons, the OCR confidence (if known) and, where a common
OCR confusion explains it, a *suggestion*. Suggestions are only applied when a user accepts
them, and then go into the user's edits, never into the OCR output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from docintel.core.result_models import OCRResult

# Characters OCR commonly confuses between letters and digits.
TO_LETTER = {"0": "O", "1": "l", "5": "S", "8": "B", "6": "G", "2": "Z", "4": "A", "7": "T", "9": "g"}
TO_DIGIT = {"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "i": "1", "|": "1", "S": "5", "s": "5",
            "B": "8", "G": "6", "Z": "2", "z": "2", "g": "9", "q": "9", "T": "7"}
_STRIP = re.compile(r"^[\"'([{<]+|[\"')\]}>.,;:!?]+$")
_ALLOWED = re.compile(r"^[\w.,:;!?'\"()\[\]{}/\\@#%&*+=<>$€£₹¥~^|`-]+$", re.UNICODE)
# Tokens that legitimately mix letters and digits (codes, units, ordinals).
_CODE_LIKE = re.compile(r"^([A-Z]{1,6}-?\d+[A-Z0-9-]*|[A-Z0-9]+(-[A-Z0-9]+)+|\d+[A-Za-z]|[A-Za-z]\d)$")  # case-sensitive
_UNIT_LIKE = re.compile(r"^\d+(st|nd|rd|th|am|pm|kg|km|mm|cm|ml|mg|gb|mb|kb|px|x)$", re.IGNORECASE)


@dataclass
class SuspiciousToken:
    text: str
    line: int
    reasons: list[str] = field(default_factory=list)
    confidence: Optional[float] = None
    suggestion: Optional[str] = None


def _suggest(core: str) -> Optional[str]:
    letters = sum(c.isalpha() for c in core)
    digits = sum(c.isdigit() for c in core)
    if letters and digits:
        if letters >= digits * 2:  # mostly a word: turn stray digits into letters
            fixed = "".join(TO_LETTER.get(c, c) if c.isdigit() else c for c in core)
            # keep the case pattern sensible: "Inv0ice" -> "Invoice" (O lower-cased inside a word)
            fixed = "".join(c.lower() if c.isupper() and i > 0 and core[i].isdigit() and core[i - 1].islower() else c
                            for i, c in enumerate(fixed))
            return fixed if fixed != core else None
        if digits >= letters * 2:  # mostly a number: turn stray letters into digits
            fixed = "".join(TO_DIGIT.get(c, c) for c in core)
            return fixed if fixed != core and fixed.replace(".", "").replace(",", "").isdigit() else None
    return None


def review(ocr: OCRResult, *, max_items: int = 200) -> list[SuspiciousToken]:
    """Return tokens that look like OCR mistakes, in reading order."""
    flagged: list[SuspiciousToken] = []
    line_index = -1
    last_key = None
    for word in ocr.words:
        key = (word.block, word.paragraph, word.line)
        if key != last_key:
            line_index += 1
            last_key = key
        raw = word.text
        core = _STRIP.sub("", raw)
        reasons: list[str] = []
        suggestion = None
        if word.low_confidence:
            reasons.append("low OCR confidence")
        if (core and any(c.isalpha() for c in core) and any(c.isdigit() for c in core)
                and not _CODE_LIKE.match(core) and not _UNIT_LIKE.match(core)):
            reasons.append("letters and digits mixed in one word")
            suggestion = _suggest(core)
        if core and not _ALLOWED.match(core):
            reasons.append("unusual characters")
        if len(core) >= 3 and re.search(r"(.)\1\1", core) and not core.isdigit() and core.lower() not in {"www", "xxx"}:
            reasons.append("same character repeated three times")
        if raw and not any(c.isalnum() for c in raw) and len(raw) >= 3:
            reasons.append("symbols only")
        if reasons:
            if suggestion and raw != core:
                suggestion = raw.replace(core, suggestion)
            flagged.append(SuspiciousToken(text=raw, line=line_index, reasons=reasons, confidence=word.confidence,
                                           suggestion=suggestion))
            if len(flagged) >= max_items:
                break
    return flagged


def apply_suggestion(text: str, token: str, suggestion: str) -> str:
    """Replace whole-word occurrences of ``token`` with ``suggestion`` (used on the user's edits only)."""
    pattern = re.compile(rf"(?<![\w]){re.escape(token)}(?![\w])")
    return pattern.sub(suggestion, text)
