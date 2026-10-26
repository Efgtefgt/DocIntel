"""Type checks and normalization for extracted field values.

Validators never correct or complete a value (no guessed country codes, centuries or
digits). They either accept it (optionally with a normalized form), flag it as
uncertain (e.g. an ambiguous date), or reject it with a message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Annotated, Any, Optional

from pydantic import AfterValidator, BaseModel, ConfigDict, create_model

FIELD_TYPES = ("string", "text", "phone", "email", "date", "number", "amount")


@dataclass(frozen=True)
class ValueCheck:
    ok: bool
    normalized: Optional[str] = None
    message: str = ""
    uncertain: bool = False


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<![\w])\+?\d[\d \-().\/]{5,}\d(?![\w])")
DATE_RE = re.compile(
    r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
    r"|\d{1,2}[ \-](?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?[ \-,]+\d{4}"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.? \d{1,2},? \d{4})\b",
    re.IGNORECASE,
)

# Patterns used to look for typed values that have no label next to them.
UNLABELED_PATTERNS = {"phone": PHONE_RE, "email": EMAIL_RE, "date": DATE_RE}


def check_phone(raw: str) -> ValueCheck:
    value = raw.strip()
    if not re.fullmatch(r"\+?[\d \-().\/]+", value):
        return ValueCheck(False, message="contains characters not expected in a phone number")
    digits = re.sub(r"\D", "", value)
    if not 7 <= len(digits) <= 15:
        return ValueCheck(False, message=f"has {len(digits)} digits; expected 7-15")
    return ValueCheck(True, normalized=("+" if value.startswith("+") else "") + digits)


def check_email(raw: str) -> ValueCheck:
    value = raw.strip()
    if EMAIL_RE.fullmatch(value):
        return ValueCheck(True, normalized=value.lower())
    return ValueCheck(False, message="is not a valid e-mail address")


_UNAMBIGUOUS_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
    "%d %B %Y", "%d %b %Y", "%d-%b-%Y", "%d-%B-%Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
    "%d %b, %Y", "%d %B, %Y",
)


def check_date(raw: str) -> ValueCheck:
    value = re.sub(r"\s+", " ", raw.strip().replace("Sept", "Sep"))
    for fmt in _UNAMBIGUOUS_FORMATS:
        try:
            return ValueCheck(True, normalized=datetime.strptime(value, fmt).date().isoformat())
        except ValueError:
            continue
    match = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{2}|\d{4})", value)
    if not match:
        return ValueCheck(False, message="is not a recognised date")
    a, b, year_text = int(match.group(1)), int(match.group(2)), match.group(3)
    if len(year_text) == 2:
        return ValueCheck(True, normalized=None, uncertain=True, message="two-digit year; century not assumed")
    year = int(year_text)
    candidates: set[date] = set()
    for day, month in ((a, b), (b, a)):
        try:
            candidates.add(date(year, month, day))
        except ValueError:
            continue
    if not candidates:
        return ValueCheck(False, message="is not a valid calendar date")
    if len(candidates) > 1:
        return ValueCheck(True, normalized=None, uncertain=True, message="ambiguous day/month order")
    return ValueCheck(True, normalized=candidates.pop().isoformat())


def check_number(raw: str) -> ValueCheck:
    value = raw.strip().replace(",", "").replace(" ", "")
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", value):
        return ValueCheck(True, normalized=value.lstrip("+"))
    return ValueCheck(False, message="is not a number")


def check_amount(raw: str) -> ValueCheck:
    value = re.sub(r"^(?:[A-Z]{3}|[$€£₹¥]|Rs\.?|INR|USD|EUR)\s*", "", raw.strip(), flags=re.IGNORECASE)
    value = re.sub(r"\s*(?:[A-Z]{3}|/-)$", "", value)
    return check_number(value) if value else ValueCheck(False, message="is not an amount")


def check_text(raw: str) -> ValueCheck:
    return ValueCheck(True, normalized=raw.strip()) if raw.strip() else ValueCheck(False, message="is empty")


def check_string(raw: str) -> ValueCheck:
    """Short single-value fields (names, places, types).

    The normalized form drops trailing separators/full stops, which OCR engines (notably
    PaddleOCR) sometimes add at line ends. The verbatim OCR value is kept in ``value``.
    """
    stripped = raw.strip()
    if not stripped:
        return ValueCheck(False, message="is empty")
    return ValueCheck(True, normalized=stripped.rstrip(".,;:").rstrip() or stripped)


_CHECKS = {
    "string": check_string,
    "text": check_text,
    "phone": check_phone,
    "email": check_email,
    "date": check_date,
    "number": check_number,
    "amount": check_amount,
}


def check_value(field_type: str, raw: str) -> ValueCheck:
    if field_type not in _CHECKS:
        raise ValueError(f"Unknown field type '{field_type}'.")
    return _CHECKS[field_type](raw)


def _pydantic_validator(field_type: str):
    def validate(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        result = check_value(field_type, value)
        if not result.ok:
            raise ValueError(f"value {result.message}")
        return value

    return validate


def build_record_model(model_name: str, field_types: dict[str, str]) -> type[BaseModel]:
    """Create a Pydantic model whose fields are optional strings validated by type.

    ``None`` means "not found" and is always allowed; required-field checks happen in
    the extractor so that a missing value is reported, not rejected.
    """
    definitions: dict[str, Any] = {
        name: (Annotated[Optional[str], AfterValidator(_pydantic_validator(ftype))], None)
        for name, ftype in field_types.items()
    }
    safe_name = re.sub(r"\W", "_", model_name) or "Record"
    return create_model(safe_name, __config__=ConfigDict(extra="forbid"), **definitions)
