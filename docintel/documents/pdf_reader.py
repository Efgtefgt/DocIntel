"""PDF opening, metadata and embedded-text analysis using PyMuPDF."""

from __future__ import annotations

import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import fitz  # PyMuPDF

from docintel.core.exceptions import CorruptFileError, EncryptedDocumentError

MIN_TEXT_CHARS = 25
IMAGE_PAGE_COVERAGE = 0.8
IMAGE_PAGE_MAX_CHARS = 100


@contextmanager
def open_pdf(data: bytes) -> Iterator[fitz.Document]:
    """Open PDF bytes in memory, closing the document afterwards."""
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except (fitz.FileDataError, RuntimeError, ValueError) as exc:
        raise CorruptFileError("The PDF could not be opened; it may be corrupt.") from exc
    try:
        if doc.needs_pass:
            raise EncryptedDocumentError("The PDF is password-protected.")
        yield doc
    finally:
        doc.close()


def read_metadata(doc: fitz.Document) -> dict[str, str | int | bool]:
    meta = doc.metadata or {}
    return {
        "page_count": doc.page_count,
        "title": meta.get("title", "") or "",
        "author": meta.get("author", "") or "",
        "subject": meta.get("subject", "") or "",
        "creator": meta.get("creator", "") or "",
        "producer": meta.get("producer", "") or "",
        "creation_date": meta.get("creationDate", "") or "",
        "modification_date": meta.get("modDate", "") or "",
        "format": meta.get("format", "") or "",
        "encrypted": bool(doc.is_encrypted),
    }


@dataclass(frozen=True)
class PageTextInfo:
    page_number: int  # 1-based
    text: str
    meaningful_chars: int
    image_coverage: float
    needs_ocr: bool
    reason: str


def _image_coverage(page: fitz.Page) -> float:
    page_area = abs(page.rect)
    if page_area <= 0:
        return 0.0
    covered = 0.0
    for info in page.get_image_info():
        rect = fitz.Rect(info["bbox"]) & page.rect
        if not rect.is_empty:
            covered += abs(rect)
    return min(1.0, covered / page_area)


def analyze_page(page: fitz.Page, min_chars: int = MIN_TEXT_CHARS) -> PageTextInfo:
    """Extract embedded text and decide whether the page needs OCR.

    A page needs OCR when it has (almost) no embedded text, when the text looks garbled
    (many replacement or private-use characters, typical of broken font encodings), or
    when it is mostly covered by an image with only a few characters of text.
    """
    text = page.get_text("text", sort=True)
    meaningful = sum(1 for ch in text if ch.isalnum())
    garbled = sum(1 for ch in text if ch == "�" or unicodedata.category(ch) == "Co")
    coverage = _image_coverage(page)
    number = page.number + 1

    if meaningful < min_chars:
        return PageTextInfo(number, text, meaningful, coverage, True, f"only {meaningful} embedded characters")
    if garbled / max(1, len(text.strip())) > 0.1:
        return PageTextInfo(number, text, meaningful, coverage, True, "embedded text appears garbled")
    if coverage >= IMAGE_PAGE_COVERAGE and meaningful < IMAGE_PAGE_MAX_CHARS:
        return PageTextInfo(number, text, meaningful, coverage, True, "page is mostly an image with little text")
    return PageTextInfo(number, text, meaningful, coverage, False, f"{meaningful} embedded characters")
