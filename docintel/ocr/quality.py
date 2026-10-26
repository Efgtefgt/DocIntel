"""Evaluate OCR output: how much of the visible text was recognised, and how confidently.

Confidence alone is misleading: an engine can report 95 while silently skipping half the
page (measured on the shadowed-photo benchmark sample). So the expected amount of text is
estimated independently, by counting glyph-sized connected components on a
shadow-corrected copy of the page, and compared with the recognised characters.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image

from docintel.core.result_models import OCRResult
from docintel.preprocessing import enhancement
from docintel.preprocessing.image_loader import pil_to_array

MAX_SIDE = 2000


def estimate_glyph_count(image: Image.Image) -> Optional[int]:
    """Approximate number of printed characters on the page (None if no text-like content)."""
    arr = pil_to_array(image)
    gray = enhancement.to_grayscale(arr)
    scale = min(1.0, MAX_SIDE / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if float(np.median(gray)) < 110:
        gray = 255 - gray
    flat, _ = enhancement.flatten_illumination(gray)
    _, mask = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    areas = stats[1:, cv2.CC_STAT_AREA]
    h_img, w_img = mask.shape
    plausible = (areas >= 6) & (heights < h_img * 0.2) & (widths < w_img * 0.3)
    if plausible.sum() < 5:
        return None
    median_h = float(np.median(heights[plausible]))
    glyphs = plausible & (heights >= 0.45 * median_h) & (heights <= 2.5 * median_h)
    return int(glyphs.sum())


def recognised_characters(ocr: OCRResult) -> int:
    return sum(ch.isalnum() for ch in ocr.text)


def coverage(ocr: OCRResult, expected: Optional[int]) -> Optional[float]:
    if not expected:
        return None
    return round(recognised_characters(ocr) / expected, 3)


def score(ocr: OCRResult, expected: Optional[int]) -> float:
    """Higher is better. Confidence (0-1) scaled by how much of the expected text was found."""
    if not ocr.words:
        return 0.0
    conf = (ocr.mean_confidence or 0.0) / 100.0
    cov = coverage(ocr, expected)
    return round(conf * (min(1.0, cov) if cov is not None else 1.0), 4)


def verdict(ocr: OCRResult, expected: Optional[int]) -> tuple[str, list[str]]:
    """'good', 'review' or 'poor', with reasons. Never a guarantee of correctness."""
    notes: list[str] = []
    if not ocr.words:
        return "poor", ["No text was recognised."]
    cov = coverage(ocr, expected)
    conf = ocr.mean_confidence or 0.0
    low_ratio = len(ocr.low_confidence_words) / max(1, len(ocr.words))
    if cov is not None and cov < 0.7:
        notes.append(f"Only about {cov:.0%} of the visible characters were recognised; text may be missing.")
    if conf < 75:
        notes.append(f"Low mean confidence ({conf:.0f}).")
    if low_ratio > 0.1:
        notes.append(f"{low_ratio:.0%} of words have low confidence.")
    if (cov is not None and cov < 0.5) or conf < 60:
        return "poor", notes
    if notes:
        return "review", notes
    return "good", ["Confidence and text coverage look normal."]
