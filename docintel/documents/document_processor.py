"""Page-by-page PDF processing: embedded text where usable, OCR where needed.

Pages are processed sequentially and rendered one at a time, so memory use is bounded by
a single rendered page regardless of document length. A failure on one page is recorded
on that page and processing continues with the next.
"""

from __future__ import annotations

import logging
import re
import time
from enum import Enum
from typing import Callable, Optional

from PIL import Image

from docintel.core.deadline import Deadline
from docintel.core.exceptions import DocIntelError, PageRangeError, ProcessingTimeoutError
from docintel.core.result_models import OCRResult, PageResult, PageSource, PreprocessStepRecord, StageError
from docintel.documents.pdf_reader import analyze_page, open_pdf
from docintel.documents.pdf_renderer import render_page

logger = logging.getLogger(__name__)

# (image, page_number) -> (ocr result, preprocessing records)
PageOCR = Callable[[Image.Image, int], tuple[OCRResult, list[PreprocessStepRecord]]]
# (pages_done, pages_total, page_number, status)
ProgressCallback = Callable[[int, int, int, str], None]


class PDFMode(str, Enum):
    AUTO = "auto"                    # embedded text when usable, OCR otherwise (mixed PDFs)
    EMBEDDED_ONLY = "embedded_only"  # never OCR; pages without text are reported empty
    OCR_ALL = "ocr_all"              # ignore embedded text and OCR every page


def parse_page_range(spec: str, page_count: int) -> list[int]:
    """Parse a 1-based page selection such as ``"1-3, 5, 8-"``. Empty or ``all`` selects all pages."""
    spec = (spec or "").strip().lower()
    if spec in ("", "all", "*"):
        return list(range(1, page_count + 1))
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)?\s*-\s*(\d+)?|(\d+)", part)
        if not match:
            raise PageRangeError(f"Invalid page selection '{part}'. Use forms like 1-3, 5, 8-.")
        if match.group(3):
            start = end = int(match.group(3))
        else:
            start = int(match.group(1) or 1)
            end = int(match.group(2) or page_count)
        if start < 1 or end > page_count or start > end:
            raise PageRangeError(f"Page selection '{part}' is outside 1-{page_count}.")
        pages.extend(range(start, end + 1))
    if not pages:
        raise PageRangeError("The page selection is empty.")
    return sorted(set(pages))


def process_pdf(
    data: bytes,
    *,
    page_numbers: list[int],
    mode: PDFMode,
    ocr_page: Optional[PageOCR],
    render_dpi: int,
    max_pixels: int,
    deadline: Deadline,
    progress: Optional[ProgressCallback] = None,
) -> list[PageResult]:
    """Process the selected pages and return one :class:`PageResult` per page."""
    results: list[PageResult] = []
    with open_pdf(data) as doc:
        total = len(page_numbers)
        for done, number in enumerate(page_numbers):
            if progress:
                progress(done, total, number, "processing")
            page_started = time.perf_counter()
            page = doc[number - 1]
            info = analyze_page(page)
            result = PageResult(
                page_number=number,
                embedded_text_chars=info.meaningful_chars,
                detection_reason=info.reason,
                width=int(page.rect.width),
                height=int(page.rect.height),
            )
            use_ocr = mode is PDFMode.OCR_ALL or (mode is PDFMode.AUTO and info.needs_ocr)
            try:
                deadline.check(f"page {number}")
                if not use_ocr:
                    result.source = PageSource.EMBEDDED_TEXT if info.text.strip() else PageSource.NONE
                    result.raw_text = info.text
                    if mode is PDFMode.EMBEDDED_ONLY and info.needs_ocr:
                        result.detection_reason += "; OCR disabled (embedded-only mode)"
                elif ocr_page is None:
                    raise DocIntelError("This page needs OCR but no OCR engine is configured.", code="ocr_not_configured")
                else:
                    image = render_page(page, render_dpi, max_pixels)
                    ocr_result, steps = ocr_page(image, number)
                    del image  # release the rendered page before the next one
                    result.source = PageSource.OCR
                    result.raw_text = ocr_result.text
                    result.ocr = ocr_result
                    result.preprocessing = steps
            except DocIntelError as exc:
                result.error = StageError(stage="page", code=exc.code, message=exc.user_message, page_number=number)
                logger.warning("Page failed", extra={"page": number, "code": exc.code})
                if isinstance(exc, ProcessingTimeoutError) and deadline.expired():
                    result.duration_seconds = round(time.perf_counter() - page_started, 3)
                    results.append(result)
                    _mark_remaining_skipped(results, page_numbers[done + 1 :], exc)
                    break
            result.duration_seconds = round(time.perf_counter() - page_started, 3)
            results.append(result)
        if progress:
            progress(total, total, page_numbers[-1] if page_numbers else 0, "done")
    return results


def _mark_remaining_skipped(results: list[PageResult], remaining: list[int], exc: DocIntelError) -> None:
    for number in remaining:
        results.append(
            PageResult(
                page_number=number,
                error=StageError(
                    stage="page", code=exc.code, message="Skipped: document time limit reached.", page_number=number
                ),
            )
        )
