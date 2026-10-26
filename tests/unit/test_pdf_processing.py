"""PDF reading, page detection and page-by-page processing (fake OCR)."""

from __future__ import annotations

import fitz
import pytest

from docintel.core.deadline import Deadline
from docintel.core.exceptions import OCRProcessingError, PageRangeError
from docintel.core.result_models import PageSource
from docintel.documents.document_processor import PDFMode, parse_page_range, process_pdf
from docintel.documents.pdf_reader import analyze_page, open_pdf, read_metadata
from docintel.documents.pdf_renderer import effective_dpi, render_page
from docintel.evaluation import synthetic_data as synth
from docintel.ocr.base_engine import OCROptions
from tests.fakes import FakeEngine


def _ocr_with(engine):
    def ocr_page(image, number):
        return engine.recognize(image, OCROptions()), []

    return ocr_page


@pytest.mark.parametrize(
    "spec, expected",
    [("", [1, 2, 3, 4, 5]), ("all", [1, 2, 3, 4, 5]), ("2", [2]), ("1-3", [1, 2, 3]), ("4-", [4, 5]), ("-2", [1, 2]), ("5, 1-2, 2", [1, 2, 5])],
)
def test_parse_page_range(spec, expected):
    assert parse_page_range(spec, 5) == expected


@pytest.mark.parametrize("spec", ["0", "6", "3-1", "a-b", "1;2"])
def test_parse_page_range_invalid(spec):
    with pytest.raises(PageRangeError):
        parse_page_range(spec, 5)


def test_metadata_and_embedded_text():
    data = synth.make_digital_pdf(["The first page has plenty of embedded text to read."])
    with open_pdf(data) as doc:
        assert read_metadata(doc)["page_count"] == 1
        info = analyze_page(doc[0])
    assert not info.needs_ocr
    assert "embedded text" in info.text


def test_scanned_page_detected_as_needing_ocr(text_image):
    data = synth.make_scanned_pdf([text_image])
    with open_pdf(data) as doc:
        info = analyze_page(doc[0])
    assert info.needs_ocr


def test_digital_pdf_uses_embedded_text_without_ocr():
    engine = FakeEngine()
    data = synth.make_digital_pdf(["Alpha page with enough embedded characters.", "Beta page with enough embedded characters."])
    pages = process_pdf(data, page_numbers=[1, 2], mode=PDFMode.AUTO, ocr_page=_ocr_with(engine), render_dpi=100, max_pixels=10**7, deadline=Deadline(60))
    assert [p.source for p in pages] == [PageSource.EMBEDDED_TEXT] * 2
    assert "Alpha" in pages[0].raw_text and "Beta" in pages[1].raw_text
    assert engine.calls == []


def test_scanned_pdf_goes_through_ocr(text_image):
    engine = FakeEngine(text="Scanned words")
    data = synth.make_scanned_pdf([text_image, text_image])
    pages = process_pdf(data, page_numbers=[1, 2], mode=PDFMode.AUTO, ocr_page=_ocr_with(engine), render_dpi=100, max_pixels=10**7, deadline=Deadline(60))
    assert all(p.source is PageSource.OCR and p.raw_text == "Scanned words" for p in pages)
    assert len(engine.calls) == 2


def test_mixed_pdf_routes_each_page(text_image):
    engine = FakeEngine(text="from ocr")
    data = synth.make_mixed_pdf("Digital page with a sufficient amount of embedded text.", text_image)
    pages = process_pdf(data, page_numbers=[1, 2], mode=PDFMode.AUTO, ocr_page=_ocr_with(engine), render_dpi=100, max_pixels=10**7, deadline=Deadline(60))
    assert pages[0].source is PageSource.EMBEDDED_TEXT
    assert pages[1].source is PageSource.OCR and pages[1].raw_text == "from ocr"


def test_page_range_and_progress(text_image):
    data = synth.make_digital_pdf([f"Page {i} text with enough embedded characters." for i in range(1, 6)])
    events = []
    pages = process_pdf(data, page_numbers=[2, 4], mode=PDFMode.AUTO, ocr_page=None, render_dpi=100, max_pixels=10**7,
                        deadline=Deadline(60), progress=lambda d, t, p, s: events.append((d, t, p, s)))
    assert [p.page_number for p in pages] == [2, 4]
    assert "Page 2" in pages[0].raw_text and "Page 4" in pages[1].raw_text
    assert events[-1][3] == "done" and events[0][:3] == (0, 2, 2)


def test_embedded_only_mode_never_ocrs(text_image):
    engine = FakeEngine()
    data = synth.make_scanned_pdf([text_image])
    pages = process_pdf(data, page_numbers=[1], mode=PDFMode.EMBEDDED_ONLY, ocr_page=_ocr_with(engine), render_dpi=100, max_pixels=10**7, deadline=Deadline(60))
    assert engine.calls == [] and pages[0].source is PageSource.NONE
    assert "OCR disabled" in pages[0].detection_reason


def test_ocr_all_mode_ignores_embedded_text():
    engine = FakeEngine(text="ocr text")
    data = synth.make_digital_pdf(["Digital text that would normally be used directly."])
    pages = process_pdf(data, page_numbers=[1], mode=PDFMode.OCR_ALL, ocr_page=_ocr_with(engine), render_dpi=100, max_pixels=10**7, deadline=Deadline(60))
    assert pages[0].source is PageSource.OCR and pages[0].raw_text == "ocr text"


def test_failure_on_one_page_does_not_stop_others(text_image):
    calls = {"n": 0}

    def flaky(image, number):
        calls["n"] += 1
        if number == 1:
            raise OCRProcessingError("engine crashed")
        return FakeEngine(text="ok").recognize(image, OCROptions()), []

    data = synth.make_scanned_pdf([text_image, text_image])
    pages = process_pdf(data, page_numbers=[1, 2], mode=PDFMode.AUTO, ocr_page=flaky, render_dpi=100, max_pixels=10**7, deadline=Deadline(60))
    assert pages[0].error is not None and pages[0].error.code == "ocr_failed"
    assert pages[1].error is None and pages[1].raw_text == "ok"


def test_render_respects_pixel_budget():
    doc = fitz.open()
    page = doc.new_page(width=2000, height=2000)  # very large page
    assert effective_dpi(page, 300, 4_000_000) < 300
    image = render_page(page, 300, 4_000_000)
    assert image.width * image.height <= 4_000_000 * 1.05
