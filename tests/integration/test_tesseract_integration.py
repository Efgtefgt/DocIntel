"""Integration tests with the real Tesseract executable (skipped if it is not installed).

Assertions use CER thresholds or key words rather than exact text, because OCR output
legitimately varies between Tesseract versions and language models.
"""

from __future__ import annotations

import pytest

from docintel.core.pipeline import DocumentPipeline, PipelineOptions
from docintel.core.result_models import PageSource
from docintel.documents.searchable_pdf import check_searchable_pdf_support, create_searchable_pdf
from docintel.documents.pdf_reader import analyze_page, open_pdf
from docintel.evaluation import synthetic_data as synth
from docintel.evaluation.benchmark import run_benchmark
from docintel.evaluation.metrics import EvalNormalization, cer
from docintel.extraction.templates import TemplateStore
from docintel.ocr.base_engine import OCROptions
from docintel.ocr.tesseract_engine import TesseractEngine
from tests.conftest import TEMPLATES_DIR

pytestmark = pytest.mark.integration
LINES = ["The quick brown fox jumps over the lazy dog.", "Invoice number 48213 was paid in full."]
REFERENCE = "\n".join(LINES)
LOOSE = EvalNormalization(lowercase=True)


@pytest.fixture(scope="module")
def page_image():
    return synth.render_text_image(LINES, font_size=36)


def _run(settings, data: bytes, name: str, **kwargs):
    options = PipelineOptions(engine="tesseract", **kwargs)
    return DocumentPipeline(settings).process(data, name, options)


def test_basic_ocr(tesseract_settings, page_image):
    result = _run(tesseract_settings, synth.image_bytes(page_image), "clean.png", preprocess_preset="original")
    assert result.succeeded, result.errors
    page = result.document.pages[0]
    assert cer(REFERENCE, page.raw_text, LOOSE) < 0.05
    assert "\n" in page.raw_text  # line breaks preserved
    assert page.ocr.mean_confidence is not None and page.ocr.words


@pytest.mark.parametrize("fmt", ["JPEG", "BMP", "TIFF", "WEBP"])
def test_image_formats(tesseract_settings, page_image, fmt):
    result = _run(tesseract_settings, synth.image_bytes(page_image, fmt), f"img.{fmt.lower()}", preprocess_preset="grayscale")
    assert cer(REFERENCE, result.document.pages[0].raw_text, LOOSE) < 0.08


def test_rotated_image_with_deskew(tesseract_settings, page_image):
    rotated = synth.degrade(page_image, rotate=6)
    result = _run(tesseract_settings, synth.image_bytes(rotated), "rotated.png", preprocess_preset="deskewed")
    steps = {s.step: s.applied for s in result.document.pages[0].preprocessing}
    assert steps["deskew"]
    assert cer(REFERENCE, result.document.pages[0].raw_text, LOOSE) < 0.1


@pytest.mark.parametrize("ccw_rotation", [90, 180, 270])
def test_page_orientation_correction(tesseract_settings, ccw_rotation):
    lines = [f"Orientation detection needs enough text on the page, line {i}." for i in range(12)]
    turned = synth.render_text_image(lines, font_size=32).rotate(ccw_rotation, expand=True)  # PIL: counter-clockwise
    engine = TesseractEngine(tesseract_settings)
    if "osd" not in engine.available_languages():
        pytest.skip("osd.traineddata not installed")
    # Undoing a counter-clockwise turn of N degrees needs a clockwise turn of N degrees.
    assert engine.detect_orientation(turned) == ccw_rotation
    result = _run(tesseract_settings, synth.image_bytes(turned), "turned.png", custom_steps=("orientation", "grayscale"))
    assert cer("\n".join(lines), result.document.pages[0].raw_text, LOOSE) < 0.1


def test_low_resolution_image_upscaled(tesseract_settings, page_image):
    small = synth.degrade(page_image, scale=0.35)
    plain = _run(tesseract_settings, synth.image_bytes(small), "small.png", preprocess_preset="original")
    upscaled = _run(tesseract_settings, synth.image_bytes(small), "small.png", custom_steps=("grayscale", "upscale"))
    assert any(s.step == "upscale" and s.applied for s in upscaled.document.pages[0].preprocessing)
    assert cer(REFERENCE, upscaled.document.pages[0].raw_text, LOOSE) <= cer(REFERENCE, plain.document.pages[0].raw_text, LOOSE) + 0.02


def test_scanned_pdf(tesseract_settings, page_image):
    data = synth.make_scanned_pdf([page_image, page_image])
    result = _run(tesseract_settings, data, "scan.pdf", preprocess_preset="auto")
    assert [p.source for p in result.document.pages] == [PageSource.OCR, PageSource.OCR]
    assert all(cer(REFERENCE, p.raw_text, LOOSE) < 0.1 for p in result.document.pages)


def test_mixed_pdf(tesseract_settings, page_image):
    data = synth.make_mixed_pdf("This page has embedded digital text for direct extraction.", page_image)
    result = _run(tesseract_settings, data, "mixed.pdf", preprocess_preset="grayscale")
    first, second = result.document.pages
    assert first.source is PageSource.EMBEDDED_TEXT and "embedded digital text" in first.raw_text
    assert second.source is PageSource.OCR and cer(REFERENCE, second.raw_text, LOOSE) < 0.1


def test_missing_language_with_real_engine(tesseract_settings, page_image):
    result = _run(tesseract_settings, synth.image_bytes(page_image), "a.png", preprocess_preset="original",
                  ocr=OCROptions(languages=("zzz",)))
    assert result.document.pages[0].error.code == "language_data_missing"


def test_ocr_timeout_with_real_engine(tesseract_settings):
    big = synth.render_text_image([f"Line {i} " * 12 for i in range(60)], font_size=30)
    result = _run(tesseract_settings, synth.image_bytes(big), "big.png", preprocess_preset="original",
                  ocr=OCROptions(timeout_seconds=0.01))
    assert result.document.pages[0].error.code == "processing_timeout"


def test_scanned_table_extraction(tesseract_settings):
    rows = [["Item", "Qty", "Price"], ["Apple", "3", "1.20"], ["Banana", "", "0.50"]]
    image = synth.render_table_image(rows)
    result = _run(tesseract_settings, synth.image_bytes(image), "table.png", preprocess_preset="grayscale", extract_tables=True,
                  ocr=OCROptions(psm=6))
    assert result.tables, "expected a table from the ruled grid"
    table = result.tables[0]
    assert table.method == "ocr_grid" and len(table.rows) == 3 and len(table.columns) == 3
    assert table.rows[2][1] is None


def test_structured_extraction_from_ocr(tesseract_settings):
    lines = synth.BENCHMARK_SAMPLES["incident_report"]["lines"]
    image = synth.render_text_image(lines, font_size=34)
    template = TemplateStore(TEMPLATES_DIR).load("incident_report")
    result = _run(tesseract_settings, synth.image_bytes(image), "report.png", preprocess_preset="grayscale", template=template)
    fields = result.structured.fields
    assert fields["caller_name"].value == "Priya Sharma"
    assert fields["incident_date"].normalized_value == "2024-05-17"


def test_benchmark_on_sample_dataset(tesseract_settings, tmp_path):
    synth.write_benchmark_dataset(tmp_path)
    report = run_benchmark(tmp_path, tesseract_settings, preset="auto")
    summary = report.summary
    assert summary["failed"] == 0
    assert summary["mean_cer"] < 0.15


@pytest.fixture
def searchable_settings(tesseract_settings):
    support = check_searchable_pdf_support(tesseract_settings)
    if not support.available:
        pytest.skip(support.reason)
    return tesseract_settings


def test_searchable_pdf(searchable_settings, page_image):
    source = synth.make_scanned_pdf([page_image])
    with open_pdf(source) as doc:
        assert analyze_page(doc[0]).needs_ocr  # no text layer before
    output = create_searchable_pdf(source, searchable_settings, languages=("eng",))
    with open_pdf(output) as doc:
        info = analyze_page(doc[0])
    assert not info.needs_ocr
    assert cer(REFERENCE, info.text, LOOSE) < 0.15


def test_searchable_pdf_mixed_skip_text_keeps_digital_page(searchable_settings, page_image):
    digital = "This first page already has embedded digital text that must be preserved."
    output = create_searchable_pdf(synth.make_mixed_pdf(digital, page_image), searchable_settings)
    with open_pdf(output) as doc:
        assert doc.page_count == 2
        assert digital in " ".join(analyze_page(doc[0]).text.split())
        assert cer(REFERENCE, analyze_page(doc[1]).text, LOOSE) < 0.15
    # The searchable output now goes through the pipeline without needing OCR at all.
    result = DocumentPipeline(searchable_settings).process(output, "searchable.pdf", PipelineOptions())
    assert [p.source for p in result.document.pages] == [PageSource.EMBEDDED_TEXT] * 2


def _a4_scan_300dpi(lines: list[str], rotate: float = 0.0) -> bytes:
    """Image-only A4 PDF at 300 DPI, like a real scanner output."""
    import fitz
    from PIL import Image, ImageDraw

    page = Image.new("L", (2480, 3508), 255)
    draw, font = ImageDraw.Draw(page), synth._font(42)
    for i, line in enumerate(lines):
        draw.text((200, 200 + i * 70), line, fill=0, font=font)
    if rotate:
        page = page.rotate(rotate, resample=Image.BICUBIC, fillcolor=255)
    doc = fitz.open()
    pdf_page = doc.new_page(width=595, height=842)
    pdf_page.insert_image(pdf_page.rect, stream=synth.image_bytes(page))
    return doc.tobytes()


SCAN_LINES = [f"Line {i}: the quick brown fox jumps over the lazy dog." for i in range(20)]


def test_searchable_pdf_force_ocr_replaces_layer_without_duplicates(searchable_settings):
    from docintel.documents.searchable_pdf import OCRMode

    first = create_searchable_pdf(_a4_scan_300dpi(SCAN_LINES), searchable_settings)
    again = create_searchable_pdf(first, searchable_settings, mode=OCRMode.FORCE_OCR)
    with open_pdf(first) as a, open_pdf(again) as b:
        first_words, again_words = analyze_page(a[0]).text.split(), analyze_page(b[0]).text.split()
    assert len(again_words) == len(first_words)  # replaced, not stacked
    assert cer("\n".join(SCAN_LINES), " ".join(again_words), LOOSE) < 0.05


def test_redo_ocr_replaces_layer_without_duplicates(searchable_settings):
    from docintel.documents.searchable_pdf import OCRMode

    first = create_searchable_pdf(_a4_scan_300dpi(SCAN_LINES), searchable_settings)
    redone = create_searchable_pdf(first, searchable_settings, mode=OCRMode.REDO_OCR)
    twice = create_searchable_pdf(redone, searchable_settings, mode=OCRMode.REDO_OCR)
    with open_pdf(first) as a, open_pdf(redone) as b, open_pdf(twice) as c:
        counts = [len(analyze_page(d[0]).text.split()) for d in (a, b, c)]
        text = analyze_page(b[0]).text
    assert counts[0] == counts[1] == counts[2]  # replaced, never stacked
    assert cer("\n".join(SCAN_LINES), text, LOOSE) < 0.05


def test_redo_ocr_fixes_a_wrong_text_layer(searchable_settings):
    """The real use case: a scan whose existing OCR layer is wrong (e.g. from a poor scanner)."""
    import fitz

    from docintel.documents.searchable_pdf import OCRMode

    doc = fitz.open(stream=_a4_scan_300dpi(SCAN_LINES), filetype="pdf")
    doc[0].insert_text((50, 60), "GARBAGE LAYER xq7 zzkk wrong words", fontsize=10, render_mode=3)
    bad = doc.tobytes()
    with open_pdf(bad) as d:
        assert "GARBAGE" in analyze_page(d[0]).text
    fixed = create_searchable_pdf(bad, searchable_settings, mode=OCRMode.REDO_OCR)
    with open_pdf(fixed) as d:
        text = analyze_page(d[0]).text
    assert "GARBAGE" not in text
    assert cer("\n".join(SCAN_LINES), text, LOOSE) < 0.05


def test_redo_ocr_keeps_digital_text_on_mixed_pdf(searchable_settings, page_image):
    from docintel.documents.searchable_pdf import OCRMode

    digital = "This first page already has embedded digital text that must be preserved."
    searchable = create_searchable_pdf(synth.make_mixed_pdf(digital, page_image), searchable_settings)
    redone = create_searchable_pdf(searchable, searchable_settings, mode=OCRMode.REDO_OCR)
    with open_pdf(searchable) as a, open_pdf(redone) as b:
        assert digital in " ".join(analyze_page(b[0]).text.split())
        assert len(analyze_page(b[1]).text.split()) == len(analyze_page(a[1]).text.split())


def test_searchable_pdf_already_has_text_error(searchable_settings):
    from docintel.core.exceptions import PDFProcessingError

    digital = synth.make_digital_pdf(["Fully digital page with plenty of embedded text already present."])
    # skip_text leaves digital pages alone, so this succeeds and keeps the text unchanged.
    out = create_searchable_pdf(digital, searchable_settings)
    with open_pdf(out) as doc:
        assert "embedded text already present" in " ".join(analyze_page(doc[0]).text.split())
    with pytest.raises(PDFProcessingError):
        create_searchable_pdf(b"%PDF-1.4 not really a pdf", searchable_settings)


def test_searchable_pdf_deskew_straightens_300dpi_scan(searchable_settings):
    import numpy as np

    from docintel.documents.pdf_renderer import render_page
    from docintel.preprocessing.deskew import estimate_skew_angle

    output = create_searchable_pdf(_a4_scan_300dpi(SCAN_LINES, rotate=5), searchable_settings, deskew=True, rotate_pages=True)
    with open_pdf(output) as doc:
        assert abs(estimate_skew_angle(np.asarray(render_page(doc[0], 100)))) < 0.5
        text = analyze_page(doc[0]).text
    assert cer("\n".join(SCAN_LINES), text, LOOSE) < 0.05


# --- smart OCR pipeline (real Tesseract) ---------------------------------------------------------
@pytest.mark.parametrize("name", ["shadow_photo", "phone_photo", "scanner_border_skew", "sideways_scan", "dark_mode_screenshot"])
def test_smart_preset_reads_hard_samples(tesseract_settings, name):
    spec = synth.HARD_SAMPLES[name]
    image = synth.render_hard_sample(spec)
    result = _run(tesseract_settings, synth.image_bytes(image), f"{name}.png")  # default preset = smart
    page = result.document.pages[0]
    assert cer("\n".join(spec["lines"]), page.raw_text, LOOSE) < 0.05, page.quality.plan
    assert page.quality is not None and page.quality.plan
    assert page.quality.verdict in ("good", "review")


def test_quality_verdict_flags_missing_text_that_confidence_hides(tesseract_settings):
    """Without preprocessing the shadowed photo loses about half its text at high confidence."""
    image = synth.render_hard_sample(synth.HARD_SAMPLES["shadow_photo"])
    result = _run(tesseract_settings, synth.image_bytes(image), "shadow.png", preprocess_preset="original")
    quality = result.document.pages[0].quality
    assert result.document.pages[0].ocr.mean_confidence > 85  # confidence alone looks fine...
    assert quality.coverage is not None and quality.coverage < 0.75  # ...but coverage shows text is missing
    assert quality.verdict != "good"


def test_smart_benchmark_beats_auto_on_hard_suite(tesseract_settings, tmp_path):
    synth.write_benchmark_dataset(tmp_path, suite="hard")
    smart = run_benchmark(tmp_path, tesseract_settings, preset="smart").summary["mean_cer"]
    auto = run_benchmark(tmp_path, tesseract_settings, preset="auto").summary["mean_cer"]
    assert smart < auto and smart < 0.05
