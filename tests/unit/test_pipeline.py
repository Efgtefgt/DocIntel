"""End-to-end pipeline behaviour with a fake OCR engine."""

from __future__ import annotations

import io

from docintel.core.pipeline import DocumentPipeline, PipelineOptions
from docintel.core.result_models import PageSource, StageStatus
from docintel.evaluation import synthetic_data as synth
from docintel.extraction.templates import TemplateStore
from docintel.ocr.base_engine import OCROptions
from tests.conftest import TEMPLATES_DIR, make_factory
from tests.fakes import FakeEngine

FAKE = PipelineOptions(engine="fake", preprocess_preset="original")


def test_image_pipeline_success(settings, png_bytes):
    engine = FakeEngine(text="Hello World\nInvoice 12345")
    result = DocumentPipeline(settings, make_factory(settings, engine)).process(png_bytes, "scan.png", FAKE)
    assert result.succeeded
    page = result.document.pages[0]
    assert page.source is PageSource.OCR
    assert page.raw_text == page.normalized_text == "Hello World\nInvoice 12345"
    assert page.ocr.mean_confidence == 95.0
    stages = {s.stage: s.status for s in result.stages}
    assert stages["validation"] is StageStatus.SUCCESS and stages["tables"] is StageStatus.SKIPPED
    assert result.duration_seconds >= 0


def test_raw_text_kept_separately_from_normalized(settings, png_bytes):
    engine = FakeEngine(text="Total:  $1,250.00\n\n\n\nRef #A-12   ", verbatim=True)
    page = DocumentPipeline(settings, make_factory(settings, engine)).process(png_bytes, "a.png", FAKE).document.pages[0]
    assert page.raw_text == "Total:  $1,250.00\n\n\n\nRef #A-12   "  # untouched OCR output
    assert page.normalized_text == "Total: $1,250.00\n\nRef #A-12"
    assert "repeated spaces collapsed" in page.normalization_changes


def test_invalid_upload_returns_structured_error(settings):
    result = DocumentPipeline(settings, make_factory(settings)).process(b"not an image", "x.png", FAKE)
    assert result.document is None and not result.succeeded
    assert result.errors[0].stage == "validation" and result.errors[0].code == "unsupported_format"


def test_engine_unavailable_reported_per_page(settings, png_bytes):
    factory = make_factory(settings, FakeEngine(available=False))
    result = DocumentPipeline(settings, factory).process(png_bytes, "a.png", FAKE)
    assert not result.succeeded
    assert result.document.pages[0].error.code == "ocr_engine_unavailable"


def test_missing_language_reported(settings, png_bytes):
    options = PipelineOptions(engine="fake", preprocess_preset="original", ocr=OCROptions(languages=("hin",)))
    result = DocumentPipeline(settings, make_factory(settings)).process(png_bytes, "a.png", options)
    assert result.document.pages[0].error.code == "language_data_missing"


def test_ocr_timeout_reported(settings, png_bytes):
    options = PipelineOptions(engine="fake", preprocess_preset="original", ocr=OCROptions(timeout_seconds=0.05))
    factory = make_factory(settings, FakeEngine(delay=5))
    result = DocumentPipeline(settings, factory).process(png_bytes, "a.png", options)
    assert result.document.pages[0].error.code == "processing_timeout"


def test_document_deadline_skips_remaining_pages(settings, text_image):
    data = synth.make_scanned_pdf([text_image] * 4)
    strict = settings.model_copy(update={"document_timeout_seconds": 0.3})
    factory = make_factory(strict, FakeEngine(delay=0.2))
    result = DocumentPipeline(strict, factory).process(data, "slow.pdf", FAKE)
    codes = [p.error.code if p.error else None for p in result.document.pages]
    assert len(codes) == 4
    assert codes[0] is None
    assert "processing_timeout" in codes


def test_multiframe_tiff_pages_and_range(settings, text_image):
    buffer = io.BytesIO()
    text_image.save(buffer, format="TIFF", save_all=True, append_images=[text_image, text_image])
    options = PipelineOptions(engine="fake", preprocess_preset="original", page_range="2-3")
    result = DocumentPipeline(settings, make_factory(settings)).process(buffer.getvalue(), "multi.tif", options)
    assert [p.page_number for p in result.document.pages] == [2, 3]


def test_mixed_pdf_through_pipeline(settings, text_image):
    data = synth.make_mixed_pdf("Embedded text page with enough characters for detection.", text_image)
    result = DocumentPipeline(settings, make_factory(settings, FakeEngine(text="scanned"))).process(data, "mixed.pdf", FAKE)
    sources = [p.source for p in result.document.pages]
    assert sources == [PageSource.EMBEDDED_TEXT, PageSource.OCR]
    combined = result.document.combined_text()
    assert "===== Page 1 (embedded_text) =====" in combined and "===== Page 2 (ocr) =====" in combined


def test_structured_extraction_stage(settings, png_bytes):
    template = TemplateStore(TEMPLATES_DIR).load("incident_report")
    engine = FakeEngine(text="Caller Name: Asha Rao\nPhone Number: 080 2222 3333")
    options = PipelineOptions(engine="fake", preprocess_preset="original", template=template)
    result = DocumentPipeline(settings, make_factory(settings, engine)).process(png_bytes, "a.png", options)
    fields = result.structured.fields
    assert fields["caller_name"].value == "Asha Rao"
    assert fields["phone_number"].normalized_value == "08022223333"
    assert fields["incident_location"].value is None


def test_batch_sequential_and_concurrent(settings, png_bytes):
    files = [(png_bytes, f"f{i}.png") for i in range(3)] + [(b"junk", "bad.png")]
    for workers in (1, 2):
        s = settings.model_copy(update={"max_workers": workers})
        done = []
        results = DocumentPipeline(s, make_factory(s)).process_batch(files, FAKE, on_file_done=lambda i, r: done.append(i))
        assert [r.succeeded for r in results] == [True, True, True, False]
        assert sorted(done) == [0, 1, 2, 3]


def test_progress_callback_reaches_completion(settings, png_bytes):
    events = []
    DocumentPipeline(settings, make_factory(settings)).process(png_bytes, "a.png", FAKE, progress=lambda f, m: events.append(f))
    assert events[0] == 0.0 and events[-1] == 1.0
