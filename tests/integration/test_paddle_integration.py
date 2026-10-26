"""PaddleOCR end-to-end tests (skipped unless paddleocr is installed in this environment).

Run them in the PaddleOCR environment, e.g.:
    & "$env:USERPROFILE\\.venvs\\docintel-paddle\\Scripts\\python.exe" -m pytest tests/integration/test_paddle_integration.py
The first run downloads the models (~16 MB) from PaddleOCR's servers and needs internet once.
"""

from __future__ import annotations

import importlib.util

import pytest

from docintel.config.settings import Settings
from docintel.core.exceptions import LanguageDataMissingError
from docintel.core.pipeline import DocumentPipeline, PipelineOptions
from docintel.core.result_models import PageSource
from docintel.evaluation import synthetic_data as synth
from docintel.evaluation.metrics import EvalNormalization, cer
from docintel.ocr.base_engine import OCROptions
from docintel.ocr.engine_factory import EngineFactory
from tests.conftest import TEMPLATES_DIR

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(importlib.util.find_spec("paddleocr") is None, reason="paddleocr not installed"),
]
LINES = ["The quick brown fox jumps over the lazy dog.", "Invoice number 48213 was paid in full."]
REFERENCE = "\n".join(LINES)
# PaddleOCR's English model often drops sentence-final full stops; score letters and digits.
LENIENT = EvalNormalization(lowercase=True, remove_punctuation=True)


@pytest.fixture(scope="module")
def paddle_factory(tmp_path_factory):
    base = tmp_path_factory.mktemp("paddle")
    settings = Settings(_env_file=None, log_to_file=False, enable_paddle=True, temp_dir=base, templates_dir=TEMPLATES_DIR)
    factory = EngineFactory(settings)
    yield factory
    factory.close_all()


def _opts(**kw) -> OCROptions:
    return OCROptions(languages=("en",), psm=None, timeout_seconds=300, **kw)


def test_paddle_available(paddle_factory):
    status = next(s for s in paddle_factory.statuses() if s.name == "paddle")
    assert status.available, status.reason
    assert "en" in status.languages and status.is_deep_learning


def test_paddle_recognises_text_with_line_order(paddle_factory):
    engine = paddle_factory.get("paddle")
    result = engine.recognize(synth.render_text_image(LINES), _opts())
    assert cer(REFERENCE, result.text, LENIENT) < 0.05
    assert result.text.index("quick") < result.text.index("Invoice")  # reading order kept
    assert "\n" in result.text
    assert result.mean_confidence is not None and 0 <= result.mean_confidence <= 100
    assert "not comparable" in result.confidence_scale


def test_paddle_rejects_multiple_languages(paddle_factory):
    with pytest.raises(LanguageDataMissingError):
        paddle_factory.get("paddle").recognize(synth.render_text_image(LINES), _opts().__class__(languages=("en", "ch")))


def test_paddle_through_pipeline_on_scanned_pdf(paddle_factory):
    data = synth.make_scanned_pdf([synth.render_text_image(LINES)])
    options = PipelineOptions(engine="paddle", ocr=_opts(), preprocess_preset="grayscale")
    result = DocumentPipeline(paddle_factory.settings, paddle_factory).process(data, "scan.pdf", options)
    assert result.succeeded, result.errors
    page = result.document.pages[0]
    assert page.source is PageSource.OCR and page.ocr.engine == "paddle"
    assert cer(REFERENCE, page.raw_text, LENIENT) < 0.05


def test_paddle_structured_extraction(paddle_factory):
    from docintel.extraction.templates import TemplateStore

    lines = synth.BENCHMARK_SAMPLES["incident_report"]["lines"]
    options = PipelineOptions(engine="paddle", ocr=_opts(), preprocess_preset="original",
                              template=TemplateStore(TEMPLATES_DIR).load("incident_report"))
    result = DocumentPipeline(paddle_factory.settings, paddle_factory).process(
        synth.image_bytes(synth.render_text_image(lines, font_size=34)), "report.png", options)
    fields = result.structured.fields
    # PaddleOCR may append a spurious full stop; the verbatim value keeps whatever was read,
    # the normalized value drops trailing separators.
    assert fields["caller_name"].value.startswith("Priya Sharma")
    assert fields["caller_name"].normalized_value == "Priya Sharma"
    assert fields["incident_date"].normalized_value == "2024-05-17"


def test_paddle_timeout(paddle_factory):
    from docintel.core.exceptions import ProcessingTimeoutError

    big = synth.render_text_image([f"Line {i} with several words of text" for i in range(80)], font_size=30)
    with pytest.raises(ProcessingTimeoutError):
        paddle_factory.get("paddle").recognize(big, OCROptions(languages=("en",), psm=None, timeout_seconds=0.01))


def test_paddle_model_released_on_close(paddle_factory):
    engine = paddle_factory.get("paddle")
    engine.recognize(synth.render_text_image(LINES), _opts())
    assert engine._model is not None
    engine.close()
    assert engine._model is None
    # Lazily reloads on next use.
    assert "quick" in engine.recognize(synth.render_text_image(LINES), _opts()).text
