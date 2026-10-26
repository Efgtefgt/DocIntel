"""OCR engine interface, Tesseract parsing and availability handling (no real OCR)."""

from __future__ import annotations

from pathlib import Path

import pytest

from docintel.core.exceptions import LanguageDataMissingError, OCREngineUnavailableError
from docintel.ocr.base_engine import OCROptions
from docintel.ocr.engine_factory import EngineFactory
from docintel.ocr.paddle_engine import PaddleEngine, _lines_to_words
from docintel.ocr.tesseract_engine import TesseractEngine, parse_tsv
from tests.fakes import FakeEngine

TSV = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
    "1\t1\t0\t0\t0\t0\t0\t0\t500\t200\t-1\t\n"
    "5\t1\t1\t1\t1\t1\t10\t10\t80\t20\t96.5\tHello\n"
    "5\t1\t1\t1\t1\t2\t100\t10\t90\t20\t91.0\tWorld\n"
    "5\t1\t1\t1\t2\t1\t10\t40\t80\t20\t42.0\tInv0ice\n"
    "5\t1\t2\t1\t1\t1\t10\t100\t80\t20\t88.0\tFooter\n"
    "5\t1\t2\t1\t1\t2\t100\t100\t80\t20\t-1\t \n"
)


def test_parse_tsv_and_assemble_text():
    from docintel.ocr.base_engine import assemble_text

    words = parse_tsv(TSV)
    assert [w.text for w in words] == ["Hello", "World", "Inv0ice", "Footer"]
    assert words[0].confidence == 96.5
    assert assemble_text(words) == "Hello World\nInv0ice\n\nFooter"


def test_missing_tesseract_reports_unavailable(settings, tmp_path: Path, text_image):
    engine = TesseractEngine(settings.model_copy(update={"tesseract_cmd": tmp_path / "nope" / "tesseract.exe"}))
    status = engine.check_availability()
    assert not status.available
    assert "does not exist" in status.reason
    with pytest.raises(OCREngineUnavailableError):
        engine.recognize(text_image, OCROptions())


def test_factory_raises_for_unavailable_engine(settings):
    factory = EngineFactory(settings)
    factory.register(FakeEngine(available=False))
    with pytest.raises(OCREngineUnavailableError, match="not available"):
        factory.get("fake")


def test_factory_unknown_engine(settings):
    with pytest.raises(OCREngineUnavailableError, match="Unknown OCR engine"):
        EngineFactory(settings).get("does-not-exist")


def test_factory_lazy_instances(settings):
    factory = EngineFactory(settings)
    assert factory._instances == {}  # nothing constructed or loaded at startup
    factory.statuses()
    assert set(factory._instances) >= {"tesseract", "paddle"}


def test_missing_language_data(fake_engine, text_image):
    with pytest.raises(LanguageDataMissingError, match="not installed"):
        fake_engine.recognize(text_image, OCROptions(languages=("deu",)))


@pytest.mark.parametrize("bad", ["eng;rm -rf", "../eng", "e"])
def test_malformed_language_codes_rejected(fake_engine, text_image, bad):
    with pytest.raises(LanguageDataMissingError, match="Invalid language"):
        fake_engine.recognize(text_image, OCROptions(languages=(bad,)))


def test_low_confidence_flags(text_image):
    engine = FakeEngine(text="maybe words", confidence=40)
    result = engine.recognize(text_image, OCROptions(low_confidence_threshold=60))
    assert all(w.low_confidence for w in result.words)
    assert result.mean_confidence == 40


def test_paddle_disabled_by_default(settings):
    status = PaddleEngine(settings).check_availability()
    assert not status.available and "Disabled" in status.reason


def test_paddle_single_language_only(settings):
    engine = PaddleEngine(settings)
    with pytest.raises(LanguageDataMissingError, match="one language"):
        engine.validate_languages(["en", "ch"])


def test_paddle_line_grouping():
    detections = [
        ([[200, 10], [300, 10], [300, 30], [200, 30]], ("right", 0.9)),
        ([[10, 12], [100, 12], [100, 32], [10, 32]], ("left", 0.8)),
        ([[10, 60], [100, 60], [100, 80], [10, 80]], ("next", 0.7)),
    ]
    words = _lines_to_words(detections)
    assert [w.text for w in words] == ["left", "right", "next"]
    assert words[0].line == words[1].line != words[2].line
    assert words[0].confidence == 80.0
