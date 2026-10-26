"""Shared fixtures. Unit tests use a fake OCR engine; integration tests need real Tesseract."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pytest

os.environ.setdefault("DOCINTEL_LOG_TO_FILE", "false")

from PIL import Image  # noqa: E402

from docintel.config.settings import PROJECT_ROOT, Settings  # noqa: E402
from docintel.evaluation import synthetic_data as synth  # noqa: E402
from docintel.ocr.base_engine import EngineStatus, OCREngine  # noqa: E402
from docintel.ocr.engine_factory import EngineFactory  # noqa: E402
from docintel.ocr.tesseract_engine import TesseractEngine  # noqa: E402
from tests.fakes import FakeEngine  # noqa: E402

TEMPLATES_DIR = PROJECT_ROOT / "data" / "templates"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        temp_dir=tmp_path / "tmp",
        export_dir=tmp_path / "exports",
        templates_dir=TEMPLATES_DIR,
        log_to_file=False,
        tesseract_cmd=None,
    )


def make_factory(settings: Settings, engine: Optional[OCREngine] = None) -> EngineFactory:
    factory = EngineFactory(settings)
    factory.register(engine or FakeEngine())
    return factory


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def text_image() -> Image.Image:
    return synth.render_text_image(["Hello World", "Invoice 12345"])


@pytest.fixture
def png_bytes(text_image: Image.Image) -> bytes:
    return synth.image_bytes(text_image, "PNG")


# --- Integration support ---------------------------------------------------------------
def _real_tesseract_status() -> EngineStatus:
    return TesseractEngine(Settings(_env_file=None, log_to_file=False)).check_availability()


@pytest.fixture(scope="session")
def tesseract_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    status = _real_tesseract_status()
    if not status.available or "eng" not in status.languages:
        pytest.skip(f"Tesseract with 'eng' data not available: {status.reason or 'eng missing'}")
    base = tmp_path_factory.mktemp("integration")
    from docintel.config.settings import get_settings

    env = get_settings()
    return Settings(_env_file=None, temp_dir=base / "tmp", export_dir=base / "exports", templates_dir=TEMPLATES_DIR,
                    log_to_file=False, tesseract_cmd=env.tesseract_cmd, tessdata_dir=env.tessdata_dir)
