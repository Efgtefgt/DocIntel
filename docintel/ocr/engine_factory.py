"""Engine registry with lazy initialisation.

Engines are constructed on first use. Deep-learning engines are mutually exclusive in
memory: obtaining one releases any other loaded deep-learning engine (8 GB RAM target).
"""

from __future__ import annotations

import logging

from docintel.config.settings import Settings
from docintel.core.exceptions import OCREngineUnavailableError
from docintel.ocr.base_engine import EngineStatus, OCREngine
from docintel.ocr.paddle_engine import PaddleEngine
from docintel.ocr.tesseract_engine import TesseractEngine

logger = logging.getLogger(__name__)

ENGINE_CLASSES: dict[str, type[OCREngine]] = {
    TesseractEngine.name: TesseractEngine,
    PaddleEngine.name: PaddleEngine,
}


class EngineFactory:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._instances: dict[str, OCREngine] = {}

    @property
    def settings(self) -> Settings:
        return self._settings

    def _instance(self, name: str) -> OCREngine:
        if name in self._instances:
            return self._instances[name]
        if name not in ENGINE_CLASSES:
            raise OCREngineUnavailableError(f"Unknown OCR engine '{name}'. Known: {', '.join(ENGINE_CLASSES)}.")
        if name not in self._instances:
            self._instances[name] = ENGINE_CLASSES[name](self._settings)
        return self._instances[name]

    def statuses(self) -> list[EngineStatus]:
        names = list(dict.fromkeys([*ENGINE_CLASSES, *self._instances]))
        return [self._instance(name).check_availability() for name in names]

    def get(self, name: str) -> OCREngine:
        """Return an available engine, or raise ``OCREngineUnavailableError`` with the reason."""
        engine = self._instance(name)
        status = engine.check_availability()
        if not status.available:
            raise OCREngineUnavailableError(f"{status.display_name} is not available: {status.reason}")
        if engine.is_deep_learning:
            for other_name, other in self._instances.items():
                if other_name != name and other.is_deep_learning:
                    other.close()
        return engine

    def register(self, engine: OCREngine) -> None:
        """Register a pre-built engine instance (used for tests and custom engines)."""
        self._instances[engine.name] = engine

    def close_all(self) -> None:
        for engine in self._instances.values():
            engine.close()
