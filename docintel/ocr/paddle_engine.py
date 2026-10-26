"""Optional PaddleOCR engine (deep learning, CPU mode).

Targets the PaddleOCR 2.x API (``paddleocr==2.9.1`` with ``paddlepaddle==2.6.2``).
The model is loaded lazily on the first ``recognize`` call and only one language model
is kept in memory at a time (~300-600 MB RSS on CPU). Models are downloaded by PaddleOCR
on first use, so run one recognition during setup while online; afterwards it is offline.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Optional

import numpy as np
from PIL import Image

from docintel.config.settings import Settings
from docintel.core.exceptions import OCREngineUnavailableError, OCRProcessingError, ProcessingTimeoutError
from docintel.core.result_models import BoundingBox, OCRResult, OCRWord
from docintel.ocr.base_engine import EngineStatus, OCREngine, OCROptions, assemble_text, mark_low_confidence, mean_confidence

logger = logging.getLogger(__name__)

# Language codes accepted by PaddleOCR 2.x (subset of the most common ones).
PADDLE_LANGUAGES = (
    "en", "ch", "chinese_cht", "japan", "korean", "french", "german", "it", "es", "pt",
    "ru", "ar", "hi", "ta", "te", "ka", "mr", "ne", "latin", "cyrillic", "devanagari",
)


class PaddleEngine(OCREngine):
    name = "paddle"
    display_name = "PaddleOCR"
    is_deep_learning = True
    supports_psm = False
    supports_multiple_languages = False
    confidence_scale = "PaddleOCR recognition score x100 (not comparable with Tesseract)"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None
        self._model_lang: Optional[str] = None
        # A dedicated worker thread lets us enforce a timeout. Note: a timed-out inference
        # keeps running in the background until it finishes; Python cannot kill a thread.
        self._executor: Optional[ThreadPoolExecutor] = None

    def check_availability(self) -> EngineStatus:
        status = EngineStatus(self.name, self.display_name, available=False, is_deep_learning=True)
        if not self._settings.enable_paddle:
            status.reason = "Disabled. Set DOCINTEL_ENABLE_PADDLE=true after installing requirements-paddle.txt."
            return status
        missing = [m for m in ("paddle", "paddleocr") if importlib.util.find_spec(m) is None]
        if missing:
            status.reason = f"Python package(s) not installed: {', '.join(missing)} (see requirements-paddle.txt)."
            return status
        status.version = importlib.metadata.version("paddleocr")
        status.languages = list(PADDLE_LANGUAGES)
        status.available = True
        return status

    def available_languages(self) -> list[str]:
        return list(PADDLE_LANGUAGES)

    def _load(self, lang: str) -> Any:
        if self._model is not None and self._model_lang == lang:
            return self._model
        status = self.check_availability()
        if not status.available:
            raise OCREngineUnavailableError(status.reason)
        self.close()  # keep only one model in memory
        from paddleocr import PaddleOCR  # heavy import, deferred until needed

        logger.info("Loading PaddleOCR model", extra={"lang": lang})
        self._model = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=False, show_log=False)
        self._model_lang = lang
        return self._model

    def recognize(self, image: Image.Image, options: OCROptions) -> OCRResult:
        self.validate_languages(options.languages)
        lang = options.languages[0]
        model = self._load(lang)
        arr = np.asarray(image.convert("RGB"))[:, :, ::-1]  # PaddleOCR expects BGR
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paddle-ocr")
        started = time.perf_counter()
        future = self._executor.submit(model.ocr, arr, cls=True)
        try:
            raw = future.result(timeout=options.timeout_seconds)
        except FutureTimeout as exc:
            raise ProcessingTimeoutError(f"PaddleOCR exceeded the {options.timeout_seconds:.0f}s time limit.") from exc
        except Exception as exc:  # PaddleOCR raises assorted internal exception types
            raise OCRProcessingError(f"PaddleOCR failed: {type(exc).__name__}") from exc
        duration = time.perf_counter() - started

        words = _lines_to_words(raw[0] if raw else None)
        mark_low_confidence(words, options.low_confidence_threshold)
        return OCRResult(
            engine=self.name,
            engine_version=importlib.metadata.version("paddleocr"),
            languages=[lang],
            text=assemble_text(words),
            words=words,
            mean_confidence=mean_confidence(words),
            confidence_scale=self.confidence_scale,
            duration_seconds=round(duration, 3),
            image_width=image.width,
            image_height=image.height,
            options=options.as_dict(),
        )

    def close(self) -> None:
        self._model = None
        self._model_lang = None
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None


def _lines_to_words(detections: Optional[list]) -> list[OCRWord]:
    """Convert PaddleOCR line detections into ordered OCRWord entries.

    PaddleOCR returns text *lines* (box + text + score). Lines whose vertical centres are
    close are grouped into one visual line, then ordered left to right.
    """
    if not detections:
        return []
    items = []
    for box, (text, score) in detections:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        bbox = BoundingBox(x=int(min(xs)), y=int(min(ys)), width=int(max(xs) - min(xs)), height=int(max(ys) - min(ys)))
        items.append((bbox, str(text), float(score) * 100))
    items.sort(key=lambda it: (it[0].center_y, it[0].x))

    words: list[OCRWord] = []
    line_index = 0
    current_center: Optional[float] = None
    current_height = 0
    line_buffer: list[tuple[BoundingBox, str, float]] = []

    def flush() -> None:
        for bbox, text, score in sorted(line_buffer, key=lambda it: it[0].x):
            words.append(OCRWord(text=text, confidence=round(score, 2), bbox=bbox, block=1, paragraph=1, line=line_index))

    for bbox, text, score in items:
        if current_center is not None and abs(bbox.center_y - current_center) > max(current_height, bbox.height) / 2:
            flush()
            line_buffer = []
            line_index += 1
        if not line_buffer:
            current_center, current_height = bbox.center_y, bbox.height
        line_buffer.append((bbox, text, score))
    flush()
    return words
