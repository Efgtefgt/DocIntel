"""Test doubles."""

from __future__ import annotations

import time

from PIL import Image

from docintel.core.exceptions import ProcessingTimeoutError
from docintel.core.result_models import BoundingBox, OCRResult, OCRWord
from docintel.ocr.base_engine import EngineStatus, OCREngine, OCROptions, assemble_text, mark_low_confidence, mean_confidence


class FakeEngine(OCREngine):
    """Deterministic stand-in for an OCR engine.

    Returns ``text`` split into words laid out on a simple grid; ``delay`` simulates slow
    recognition and honours the timeout like a real engine.
    """

    name = "fake"
    display_name = "Fake OCR"
    supports_psm = True
    confidence_scale = "fake (0-100)"

    def __init__(self, text: str = "Hello World", confidence: float = 95.0, delay: float = 0.0,
                 available: bool = True, languages: tuple[str, ...] = ("eng", "osd"), verbatim: bool = False) -> None:
        self.text = text
        self.verbatim = verbatim
        self.confidence = confidence
        self.delay = delay
        self.available = available
        self.languages = languages
        self.calls: list[tuple[int, int]] = []

    def check_availability(self) -> EngineStatus:
        return EngineStatus(self.name, self.display_name, self.available,
                            reason="" if self.available else "fake engine disabled",
                            languages=list(self.languages))

    def available_languages(self) -> list[str]:
        return list(self.languages)

    def recognize(self, image: Image.Image, options: OCROptions) -> OCRResult:
        self.validate_languages(options.languages)
        self.calls.append(image.size)
        if self.delay:
            if self.delay > options.timeout_seconds:
                time.sleep(min(options.timeout_seconds, 0.2))
                raise ProcessingTimeoutError(f"Fake OCR exceeded the {options.timeout_seconds:.0f}s time limit.")
            time.sleep(self.delay)
        words = []
        for line_no, line in enumerate(self.text.split("\n")):
            for word_no, token in enumerate(line.split()):
                words.append(OCRWord(text=token, confidence=self.confidence,
                                     bbox=BoundingBox(x=10 + 120 * word_no, y=10 + 40 * line_no, width=100, height=30),
                                     block=1, paragraph=1, line=line_no))
        mark_low_confidence(words, options.low_confidence_threshold)
        # verbatim=True returns ``text`` exactly (e.g. with irregular whitespace) like a raw engine might.
        text = self.text if self.verbatim else assemble_text(words)
        return OCRResult(engine=self.name, languages=list(options.languages), text=text, words=words,
                         mean_confidence=mean_confidence(words), confidence_scale=self.confidence_scale,
                         image_width=image.width, image_height=image.height, options=options.as_dict())
