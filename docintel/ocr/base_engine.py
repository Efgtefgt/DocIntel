"""Common OCR engine interface.

To add an engine: subclass :class:`OCREngine`, implement the abstract methods, and
register the class in ``engine_factory.ENGINE_CLASSES``. Nothing else needs to change.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import groupby
from typing import ClassVar, Optional, Sequence

from PIL import Image

from docintel.core.exceptions import LanguageDataMissingError
from docintel.core.result_models import OCRResult, OCRWord

_LANGUAGE_CODE = re.compile(r"^[A-Za-z0-9_]{2,32}$")


@dataclass(frozen=True)
class OCROptions:
    languages: tuple[str, ...] = ("eng",)
    psm: Optional[int] = 3           # Tesseract page segmentation mode; ignored by other engines
    oem: Optional[int] = None        # Tesseract engine mode; None = Tesseract default
    timeout_seconds: float = 60.0
    low_confidence_threshold: float = 60.0  # on the engine's 0-100 display scale

    def as_dict(self) -> dict[str, object]:
        return {
            "languages": list(self.languages),
            "psm": self.psm,
            "oem": self.oem,
            "timeout_seconds": self.timeout_seconds,
            "low_confidence_threshold": self.low_confidence_threshold,
        }


@dataclass
class EngineStatus:
    name: str
    display_name: str
    available: bool
    reason: str = ""
    version: str = ""
    languages: list[str] = field(default_factory=list)
    is_deep_learning: bool = False


class OCREngine(ABC):
    """Abstract OCR engine. Implementations must be safe to construct without loading models."""

    name: ClassVar[str]
    display_name: ClassVar[str]
    is_deep_learning: ClassVar[bool] = False
    supports_psm: ClassVar[bool] = False
    supports_multiple_languages: ClassVar[bool] = True
    confidence_scale: ClassVar[str] = ""

    @abstractmethod
    def check_availability(self) -> EngineStatus:
        """Report whether the engine and its data are usable, without raising."""

    @abstractmethod
    def available_languages(self) -> list[str]:
        """Return language codes whose model/data files are installed."""

    @abstractmethod
    def recognize(self, image: Image.Image, options: OCROptions) -> OCRResult:
        """Run OCR on one image.

        Raises:
            OCREngineUnavailableError, LanguageDataMissingError, OCRProcessingError,
            ProcessingTimeoutError.
        """

    def detect_orientation(self, image: Image.Image, timeout_seconds: float = 30.0) -> Optional[int]:
        """Return clockwise degrees (0/90/180/270) needed to make the page upright, or None."""
        return None

    def close(self) -> None:
        """Release loaded models. Engines are re-initialised lazily on next use."""

    def validate_languages(self, languages: Sequence[str]) -> None:
        if not languages:
            raise LanguageDataMissingError("Select at least one OCR language.")
        bad = [lang for lang in languages if not _LANGUAGE_CODE.match(lang)]
        if bad:
            raise LanguageDataMissingError(f"Invalid language code(s): {', '.join(bad)}.")
        if not self.supports_multiple_languages and len(languages) > 1:
            raise LanguageDataMissingError(f"{self.display_name} supports one language at a time.")
        installed = set(self.available_languages())
        missing = [lang for lang in languages if lang not in installed]
        if missing:
            raise LanguageDataMissingError(
                f"Language data not installed for {self.display_name}: {', '.join(missing)}. "
                f"Installed: {', '.join(sorted(installed)) or 'none'}."
            )


def mark_low_confidence(words: list[OCRWord], threshold: float) -> None:
    for word in words:
        word.low_confidence = word.confidence is not None and word.confidence < threshold


def mean_confidence(words: Sequence[OCRWord]) -> Optional[float]:
    values = [w.confidence for w in words if w.confidence is not None]
    return round(sum(values) / len(values), 2) if values else None


def assemble_text(words: Sequence[OCRWord]) -> str:
    """Rebuild text from words, keeping line breaks and blank lines between blocks.

    Words must already be in reading order with block/paragraph/line indices set.
    """
    blocks = []
    for _, block_words in groupby(words, key=lambda w: w.block):
        paragraphs = []
        for _, par_words in groupby(block_words, key=lambda w: w.paragraph):
            lines = [" ".join(w.text for w in line_words) for _, line_words in groupby(par_words, key=lambda w: w.line)]
            paragraphs.append("\n".join(lines))
        blocks.append("\n".join(paragraphs))
    return "\n\n".join(blocks)
