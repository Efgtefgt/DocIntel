"""Tesseract OCR engine using the native ``tesseract`` executable.

The executable is invoked directly (no shell, argument list only) rather than through a
wrapper library. This gives reliable timeouts and lets the tessdata path contain spaces.
Installing a Python wrapper does *not* install Tesseract; availability is checked by
actually running the executable.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

from PIL import Image

from docintel.config.settings import Settings
from docintel.core.exceptions import (
    LanguageDataMissingError,
    OCREngineUnavailableError,
    OCRProcessingError,
    ProcessingTimeoutError,
)
from docintel.core.result_models import BoundingBox, OCRResult, OCRWord
from docintel.core.security import run_subprocess
from docintel.core.temp_files import managed_temp_dir
from docintel.ocr.base_engine import (
    EngineStatus,
    OCREngine,
    OCROptions,
    assemble_text,
    mark_low_confidence,
    mean_confidence,
)

logger = logging.getLogger(__name__)

VALID_PSM = range(0, 14)
VALID_OEM = range(0, 4)
PSM_DESCRIPTIONS = {
    1: "Automatic with orientation detection",
    3: "Fully automatic page segmentation (default)",
    4: "Single column of variable-size text",
    6: "Single uniform block of text",
    7: "Single text line",
    11: "Sparse text, as much as possible",
    12: "Sparse text with orientation detection",
    13: "Raw line (bypass layout analysis)",
}


def resolve_tesseract_cmd(configured: Optional[Path]) -> Optional[Path]:
    """Locate the Tesseract executable: configured path, PATH, then standard Windows locations."""
    if configured:
        return configured if configured.is_file() else None
    found = shutil.which("tesseract")
    if found:
        return Path(found)
    candidates = []
    for env_var in ("ProgramFiles", "ProgramFiles(x86)"):
        if os.environ.get(env_var):
            candidates.append(Path(os.environ[env_var]) / "Tesseract-OCR" / "tesseract.exe")
    if os.environ.get("LOCALAPPDATA"):
        candidates.append(Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Tesseract-OCR" / "tesseract.exe")
    return next((c for c in candidates if c.is_file()), None)


class TesseractEngine(OCREngine):
    name = "tesseract"
    display_name = "Tesseract"
    is_deep_learning = False
    supports_psm = True
    supports_multiple_languages = True
    confidence_scale = "Tesseract word confidence (0-100)"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cmd = resolve_tesseract_cmd(settings.tesseract_cmd)

    # --- helpers -------------------------------------------------------------------
    def _require_cmd(self) -> Path:
        if self._cmd is None:
            hint = (
                f"configured path '{self._settings.tesseract_cmd}' does not exist"
                if self._settings.tesseract_cmd
                else "it was not found on PATH or in the standard install folders"
            )
            raise OCREngineUnavailableError(
                f"Tesseract executable not available: {hint}. Install Tesseract and/or set DOCINTEL_TESSERACT_CMD."
            )
        return self._cmd

    def _data_args(self) -> list[str]:
        return ["--tessdata-dir", str(self._settings.tessdata_dir)] if self._settings.tessdata_dir else []

    # --- interface -----------------------------------------------------------------
    def version(self) -> str:
        return _cached_version(str(self._require_cmd()))

    def available_languages(self) -> list[str]:
        return list(_cached_languages(str(self._require_cmd()), tuple(self._data_args())))

    def check_availability(self) -> EngineStatus:
        status = EngineStatus(self.name, self.display_name, available=False, is_deep_learning=False)
        try:
            status.version = self.version()
            status.languages = [lang for lang in self.available_languages() if lang != "osd"]
        except (OCREngineUnavailableError, OCRProcessingError, ProcessingTimeoutError) as exc:
            status.reason = exc.user_message
            return status
        if not status.languages:
            status.reason = "Tesseract runs, but no language data (.traineddata) was found."
            return status
        status.available = True
        return status

    def recognize(self, image: Image.Image, options: OCROptions) -> OCRResult:
        cmd = self._require_cmd()
        self.validate_languages(options.languages)
        if options.psm is not None and options.psm not in VALID_PSM:
            raise OCRProcessingError(f"Invalid page segmentation mode {options.psm}.")
        if options.oem is not None and options.oem not in VALID_OEM:
            raise OCRProcessingError(f"Invalid OCR engine mode {options.oem}.")

        args_tail = [*self._data_args(), "-l", "+".join(options.languages)]
        if options.psm is not None:
            args_tail += ["--psm", str(options.psm)]
        if options.oem is not None:
            args_tail += ["--oem", str(options.oem)]
        args_tail += ["-c", "preserve_interword_spaces=1", "tsv"]

        started = time.perf_counter()
        with managed_temp_dir(self._settings) as tmp:
            image_path = tmp / "page.png"
            image.save(image_path, format="PNG")
            proc = run_subprocess([str(cmd), str(image_path), "stdout", *args_tail], timeout=options.timeout_seconds)
        duration = time.perf_counter() - started

        if proc.returncode != 0:
            raise self._error_from_stderr(proc.stderr)

        words = parse_tsv(proc.stdout)
        mark_low_confidence(words, options.low_confidence_threshold)
        logger.info(
            "Tesseract OCR complete",
            extra={"duration_s": round(duration, 3), "word_count": len(words), "psm": options.psm},
        )
        return OCRResult(
            engine=self.name,
            engine_version=self.version(),
            languages=list(options.languages),
            text=assemble_text(words),
            words=words,
            mean_confidence=mean_confidence(words),
            confidence_scale=self.confidence_scale,
            duration_seconds=round(duration, 3),
            image_width=image.width,
            image_height=image.height,
            options=options.as_dict(),
        )

    def detect_orientation(self, image: Image.Image, timeout_seconds: float = 30.0) -> Optional[int]:
        """Use Tesseract OSD (requires ``osd.traineddata``). Returns None when unsure."""
        try:
            cmd = self._require_cmd()
            if "osd" not in self.available_languages():
                return None
        except (OCREngineUnavailableError, OCRProcessingError):
            return None
        with managed_temp_dir(self._settings) as tmp:
            image_path = tmp / "osd.png"
            image.save(image_path, format="PNG")
            try:
                proc = run_subprocess(
                    [str(cmd), str(image_path), "stdout", *self._data_args(), "--psm", "0"],
                    timeout=timeout_seconds,
                )
            except ProcessingTimeoutError:
                return None
        # OSD fails with "Too few characters" on sparse pages; treat that as "unknown".
        rotate = re.search(r"Rotate:\s*(\d+)", proc.stdout)
        confidence = re.search(r"Orientation confidence:\s*([\d.]+)", proc.stdout)
        if proc.returncode != 0 or not rotate:
            return None
        if confidence and float(confidence.group(1)) < 1.0:
            return None
        # Tesseract reports the clockwise rotation that makes the page upright.
        return int(rotate.group(1)) % 360

    @staticmethod
    def _error_from_stderr(stderr: str) -> Exception:
        first_line = next((ln.strip() for ln in stderr.splitlines() if ln.strip()), "unknown error")
        if "Failed loading language" in stderr or "Could not initialize tesseract" in stderr:
            return LanguageDataMissingError(f"Tesseract could not load the language data: {first_line[:200]}")
        return OCRProcessingError(f"Tesseract failed: {first_line[:200]}")


def parse_tsv(tsv: str) -> list[OCRWord]:
    """Parse Tesseract TSV output into words (level 5 rows with non-empty text)."""
    words: list[OCRWord] = []
    lines = tsv.splitlines()
    for row in lines[1:]:  # first line is the header
        parts = row.split("\t", 11)
        if len(parts) < 12 or parts[0] != "5":
            continue
        text = parts[11].strip()
        if not text:
            continue
        try:
            conf = float(parts[10])
            left, top, width, height = (int(parts[i]) for i in (6, 7, 8, 9))
            block, paragraph, line = int(parts[2]), int(parts[3]), int(parts[4])
        except ValueError:
            continue
        words.append(
            OCRWord(
                text=text,
                confidence=round(conf, 2) if conf >= 0 else None,
                bbox=BoundingBox(x=left, y=top, width=width, height=height),
                block=block,
                paragraph=paragraph,
                line=line,
            )
        )
    return words


@lru_cache(maxsize=8)
def _cached_version(cmd: str) -> str:
    proc = run_subprocess([cmd, "--version"], timeout=15)
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 or "tesseract" not in output.lower():
        raise OCREngineUnavailableError(f"'{cmd}' did not respond like a Tesseract executable.")
    return output.splitlines()[0].strip()


@lru_cache(maxsize=8)
def _cached_languages(cmd: str, data_args: tuple[str, ...]) -> tuple[str, ...]:
    proc = run_subprocess([cmd, "--list-langs", *data_args], timeout=15)
    if proc.returncode != 0:
        raise OCRProcessingError("Tesseract could not list installed languages; check the tessdata directory.")
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    langs = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.lower().startswith(("list of available", "error", "warning")):
            continue
        if re.fullmatch(r"[A-Za-z0-9_]+", line):
            langs.append(line)
    return tuple(sorted(set(langs)))


def clear_tesseract_caches() -> None:
    """Forget cached version/language lookups (after installing language data)."""
    _cached_version.cache_clear()
    _cached_languages.cache_clear()
