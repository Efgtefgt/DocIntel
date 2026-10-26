"""Optional searchable-PDF generation with OCRmyPDF.

OCRmyPDF is run as a separate process (``python -m ocrmypdf``) with an argument list, so
the time limit can be enforced and no shell is involved. It needs the native Tesseract
and Ghostscript executables in addition to the Python package.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

from docintel.config.settings import Settings
from docintel.core.exceptions import (
    EncryptedDocumentError,
    OptionalDependencyMissingError,
    PDFProcessingError,
)
from docintel.core.security import run_subprocess
from docintel.core.temp_files import managed_temp_dir
from docintel.ocr.tesseract_engine import TesseractEngine, resolve_tesseract_cmd

logger = logging.getLogger(__name__)


class OCRMode(str, Enum):
    SKIP_TEXT = "skip_text"   # OCR only pages without text (safe for mixed PDFs)
    REDO_OCR = "redo_ocr"     # replace existing invisible OCR text; visible digital text is kept
    FORCE_OCR = "force_ocr"   # rasterise and OCR every page, replacing any text layer (loses vector quality)


# REDO_OCR does NOT use OCRmyPDF's --redo-ocr: with OCRmyPDF 16.7 and 16.13 that kept the old
# OCR layer and added a second one, duplicating every word (verified 2026-10-23). Instead the
# invisible text is removed first (text_layer.strip_invisible_text) and --skip-text then OCRs
# exactly the pages that no longer have text.
_MODE_FLAGS = {OCRMode.SKIP_TEXT: "--skip-text", OCRMode.REDO_OCR: "--skip-text", OCRMode.FORCE_OCR: "--force-ocr"}

# OCRmyPDF exit codes (see its documentation).
_EXIT_MESSAGES = {
    1: "invalid arguments",
    2: "the input file is not a valid PDF",
    3: "a required dependency is missing",
    4: "the output PDF is invalid",
    5: "file access error",
    6: "the PDF already contains text (choose 'redo OCR' or 'force OCR' mode)",
    7: "a child process failed",
    8: "the PDF is encrypted",
    9: "invalid OCRmyPDF configuration",
    10: "PDF/A conversion failed",
    15: "unexpected error",
    130: "interrupted",
}


@dataclass
class SearchablePdfStatus:
    available: bool
    reason: str = ""
    ghostscript: Optional[str] = None


def find_ghostscript() -> Optional[Path]:
    for name in ("gswin64c", "gswin32c", "gs"):
        found = shutil.which(name)
        if found:
            return Path(found)
    for env_var in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(env_var)
        if base:
            matches = sorted(Path(base).glob("gs/gs*/bin/gswin*c.exe"), reverse=True)
            if matches:
                return matches[0]
    return None


def check_searchable_pdf_support(settings: Settings) -> SearchablePdfStatus:
    if importlib.util.find_spec("ocrmypdf") is None:
        return SearchablePdfStatus(False, "Python package 'ocrmypdf' is not installed (requirements-optional.txt).")
    if resolve_tesseract_cmd(settings.tesseract_cmd) is None:
        return SearchablePdfStatus(False, "Tesseract executable not found.")
    gs = find_ghostscript()
    if gs is None:
        return SearchablePdfStatus(False, "Ghostscript (gswin64c.exe) not found; install it and add its bin folder to PATH.")
    return SearchablePdfStatus(True, ghostscript=str(gs))


def create_searchable_pdf(
    data: bytes,
    settings: Settings,
    *,
    languages: Sequence[str] = ("eng",),
    mode: OCRMode = OCRMode.SKIP_TEXT,
    deskew: bool = False,
    rotate_pages: bool = False,
    timeout_seconds: Optional[float] = None,
) -> bytes:
    """Return a copy of the PDF with an invisible OCR text layer added."""
    status = check_searchable_pdf_support(settings)
    if not status.available:
        raise OptionalDependencyMissingError(f"Searchable PDF export unavailable: {status.reason}")
    engine = TesseractEngine(settings)
    engine.validate_languages(list(languages))  # also rejects malformed language codes

    tesseract_dir = str(resolve_tesseract_cmd(settings.tesseract_cmd).parent)  # type: ignore[union-attr]
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([tesseract_dir, str(Path(status.ghostscript).parent), env.get("PATH", "")])  # type: ignore[arg-type]
    if settings.tessdata_dir:
        env["TESSDATA_PREFIX"] = str(settings.tessdata_dir)

    if mode is OCRMode.REDO_OCR:
        from docintel.documents.text_layer import strip_invisible_text  # needs pikepdf (optional)

        data, report = strip_invisible_text(data)
        logger.info("Redo OCR: previous text layer removed",
                    extra={"pages": report.pages_changed, "objects": report.text_objects_removed,
                           "mixed_kept": report.mixed_text_objects_kept})

    with managed_temp_dir(settings) as tmp:
        src, dst = tmp / "input.pdf", tmp / "output.pdf"
        src.write_bytes(data)
        args = [
            sys.executable, "-m", "ocrmypdf",
            "--output-type", "pdf",
            "--jobs", str(settings.max_workers),
            "-l", "+".join(languages),
            _MODE_FLAGS[mode],
        ]
        if deskew:
            args.append("--deskew")
        if rotate_pages:
            args.append("--rotate-pages")
        args += [str(src), str(dst)]
        proc = run_subprocess(args, timeout=timeout_seconds or settings.document_timeout_seconds, cwd=tmp, env=env)
        if proc.returncode == 8:
            raise EncryptedDocumentError("OCRmyPDF cannot process encrypted PDFs.")
        if proc.returncode != 0 or not dst.is_file():
            reason = _EXIT_MESSAGES.get(proc.returncode, f"exit code {proc.returncode}")
            logger.warning("OCRmyPDF failed", extra={"exit_code": proc.returncode})
            raise PDFProcessingError(f"Searchable PDF creation failed: {reason}.")
        output = dst.read_bytes()
    logger.info("Searchable PDF created", extra={"input_bytes": len(data), "output_bytes": len(output)})
    return output
