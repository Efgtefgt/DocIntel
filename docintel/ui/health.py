"""Dependency health checks for the dashboard and About page.

Checks are real but lightweight: executables are run once with ``--version`` (cached), Python
packages are located without importing heavy modules, and OCR models are never loaded - for
PaddleOCR only the presence of the downloaded model folders is checked.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from docintel.config.settings import Settings
from docintel.core.exceptions import ProcessingTimeoutError
from docintel.core.security import run_subprocess
from docintel.documents.searchable_pdf import find_ghostscript
from docintel.ocr.base_engine import EngineStatus

Status = Literal["ready", "optional", "setup", "unavailable", "error"]
STATUS_LABELS: dict[str, str] = {
    "ready": "Ready",
    "optional": "Not installed",
    "setup": "Needs setup",
    "unavailable": "Unavailable",
    "error": "Error",
}


@dataclass(frozen=True)
class HealthItem:
    name: str
    status: Status
    detail: str
    label: str = ""

    @property
    def display_label(self) -> str:
        return self.label or STATUS_LABELS[self.status]


@lru_cache(maxsize=4)
def ghostscript_version(path: str) -> str | None:
    try:
        proc = run_subprocess([path, "--version"], timeout=10)
    except (ProcessingTimeoutError, OSError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else None


def _pikepdf_compatible() -> tuple[bool, str]:
    """OCRmyPDF 16.7 calls pikepdf.Pdf.check(), which pikepdf 10 removed."""
    if importlib.util.find_spec("pikepdf") is None:
        return False, "pikepdf not installed"
    import pikepdf  # small C-extension import, no models

    if hasattr(pikepdf.Pdf, "check"):
        return True, f"pikepdf {pikepdf.__version__}"
    return False, f"pikepdf {pikepdf.__version__} is incompatible with OCRmyPDF 16.7 (needs < 10)"


def paddle_models_present(home: Path | None = None) -> bool:
    """True when PaddleOCR's detection and recognition models have been downloaded."""
    root = (home or Path.home()) / ".paddleocr" / "whl"
    return any((root / "det").glob("**/*.pdiparams")) and any((root / "rec").glob("**/*.pdiparams"))


def tesseract_item(status: EngineStatus, has_osd: bool) -> HealthItem:
    if not status.available:
        return HealthItem("Tesseract OCR", "unavailable", status.reason)
    langs = ", ".join(status.languages[:8]) + (" ..." if len(status.languages) > 8 else "")
    extra = "" if has_osd else " · orientation data (osd) missing"
    return HealthItem("Tesseract OCR", "ready", f"{status.version} · languages: {langs}{extra}")


def searchable_pdf_item(settings: Settings, tesseract_ready: bool) -> HealthItem:
    name = "Searchable PDF"
    if importlib.util.find_spec("ocrmypdf") is None:
        return HealthItem(name, "optional", "OCRmyPDF not installed (requirements-optional.txt)")
    gs = find_ghostscript()
    if gs is None:
        return HealthItem(name, "unavailable", "Ghostscript not found; install it from the Artifex releases page")
    gs_version = ghostscript_version(str(gs))
    if gs_version is None:
        return HealthItem(name, "error", f"Ghostscript at {gs} did not run")
    ok, pike = _pikepdf_compatible()
    if not ok:
        return HealthItem(name, "error", pike)
    if not tesseract_ready:
        return HealthItem(name, "unavailable", "Tesseract is required")
    return HealthItem(name, "ready", f"OCRmyPDF · Ghostscript {gs_version} · {pike}")


def paddle_item(settings: Settings, status: EngineStatus | None) -> HealthItem:
    name = "PaddleOCR"
    if importlib.util.find_spec("paddleocr") is None:
        return HealthItem(name, "optional", "Deep-learning engine; see README section 11 to install")
    if not settings.enable_paddle:
        return HealthItem(name, "optional", "Installed but disabled (set DOCINTEL_ENABLE_PADDLE=true)", label="Disabled")
    if status is not None and not status.available:
        return HealthItem(name, "unavailable", status.reason)
    if not paddle_models_present():
        return HealthItem(name, "setup", "Models download on first use (needs internet once)")
    return HealthItem(name, "ready", f"PaddleOCR {status.version if status else ''} · models downloaded".strip())


def tables_item() -> HealthItem:
    camelot = importlib.util.find_spec("camelot") is not None
    detail = "PyMuPDF table finder" + (" · Camelot lattice/stream" if camelot else " · Camelot not installed (optional)")
    return HealthItem("Table extraction", "ready", detail)


def collect(settings: Settings, statuses: list[EngineStatus], has_osd: bool) -> list[HealthItem]:
    by_name = {s.name: s for s in statuses}
    tess = by_name.get("tesseract")
    items = [tesseract_item(tess, has_osd) if tess else HealthItem("Tesseract OCR", "unavailable", "engine not registered")]
    items.append(searchable_pdf_item(settings, bool(tess and tess.available)))
    items.append(paddle_item(settings, by_name.get("paddle")))
    items.append(tables_item())
    return items
