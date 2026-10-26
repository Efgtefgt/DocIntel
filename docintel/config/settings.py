"""Typed application settings loaded from environment variables or a ``.env`` file.

All environment variables use the ``DOCINTEL_`` prefix, e.g. ``DOCINTEL_TESSERACT_CMD``.
Settings objects are immutable; the UI creates modified copies with ``model_copy``
instead of mutating global state.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_batch_dir() -> Path:
    """Per-user folder outside the program/project folder, so batches survive app updates."""
    local = os.environ.get("LOCALAPPDATA")
    return Path(local) / "DocIntel" / "batches" if local else PROJECT_ROOT / "outputs" / "batches"


class Settings(BaseSettings):
    """Runtime configuration. Defaults are tuned for a CPU-only laptop with 8 GB RAM."""

    model_config = SettingsConfigDict(
        env_prefix="DOCINTEL_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- OCR -------------------------------------------------------------------------
    default_engine: str = "tesseract"
    default_languages: str = "eng"
    tesseract_cmd: Optional[Path] = None
    tessdata_dir: Optional[Path] = None
    low_confidence_threshold: float = Field(60.0, ge=0, le=100)

    # --- Limits ----------------------------------------------------------------------
    max_file_size_mb: float = Field(25.0, gt=0, le=500)
    max_image_pixels: int = Field(40_000_000, gt=0)
    max_image_dimension: int = Field(12_000, gt=0)
    min_image_dimension: int = Field(8, gt=0)
    max_pdf_pages: int = Field(200, gt=0)
    max_tiff_frames: int = Field(50, gt=0)

    # --- Performance -----------------------------------------------------------------
    pdf_render_dpi: int = Field(200, ge=72, le=600)
    ocr_timeout_seconds: float = Field(60.0, gt=0)
    document_timeout_seconds: float = Field(900.0, gt=0)
    max_workers: int = Field(1, ge=1, le=4)
    max_preprocessing_variants: int = Field(3, ge=1, le=6)
    max_ocr_attempts: int = Field(3, ge=1, le=4)  # smart preset: extra attempts only when a result looks weak

    # --- Files -----------------------------------------------------------------------
    temp_dir: Optional[Path] = None
    export_dir: Path = PROJECT_ROOT / "outputs"
    templates_dir: Path = PROJECT_ROOT / "data" / "templates"

    # --- Batch processing --------------------------------------------------------------
    batch_dir: Path = Field(default_factory=_default_batch_dir)  # queue state, inputs and results
    batch_max_files: int = Field(500, ge=1, le=10_000)
    batch_min_free_memory_mb: int = Field(600, ge=0)  # wait (up to a minute) before a file below this

    # --- Logging ---------------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: Path = PROJECT_ROOT / "logs"
    log_to_file: bool = True

    # --- User interface --------------------------------------------------------------
    ui_theme: Literal["light", "dark"] = "light"   # initial theme; switchable in the sidebar

    # --- Optional features -----------------------------------------------------------
    enable_paddle: bool = False
    api_max_request_mb: float = Field(25.0, gt=0, le=500)

    @field_validator("batch_dir", mode="before")
    @classmethod
    def _empty_batch_dir(cls, value: object) -> object:
        return _default_batch_dir() if isinstance(value, str) and not value.strip() else value

    @field_validator("tesseract_cmd", "tessdata_dir", "temp_dir", mode="before")
    @classmethod
    def _empty_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def max_file_size_bytes(self) -> int:
        return int(self.max_file_size_mb * 1024 * 1024)

    @property
    def api_max_request_bytes(self) -> int:
        return int(self.api_max_request_mb * 1024 * 1024)

    @property
    def language_list(self) -> list[str]:
        return [part for part in self.default_languages.replace(",", "+").split("+") if part]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide default settings (read once from the environment)."""
    return Settings()
