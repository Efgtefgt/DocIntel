"""Temporary directory management.

Intermediate files (e.g. PDFs handed to Camelot or OCRmyPDF) live in uniquely named
temporary directories that are removed as soon as processing finishes.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from docintel.config.settings import Settings

logger = logging.getLogger(__name__)

TEMP_PREFIX = "docintel_"


def _temp_root(settings: Settings) -> Path:
    root = settings.temp_dir or Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    return root


@contextmanager
def managed_temp_dir(settings: Settings) -> Iterator[Path]:
    """Yield a fresh temporary directory that is always deleted afterwards."""
    path = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX, dir=_temp_root(settings)))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        if path.exists():
            logger.warning("Temporary directory could not be fully removed", extra={"path": str(path)})


def cleanup_stale_temp_dirs(settings: Settings, older_than_seconds: float = 3600) -> int:
    """Remove leftover DocIntel temp directories (e.g. after a crash). Returns the count removed."""
    root = _temp_root(settings)
    cutoff = time.time() - older_than_seconds
    removed = 0
    for entry in root.glob(f"{TEMP_PREFIX}*"):
        try:
            if entry.is_dir() and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry)
                removed += 1
        except OSError as exc:
            logger.warning("Could not remove stale temp dir", extra={"reason": str(exc)})
    return removed


def count_temp_dirs(settings: Settings) -> int:
    """Return the number of DocIntel temp directories currently present."""
    return sum(1 for entry in _temp_root(settings).glob(f"{TEMP_PREFIX}*") if entry.is_dir())
