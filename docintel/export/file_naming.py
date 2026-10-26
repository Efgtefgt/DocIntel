"""Predictable, safe export filenames and non-overwriting saves."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path, PurePath
from typing import Optional

from docintel.core.exceptions import ExportError
from docintel.core.security import ensure_within, safe_filename


def export_filename(source_name: str, suffix: str, extension: str, *, timestamp: Optional[datetime] = None) -> str:
    """Build ``<source-stem>_<suffix>_<YYYYmmdd-HHMMSS>.<ext>`` from an untrusted source name."""
    stem = PurePath(safe_filename(source_name)).stem or "document"
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d-%H%M%S")
    suffix = safe_filename(suffix, default="export")
    extension = safe_filename(extension.lstrip("."), default="txt")
    return f"{stem}_{suffix}_{stamp}.{extension}"


def save_export(data: bytes, filename: str, export_dir: Path) -> Path:
    """Write ``data`` into ``export_dir`` without overwriting existing files.

    If the name is taken, `` (1)``, `` (2)``... is appended. Files are created with
    exclusive mode so a concurrent writer cannot be clobbered either.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    name = safe_filename(filename)
    stem, suffix = PurePath(name).stem, PurePath(name).suffix
    for attempt in range(1000):
        candidate_name = name if attempt == 0 else f"{stem} ({attempt}){suffix}"
        target = ensure_within(export_dir, Path(candidate_name))
        try:
            with open(target, "xb") as handle:
                handle.write(data)
            return target
        except FileExistsError:
            continue
        except OSError as exc:
            raise ExportError(f"Could not write export file: {exc.strerror}.") from exc
    raise ExportError("Could not find a free export filename.")
