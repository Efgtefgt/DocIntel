"""Filesystem and subprocess safety helpers."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import Mapping, Sequence

from docintel.core.exceptions import ProcessingTimeoutError, UnsafePathError

logger = logging.getLogger(__name__)

_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._\- ]+")
# Windows: start console tools (Tesseract, OCRmyPDF) without a console window, so none flashes
# up when the app runs windowless (installed version, pythonw.exe).
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def safe_filename(name: str, *, default: str = "document", max_length: int = 120) -> str:
    """Reduce an untrusted filename to a safe basename.

    Directory components, control characters and characters that are invalid on Windows
    are removed. The result never contains path separators and is never a reserved device
    name such as ``CON``.
    """
    # Strip any directory part, using both separators regardless of platform.
    base = re.split(r"[\\/]", name or "")[-1]
    base = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode("ascii")
    base = _UNSAFE_CHARS.sub("_", base).strip(" .")
    base = re.sub(r"_+", "_", base)
    if not base or base in {".", ".."}:
        base = default
    stem, dot, suffix = base.rpartition(".")
    if not dot:
        stem, suffix = base, ""
    if stem.upper() in _WINDOWS_RESERVED or base.upper() in _WINDOWS_RESERVED:
        stem = f"_{stem or base}"
    suffix = suffix[:10]
    stem = stem[: max(1, max_length - len(suffix) - 1)]
    return f"{stem}.{suffix}" if suffix else stem


def display_path(value: object) -> str:
    """Text for the screen with the user's own folders replaced by their variable names.

    ``C:\\Users\\<name>\\AppData\\Local\\...`` becomes ``%LOCALAPPDATA%\\...`` and the rest of the
    profile becomes ``%USERPROFILE%``, so pages and screenshots never show the Windows user name.
    Works on any text (error messages too); every occurrence is replaced, ignoring case.
    """
    text = str(value)
    for var in ("LOCALAPPDATA", "APPDATA", "USERPROFILE"):
        folder = os.environ.get(var) or (str(Path.home()) if var == "USERPROFILE" else "")
        if folder and len(folder) > 3:  # never rewrite a bare drive such as C:\
            for form in _path_forms(folder):
                text = re.sub(re.escape(form), lambda _m, v=var: f"%{v}%", text, flags=re.IGNORECASE)
    return text


def _path_forms(folder: str) -> list[str]:
    """``folder`` as written, in long form and in 8.3 short form (C:\\Users\\JANE~1.DOE), longest first."""
    forms = {folder.rstrip("\\/")}
    if os.name == "nt":
        import ctypes

        for convert in (ctypes.windll.kernel32.GetLongPathNameW, ctypes.windll.kernel32.GetShortPathNameW):
            buffer = ctypes.create_unicode_buffer(1024)
            if 0 < convert(folder, buffer, 1024) < 1024:
                forms.add(buffer.value.rstrip("\\/"))
    return sorted((f for f in forms if len(f) > 3), key=len, reverse=True)


def ensure_within(base_dir: Path, candidate: Path) -> Path:
    """Resolve ``candidate`` and verify it is located inside ``base_dir``.

    Raises:
        UnsafePathError: if the resolved path escapes ``base_dir`` (path traversal).
    """
    base = base_dir.resolve()
    resolved = (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if resolved != base and base not in resolved.parents:
        raise UnsafePathError("The requested path is outside the permitted directory.")
    return resolved


def run_subprocess(
    args: Sequence[str],
    *,
    timeout: float,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an external program without a shell.

    Arguments are passed as a list, so untrusted values are never interpreted by a shell.
    Output is captured (not logged) because it may contain document text.

    Raises:
        ProcessingTimeoutError: when the process exceeds ``timeout`` seconds.
    """
    if not args or not all(isinstance(a, str) for a in args):
        raise ValueError("Subprocess arguments must be a non-empty list of strings.")
    logger.debug("Running subprocess", extra={"program": Path(args[0]).name, "argc": len(args)})
    try:
        return subprocess.run(  # noqa: S603 - shell=False with a validated argument list
            list(args),
            shell=False,
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env is not None else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            creationflags=_NO_WINDOW,
        )
    except subprocess.TimeoutExpired as exc:
        raise ProcessingTimeoutError(
            f"External tool '{Path(args[0]).name}' exceeded the {timeout:.0f}s time limit."
        ) from exc
