"""File type detection from content signatures ("magic bytes").

File extensions are only used to raise warnings when they disagree with the content.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath
from typing import Optional

from docintel.core.result_models import FileKind


@dataclass(frozen=True)
class DetectedType:
    kind: FileKind
    format: str        # e.g. "PNG", "JPEG", "PDF"
    mime_type: str
    extensions: tuple[str, ...]


PNG = DetectedType(FileKind.IMAGE, "PNG", "image/png", (".png",))
JPEG = DetectedType(FileKind.IMAGE, "JPEG", "image/jpeg", (".jpg", ".jpeg", ".jpe"))
TIFF = DetectedType(FileKind.IMAGE, "TIFF", "image/tiff", (".tif", ".tiff"))
BMP = DetectedType(FileKind.IMAGE, "BMP", "image/bmp", (".bmp", ".dib"))
WEBP = DetectedType(FileKind.IMAGE, "WEBP", "image/webp", (".webp",))
PDF = DetectedType(FileKind.PDF, "PDF", "application/pdf", (".pdf",))

SUPPORTED_TYPES = (PNG, JPEG, TIFF, BMP, WEBP, PDF)
SUPPORTED_EXTENSIONS = tuple(ext.lstrip(".") for t in SUPPORTED_TYPES for ext in t.extensions)


def detect_type(data: bytes) -> Optional[DetectedType]:
    """Identify a supported file type from its leading bytes, or return ``None``."""
    head = data[:16]
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return PNG
    if head.startswith(b"\xff\xd8\xff"):
        return JPEG
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return TIFF
    if head.startswith(b"BM") and len(data) > 26:
        return BMP
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return WEBP
    # The PDF specification allows junk before the header; readers accept it within 1 KB.
    if b"%PDF-" in data[:1024]:
        return PDF
    return None


def extension_warning(filename: str, detected: DetectedType) -> Optional[str]:
    """Return a warning if the filename's extension does not match the detected content."""
    suffix = PurePath(filename).suffix.lower()
    if suffix and suffix not in detected.extensions:
        return (
            f"File extension '{suffix}' does not match its content ({detected.format}); "
            "the content type was used."
        )
    return None
