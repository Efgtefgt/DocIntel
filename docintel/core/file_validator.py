"""Upload validation: size, content type, image integrity, dimensions and PDF page limits."""

from __future__ import annotations

import io
import logging
import warnings
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

from docintel.config.settings import Settings
from docintel.core.exceptions import (
    CorruptFileError,
    EmptyFileError,
    EncryptedDocumentError,
    FileTooLargeError,
    ImageDimensionError,
    PageLimitError,
    UnsupportedFormatError,
)
from docintel.core.file_detector import SUPPORTED_TYPES, DetectedType, detect_type, extension_warning
from docintel.core.result_models import FileInfo, FileKind
from docintel.core.security import safe_filename

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidatedFile:
    """An upload that passed validation. ``data`` stays in memory; nothing is written to disk."""

    data: bytes
    info: FileInfo
    detected: DetectedType


def validate_upload(data: bytes, filename: str, settings: Settings) -> ValidatedFile:
    """Validate raw upload bytes and return file metadata.

    Raises:
        InputValidationError subclasses describing why the file was rejected.
    """
    name = safe_filename(filename)
    if not data:
        raise EmptyFileError(f"'{name}' is empty.")
    if len(data) > settings.max_file_size_bytes:
        raise FileTooLargeError(
            f"'{name}' is {len(data) / 1_048_576:.1f} MB; the limit is {settings.max_file_size_mb:.0f} MB."
        )
    detected = detect_type(data)
    if detected is None:
        supported = ", ".join(t.format for t in SUPPORTED_TYPES)
        raise UnsupportedFormatError(f"'{name}' is not a supported file type. Supported: {supported}.")

    file_warnings = [w for w in [extension_warning(filename, detected)] if w]
    if detected.kind is FileKind.PDF:
        info = _validate_pdf(data, name, detected, settings)
    else:
        info = _validate_image(data, name, detected, settings)
    info.warnings.extend(file_warnings)
    logger.info(
        "File validated",
        extra={"format": info.format, "size_bytes": info.size_bytes, "pages": info.page_count},
    )
    return ValidatedFile(data=data, info=info, detected=detected)


def check_image_dimensions(width: int, height: int, settings: Settings) -> None:
    """Reject images that are too small to OCR or large enough to exhaust memory."""
    if width < settings.min_image_dimension or height < settings.min_image_dimension:
        raise ImageDimensionError(
            f"Image is {width}x{height} px; the minimum is {settings.min_image_dimension} px per side."
        )
    if width > settings.max_image_dimension or height > settings.max_image_dimension:
        raise ImageDimensionError(
            f"Image is {width}x{height} px; the maximum is {settings.max_image_dimension} px per side."
        )
    if width * height > settings.max_image_pixels:
        raise ImageDimensionError(
            f"Image has {width * height:,} pixels; the limit is {settings.max_image_pixels:,} "
            "(protection against decompression bombs)."
        )


def _validate_image(data: bytes, name: str, detected: DetectedType, settings: Settings) -> FileInfo:
    try:
        with warnings.catch_warnings():
            # Treat Pillow's decompression-bomb warning as an error rather than continuing.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as img:
                # Header-only read: check dimensions before decoding any pixel data.
                width, height = img.size
                check_image_dimensions(width, height, settings)
                frames = getattr(img, "n_frames", 1)
                img.verify()
            with Image.open(io.BytesIO(data)) as img:
                img.load()  # verify() does not decode pixels; a full load catches truncated files
    except ImageDimensionError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageDimensionError(f"'{name}' exceeds the safe pixel limit.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise CorruptFileError(f"'{name}' could not be decoded; the image may be corrupt or truncated.") from exc

    if frames > settings.max_tiff_frames:
        raise PageLimitError(f"'{name}' has {frames} frames; the limit is {settings.max_tiff_frames}.")
    return FileInfo(
        filename=name,
        kind=FileKind.IMAGE,
        format=detected.format,
        mime_type=detected.mime_type,
        size_bytes=len(data),
        width=width,
        height=height,
        page_count=frames,
    )


def _validate_pdf(data: bytes, name: str, detected: DetectedType, settings: Settings) -> FileInfo:
    import fitz  # PyMuPDF

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except (fitz.FileDataError, RuntimeError, ValueError) as exc:
        raise CorruptFileError(f"'{name}' could not be opened as a PDF; it may be corrupt.") from exc
    with doc:
        if doc.needs_pass:
            raise EncryptedDocumentError(f"'{name}' is password-protected. Remove the password and retry.")
        page_count = doc.page_count
        if page_count == 0:
            raise CorruptFileError(f"'{name}' contains no readable pages.")
        if page_count > settings.max_pdf_pages:
            raise PageLimitError(
                f"'{name}' has {page_count} pages; the limit is {settings.max_pdf_pages}. "
                "Split the document or raise DOCINTEL_MAX_PDF_PAGES."
            )
        warnings_list = ["PDF was repaired while opening; results may be incomplete."] if doc.is_repaired else []
    return FileInfo(
        filename=name,
        kind=FileKind.PDF,
        format=detected.format,
        mime_type=detected.mime_type,
        size_bytes=len(data),
        page_count=page_count,
        warnings=warnings_list,
    )
