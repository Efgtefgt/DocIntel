"""Upload validation, type detection and path-safety tests."""

from __future__ import annotations

import io
import struct
import zlib
from pathlib import Path

import fitz
import pytest
from PIL import Image

from docintel.core.exceptions import (
    CorruptFileError,
    EmptyFileError,
    EncryptedDocumentError,
    FileTooLargeError,
    ImageDimensionError,
    PageLimitError,
    UnsafePathError,
    UnsupportedFormatError,
)
from docintel.core.file_detector import detect_type
from docintel.core.file_validator import validate_upload
from docintel.core.result_models import FileKind
from docintel.core.security import display_path, ensure_within, safe_filename
from docintel.evaluation import synthetic_data as synth


@pytest.mark.parametrize("fmt, name", [("PNG", "a.png"), ("JPEG", "a.jpg"), ("BMP", "a.bmp"), ("TIFF", "a.tiff"), ("WEBP", "a.webp")])
def test_valid_image_formats(settings, text_image, fmt, name):
    data = synth.image_bytes(text_image, fmt)
    result = validate_upload(data, name, settings)
    assert result.info.kind is FileKind.IMAGE
    assert result.info.format == fmt
    assert (result.info.width, result.info.height) == text_image.size
    assert result.info.warnings == []


def test_multipage_tiff_counts_frames(settings, text_image):
    buffer = io.BytesIO()
    text_image.save(buffer, format="TIFF", save_all=True, append_images=[text_image.rotate(180)])
    assert validate_upload(buffer.getvalue(), "scan.tif", settings).info.page_count == 2


def test_extension_mismatch_is_warned_not_trusted(settings, text_image):
    data = synth.image_bytes(text_image, "JPEG")
    result = validate_upload(data, "photo.png", settings)
    assert result.info.format == "JPEG"
    assert "does not match" in result.info.warnings[0]


def test_unsupported_content_rejected_even_with_image_extension(settings):
    with pytest.raises(UnsupportedFormatError):
        validate_upload(b"this is plain text, not an image" * 10, "fake.png", settings)


def test_empty_file_rejected(settings):
    with pytest.raises(EmptyFileError):
        validate_upload(b"", "empty.png", settings)


def test_corrupt_png_rejected(settings, png_bytes):
    with pytest.raises(CorruptFileError):
        validate_upload(png_bytes[: len(png_bytes) // 2], "broken.png", settings)


def test_large_file_rejected(settings, png_bytes):
    small_limit = settings.model_copy(update={"max_file_size_mb": 0.001})
    with pytest.raises(FileTooLargeError):
        validate_upload(png_bytes + b"\0" * 5000, "big.png", small_limit)


def test_too_small_image_rejected(settings):
    data = synth.image_bytes(Image.new("RGB", (4, 4), "white"))
    with pytest.raises(ImageDimensionError):
        validate_upload(data, "tiny.png", settings)


def _png_with_header_size(width: int, height: int) -> bytes:
    """A PNG whose IHDR claims huge dimensions (decompression-bomb style)."""
    data = bytearray(synth.image_bytes(Image.new("L", (10, 10), 255)))
    ihdr = struct.pack(">II", width, height) + bytes(data[24:29])
    data[16:29] = ihdr
    data[29:33] = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF)
    return bytes(data)


def test_decompression_bomb_rejected_before_decoding(settings):
    with pytest.raises(ImageDimensionError):
        validate_upload(_png_with_header_size(60_000, 60_000), "bomb.png", settings)


def test_pixel_limit_enforced(settings, png_bytes):
    strict = settings.model_copy(update={"max_image_pixels": 1000})
    with pytest.raises(ImageDimensionError):
        validate_upload(png_bytes, "a.png", strict)


def test_valid_pdf(settings):
    data = synth.make_digital_pdf(["page one", "page two"])
    info = validate_upload(data, "doc.pdf", settings).info
    assert info.kind is FileKind.PDF and info.page_count == 2


def test_corrupt_pdf_rejected(settings):
    with pytest.raises(CorruptFileError):
        validate_upload(b"%PDF-1.7\n" + b"\x00garbage" * 50, "bad.pdf", settings)


def test_encrypted_pdf_rejected(settings):
    doc = fitz.open()
    doc.new_page()
    data = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    with pytest.raises(EncryptedDocumentError):
        validate_upload(data, "locked.pdf", settings)


def test_pdf_page_limit(settings):
    data = synth.make_digital_pdf(["a", "b", "c"])
    with pytest.raises(PageLimitError):
        validate_upload(data, "long.pdf", settings.model_copy(update={"max_pdf_pages": 2}))


def test_detect_type_signatures():
    assert detect_type(b"%PDF-1.4 rest") is not None
    assert detect_type(b"GIF89a....") is None


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("../../etc/passwd", "passwd"),
        ("..\\..\\Windows\\system.ini", "system.ini"),
        ("CON.txt", "_CON.txt"),
        ("report<>:|?.pdf", "report_.pdf"),
        ("", "document"),
        ("..", "document"),
    ],
)
def test_safe_filename(raw, expected):
    assert safe_filename(raw) == expected


def test_ensure_within_blocks_traversal(tmp_path: Path):
    assert ensure_within(tmp_path, Path("ok.txt")) == (tmp_path / "ok.txt").resolve()
    with pytest.raises(UnsafePathError):
        ensure_within(tmp_path, Path("../escape.txt"))


def test_display_path_hides_the_user_folder(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("USERPROFILE", r"C:\Users\Jane.Doe")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Jane.Doe\AppData\Local")
    monkeypatch.setenv("APPDATA", r"C:\Users\Jane.Doe\AppData\Roaming")
    assert display_path(r"C:\Users\Jane.Doe\AppData\Local\DocIntel\logs") == r"%LOCALAPPDATA%\DocIntel\logs"
    assert display_path(Path(r"C:\Users\jane.doe\Documents\out.txt")) == r"%USERPROFILE%\Documents\out.txt"
    assert display_path(r"Saved C:\Users\Jane.Doe\a and C:\Users\Jane.Doe\b") == r"Saved %USERPROFILE%\a and %USERPROFILE%\b"
    assert display_path(r"C:\Program Files\Tesseract-OCR\tesseract.exe") == r"C:\Program Files\Tesseract-OCR\tesseract.exe"
