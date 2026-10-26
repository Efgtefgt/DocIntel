"""Render PDF pages to Pillow images within a pixel budget."""

from __future__ import annotations

import math

import fitz  # PyMuPDF
from PIL import Image


def effective_dpi(page: fitz.Page, dpi: int, max_pixels: int) -> int:
    """Lower ``dpi`` if rendering at it would exceed ``max_pixels`` (e.g. poster-sized pages)."""
    width_in, height_in = page.rect.width / 72, page.rect.height / 72
    pixels = width_in * dpi * height_in * dpi
    if pixels <= max_pixels:
        return dpi
    return max(36, int(math.sqrt(max_pixels / (width_in * height_in))))


def render_page(page: fitz.Page, dpi: int = 200, max_pixels: int = 40_000_000) -> Image.Image:
    """Render one page as an RGB image."""
    zoom = effective_dpi(page, dpi, max_pixels) / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def render_page_from_bytes(data: bytes, page_number: int, dpi: int = 200, max_pixels: int = 40_000_000) -> Image.Image:
    """Render a 1-based page from PDF bytes (used for previews)."""
    from docintel.documents.pdf_reader import open_pdf

    with open_pdf(data) as doc:
        return render_page(doc[page_number - 1], dpi, max_pixels)
