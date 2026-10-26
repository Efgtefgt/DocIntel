"""Decode validated image bytes into normalized Pillow images and NumPy arrays."""

from __future__ import annotations

import io
from typing import Iterable, Iterator

import numpy as np
from PIL import Image, ImageOps, ImageSequence


def _to_ocr_mode(img: Image.Image) -> Image.Image:
    """Convert any Pillow mode to RGB or L, flattening transparency onto white.

    Transparent PNGs often contain black text on a transparent background; dropping the
    alpha channel would produce black-on-black, so the image is composited onto white.
    """
    if img.mode in ("RGB", "L"):
        return img.copy()
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(img, dtype=np.float64)
        span = float(arr.max() - arr.min()) or 1.0
        return Image.fromarray(((arr - arr.min()) / span * 255).astype(np.uint8))
    if img.mode == "1":
        return img.convert("L")
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return img.convert("RGB")


def iter_frames(data: bytes, page_numbers: Iterable[int] | None = None, max_frames: int = 50) -> Iterator[tuple[int, Image.Image]]:
    """Yield ``(page_number, image)`` for the selected 1-based frames, one at a time.

    Multi-page TIFFs are decoded lazily so only one frame is held in memory. EXIF
    orientation is honoured.
    """
    wanted = set(page_numbers) if page_numbers is not None else None
    with Image.open(io.BytesIO(data)) as img:
        for index, frame in enumerate(ImageSequence.Iterator(img)):
            number = index + 1
            if number > max_frames:
                break
            if wanted is not None and number not in wanted:
                continue
            yield number, _to_ocr_mode(ImageOps.exif_transpose(frame))


def load_frames(data: bytes, max_frames: int = 50) -> list[Image.Image]:
    """Decode all frames (use :func:`iter_frames` for large multi-page files)."""
    return [frame for _, frame in iter_frames(data, None, max_frames)]


def load_first_frame(data: bytes) -> Image.Image:
    return next(iter_frames(data, [1], 1))[1]


def pil_to_array(img: Image.Image) -> np.ndarray:
    """Return an RGB (H, W, 3) or grayscale (H, W) uint8 array."""
    return np.asarray(_to_ocr_mode(img), dtype=np.uint8).copy()


def array_to_pil(arr: np.ndarray) -> Image.Image:
    """Mode is inferred from shape: 2-D uint8 -> "L", (H, W, 3) uint8 -> "RGB"."""
    return Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8))


def image_to_png_bytes(img: Image.Image) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()
