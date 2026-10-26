"""Grayscale, scaling, thresholding, contrast, sharpening and border-cropping operations.

All functions take and return uint8 NumPy arrays (RGB or grayscale) and never modify
their input in place.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def to_grayscale(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr.copy()
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def estimate_text_height(arr: np.ndarray) -> float | None:
    """Median height (px) of character-like connected components, or None if unclear.

    Image size is a poor proxy for text size (a wide receipt strip is "small" but its text
    may be large), so upscaling decisions are based on measured glyph height instead.
    """
    gray = to_grayscale(arr)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    height, width = mask.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    heights = [
        int(h)
        for _, _, w, h, area in stats[1:count]
        if area >= 4 and 3 <= h <= height * 0.2 and w <= width * 0.3
    ]
    if len(heights) < 10:
        return None
    return float(np.median(heights))


def compute_upscale_factor(
    width: int, height: int, text_height: float, target_height: float, max_scale: float, max_pixels: int
) -> float:
    """Return a factor (>= 1) that brings ``text_height`` up to ``target_height``.

    Capped by ``max_scale`` and by the pixel budget so upscaling never exceeds memory limits.
    """
    if text_height <= 0 or text_height >= target_height:
        return 1.0
    factor = min(target_height / text_height, max_scale)
    budget = math.sqrt(max_pixels / float(width * height))
    return max(1.0, min(factor, budget))


def resize(arr: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-6:
        return arr.copy()
    interpolation = cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA
    return cv2.resize(arr, None, fx=factor, fy=factor, interpolation=interpolation)


def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    """Global Otsu binarization; best for evenly lit images with bimodal histograms."""
    gray = to_grayscale(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def adaptive_threshold(gray: np.ndarray, block_size: int = 31, c: int = 15) -> np.ndarray:
    """Local Gaussian thresholding; handles shadows and uneven illumination."""
    gray = to_grayscale(gray)
    block_size = max(3, block_size | 1)  # must be odd and >= 3
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)


def enhance_contrast(arr: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """CLAHE contrast enhancement (applied to luminance for colour images)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    if arr.ndim == 2:
        return clahe.apply(arr)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def blur_score(arr: np.ndarray) -> float:
    """Variance of the Laplacian; low values (< ~100) indicate a blurry image."""
    return float(cv2.Laplacian(to_grayscale(arr), cv2.CV_64F).var())


def sharpen(arr: np.ndarray, amount: float = 1.0, radius: float = 1.0) -> np.ndarray:
    """Unsharp masking."""
    blurred = cv2.GaussianBlur(arr, (0, 0), radius)
    return cv2.addWeighted(arr, 1 + amount, blurred, -amount, 0)


def find_content_box(arr: np.ndarray, margin: int = 10) -> tuple[int, int, int, int] | None:
    """Return (x, y, w, h) enclosing the content, ignoring large dark scan borders.

    Connected components that touch the image edge and span more than half of a side are
    treated as scanner borders rather than content.
    """
    gray = to_grayscale(arr)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    height, width = mask.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        touches_edge = x == 0 or y == 0 or x + w >= width or y + h >= height
        is_border = touches_edge and (w > width * 0.5 or h > height * 0.5)
        keep[label] = area >= 3 and not is_border
    if not keep.any():
        return None
    boxes = stats[keep]
    x0 = max(0, int(boxes[:, 0].min()) - margin)
    y0 = max(0, int(boxes[:, 1].min()) - margin)
    x1 = min(width, int((boxes[:, 0] + boxes[:, 2]).max()) + margin)
    y1 = min(height, int((boxes[:, 1] + boxes[:, 3]).max()) + margin)
    return x0, y0, x1 - x0, y1 - y0


def crop_borders(arr: np.ndarray, margin: int = 10, min_gain: float = 0.02) -> tuple[np.ndarray, bool]:
    """Crop to the content box when doing so removes at least ``min_gain`` of the area."""
    box = find_content_box(arr, margin)
    if box is None:
        return arr.copy(), False
    x, y, w, h = box
    total = arr.shape[0] * arr.shape[1]
    if w * h > total * (1 - min_gain):
        return arr.copy(), False
    return arr[y : y + h, x : x + w].copy(), True


def flatten_illumination(arr: np.ndarray, text_height: float | None = None) -> tuple[np.ndarray, float]:
    """Divide the image by its estimated background to remove shadows and uneven lighting.

    The background is estimated by removing text with a max filter sized from the glyph height,
    then smoothing. Returns (flattened grayscale image, background variation before flattening).
    """
    gray = to_grayscale(arr)
    th = text_height or estimate_text_height(gray) or 12.0
    k = int(max(15, 2.5 * th)) | 1
    background = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    background = cv2.medianBlur(background, min(255, k))
    lo, hi = np.percentile(background, 5), np.percentile(background, 95)
    variation = float((hi - lo) / max(float(hi), 1.0))
    flat = cv2.divide(gray, np.maximum(background, 1), scale=255)
    return flat, variation
