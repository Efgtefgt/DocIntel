"""Skew estimation (projection-profile search) and rotation helpers."""

from __future__ import annotations

import cv2
import numpy as np

from docintel.preprocessing.enhancement import to_grayscale

_ANALYSIS_MAX_SIDE = 1000


def _profile_score(binary: np.ndarray, angle: float) -> float:
    h, w = binary.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(binary, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    rows = rotated.sum(axis=1, dtype=np.float64)
    # Text lines aligned with the x-axis produce sharp peaks/valleys in the row profile.
    return float(np.sum(np.diff(rows) ** 2))


def estimate_skew_angle(arr: np.ndarray, max_angle: float = 15.0) -> float:
    """Estimate the rotation (degrees, counter-clockwise positive) that straightens text lines.

    Uses a coarse-to-fine projection-profile search on a downscaled binary image, which is
    robust to noise and cheap enough for CPU-only machines. Returns 0.0 for blank images.
    """
    gray = to_grayscale(arr)
    scale = min(1.0, _ANALYSIS_MAX_SIDE / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if binary.sum() < 50:
        return 0.0

    def search(center: float, span: float, step: float) -> float:
        angles = np.arange(center - span, center + span + step / 2, step)
        scores = [_profile_score(binary, float(a)) for a in angles]
        return float(angles[int(np.argmax(scores))])

    coarse = search(0.0, max_angle, 1.0)
    fine = search(coarse, 1.0, 0.1)
    return round(fine, 2)


def rotate(arr: np.ndarray, angle: float, background: int = 255) -> np.ndarray:
    """Rotate counter-clockwise by ``angle`` degrees, expanding the canvas to avoid cropping."""
    h, w = arr.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2 - w / 2
    matrix[1, 2] += new_h / 2 - h / 2
    border = background if arr.ndim == 2 else (background,) * 3
    return cv2.warpAffine(arr, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderValue=border)


def rotate_right_angle(arr: np.ndarray, clockwise_degrees: int) -> np.ndarray:
    """Lossless rotation by a multiple of 90 degrees (clockwise)."""
    turns = (clockwise_degrees // 90) % 4
    codes = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}
    return arr.copy() if turns == 0 else cv2.rotate(arr, codes[turns])


def edge_background(arr: np.ndarray) -> int:
    """Median brightness of the image border, used to fill corners exposed by rotation."""
    gray = to_grayscale(arr)
    edges = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    return int(np.median(edges))
