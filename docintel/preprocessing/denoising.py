"""Noise estimation and reduction."""

from __future__ import annotations

import cv2
import numpy as np

from docintel.preprocessing.enhancement import to_grayscale


def estimate_noise(arr: np.ndarray) -> float:
    """Mean absolute difference between the image and its median-filtered version.

    Clean scans typically score below ~3; salt-and-pepper or sensor noise scores higher.
    """
    gray = to_grayscale(arr)
    median = cv2.medianBlur(gray, 3)
    return float(np.mean(cv2.absdiff(gray, median)))


def denoise(arr: np.ndarray, strength: int = 10) -> np.ndarray:
    """Non-local-means denoising (grayscale or colour)."""
    if arr.ndim == 2:
        return cv2.fastNlMeansDenoising(arr, None, h=strength, templateWindowSize=7, searchWindowSize=21)
    return cv2.fastNlMeansDenoisingColored(arr, None, strength, strength, 7, 21)
