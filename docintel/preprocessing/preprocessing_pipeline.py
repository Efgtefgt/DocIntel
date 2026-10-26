"""Configurable preprocessing pipelines.

Steps always run in a fixed canonical order (orientation before cropping, deskew before
thresholding, ...) regardless of the order they were selected in. Several steps are
conditional: upscaling only happens for small images, deskew only for measurable skew,
and sharpening only for blurry images, so selecting a step does not blindly apply it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from PIL import Image

from docintel.core.result_models import PreprocessStepRecord
from docintel.preprocessing import denoising, deskew, enhancement
from docintel.preprocessing.image_loader import array_to_pil, pil_to_array

# Canonical execution order. Denoise runs before upscaling: noise is measured at native
# resolution (upscaling smooths it and hides it from the estimate) and is cheaper there.
STEP_ORDER = (
    "orientation",
    "crop_borders",
    "grayscale",
    "invert",
    "flatten_illumination",
    "denoise",
    "deskew",
    "upscale",
    "contrast",
    "sharpen",
    "adaptive_threshold",
    "otsu_threshold",
)

STEP_DESCRIPTIONS = {
    "orientation": "Correct 90/180/270 degree page rotation (needs Tesseract OSD data)",
    "crop_borders": "Crop dark scanner borders and empty margins",
    "grayscale": "Convert to grayscale",
    "invert": "Make light-on-dark text dark-on-light (only when the background is dark)",
    "flatten_illumination": "Remove shadows and uneven lighting by dividing out the background",
    "upscale": "Enlarge the image when measured character height is small",
    "deskew": "Straighten slightly rotated text lines",
    "denoise": "Reduce speckle and sensor noise",
    "contrast": "Improve local contrast (CLAHE)",
    "sharpen": "Sharpen blurry images (skipped for sharp images)",
    "adaptive_threshold": "Binarize with local thresholds (uneven lighting, shadows)",
    "otsu_threshold": "Binarize with a global Otsu threshold (evenly lit scans)",
}

PRESETS: dict[str, tuple[str, ...]] = {
    "smart": (),  # steps are chosen per page by analysis.smart_plan (see core/pipeline.py)
    "original": (),
    "auto": ("orientation", "grayscale", "upscale", "deskew", "denoise"),
    "grayscale": ("grayscale",),
    "grayscale_threshold": ("grayscale", "otsu_threshold"),
    "adaptive_threshold": ("grayscale", "adaptive_threshold"),
    "denoised": ("grayscale", "denoise"),
    "deskewed": ("grayscale", "deskew"),
    "enhanced_contrast": ("grayscale", "contrast"),
}

PRESET_DESCRIPTIONS = {
    "smart": "Recommended: analyse each page, choose steps automatically, retry weak results",
    "original": "No changes; OCR runs on the image as uploaded",
    "auto": "Grayscale, then upscale/deskew/denoise only when measurements show a need",
    "grayscale": "Grayscale only",
    "grayscale_threshold": "Grayscale + Otsu threshold",
    "adaptive_threshold": "Grayscale + adaptive threshold",
    "denoised": "Grayscale + denoise",
    "deskewed": "Grayscale + deskew",
    "enhanced_contrast": "Grayscale + CLAHE contrast enhancement",
}

OrientationDetector = Callable[[Image.Image], Optional[int]]


@dataclass(frozen=True)
class PreprocessConfig:
    steps: tuple[str, ...] = ()
    min_text_height: float = 20.0      # upscale when median glyph height (px) is below this
    target_text_height: float = 32.0   # ...to roughly this height
    max_upscale: float = 3.0
    max_pixels: int = 40_000_000
    max_skew_angle: float = 15.0
    min_skew_angle: float = 0.3
    denoise_strength: int = 10
    denoise_only_if_noisy: bool = False
    noise_threshold: float = 4.0
    sharpen_blur_threshold: float = 100.0
    adaptive_block_size: int = 31
    adaptive_c: int = 15
    contrast_clip_limit: float = 2.0

    def __post_init__(self) -> None:
        unknown = [s for s in self.steps if s not in STEP_ORDER]
        if unknown:
            raise ValueError(f"Unknown preprocessing step(s): {', '.join(unknown)}")

    @classmethod
    def from_preset(cls, preset: str, **overrides: object) -> "PreprocessConfig":
        if preset not in PRESETS:
            raise ValueError(f"Unknown preprocessing preset '{preset}'.")
        if preset == "auto":
            overrides.setdefault("denoise_only_if_noisy", True)
        return cls(steps=PRESETS[preset], **overrides)  # type: ignore[arg-type]


@dataclass
class PreprocessResult:
    image: Image.Image
    steps: list[PreprocessStepRecord] = field(default_factory=list)
    scale_factor: float = 1.0

    @property
    def applied_steps(self) -> list[str]:
        return [s.step for s in self.steps if s.applied]


def run_pipeline(
    image: Image.Image,
    config: PreprocessConfig,
    orientation_detector: OrientationDetector | None = None,
) -> PreprocessResult:
    """Apply the configured steps to a copy of ``image``. The input image is never modified."""
    arr = pil_to_array(image)
    records: list[PreprocessStepRecord] = []
    scale = 1.0
    selected = set(config.steps)

    for step in STEP_ORDER:
        if step not in selected:
            continue
        applied, detail = True, ""
        if step == "orientation":
            if orientation_detector is None:
                applied, detail = False, "no orientation detector available"
            else:
                rotation = orientation_detector(array_to_pil(arr))
                if rotation is None:
                    applied, detail = False, "orientation could not be determined"
                elif rotation % 360 == 0:
                    applied, detail = False, "already upright"
                else:
                    arr = deskew.rotate_right_angle(arr, rotation)
                    detail = f"rotated {rotation} degrees clockwise"
        elif step == "crop_borders":
            arr, applied = enhancement.crop_borders(arr)
            detail = "cropped to content" if applied else "no significant border found"
        elif step == "grayscale":
            arr = enhancement.to_grayscale(arr)
        elif step == "invert":
            gray = enhancement.to_grayscale(arr)
            if float(np.median(gray)) < 110:
                arr = 255 - gray
                detail = "dark background inverted"
            else:
                applied, detail = False, "background is already light"
        elif step == "flatten_illumination":
            arr, variation = enhancement.flatten_illumination(arr)
            detail = f"background variation {variation:.0%}"
        elif step == "upscale":
            h, w = arr.shape[:2]
            text_height = enhancement.estimate_text_height(arr)
            if text_height is None:
                applied, detail = False, "text height could not be measured"
            elif text_height >= config.min_text_height:
                applied, detail = False, f"text height {text_height:.0f}px is sufficient"
            else:
                factor = enhancement.compute_upscale_factor(
                    w, h, text_height, config.target_text_height, config.max_upscale, config.max_pixels
                )
                if factor > 1.01:
                    arr = enhancement.resize(arr, factor)
                    scale *= factor
                    detail = f"text height {text_height:.0f}px, scaled x{factor:.2f}"
                else:
                    applied, detail = False, "pixel budget does not allow upscaling"
        elif step == "deskew":
            angle = deskew.estimate_skew_angle(arr, config.max_skew_angle)
            if abs(angle) >= config.min_skew_angle:
                # Fill the exposed corners with the page's own edge colour, not white: a bright wedge
                # next to a shadowed page breaks global thresholding (measured: all text lost).
                arr = deskew.rotate(arr, angle, background=deskew.edge_background(arr))
                detail = f"rotated {angle:+.2f} degrees"
            else:
                applied, detail = False, f"skew {angle:+.2f} degrees is below threshold"
        elif step == "denoise":
            noise = denoising.estimate_noise(arr)
            if config.denoise_only_if_noisy and noise < config.noise_threshold:
                applied, detail = False, f"noise level {noise:.1f} is low"
            else:
                arr = denoising.denoise(arr, config.denoise_strength)
                detail = f"noise level {noise:.1f}"
        elif step == "contrast":
            arr = enhancement.enhance_contrast(arr, config.contrast_clip_limit)
        elif step == "sharpen":
            score = enhancement.blur_score(arr)
            if score < config.sharpen_blur_threshold:
                arr = enhancement.sharpen(arr)
                detail = f"blur score {score:.0f}"
            else:
                applied, detail = False, f"image is sharp (blur score {score:.0f})"
        elif step == "adaptive_threshold":
            arr = enhancement.adaptive_threshold(arr, config.adaptive_block_size, config.adaptive_c)
        elif step == "otsu_threshold":
            if "adaptive_threshold" in selected:
                applied, detail = False, "adaptive threshold already applied"
            else:
                arr = enhancement.otsu_threshold(arr)
        records.append(PreprocessStepRecord(step=step, applied=applied, detail=detail))

    return PreprocessResult(image=array_to_pil(np.ascontiguousarray(arr)), steps=records, scale_factor=scale)
