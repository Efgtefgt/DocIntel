"""Image quality analysis before OCR, and the "smart" preprocessing plan derived from it.

All measurements are cheap (a downscaled copy is analysed; only text height uses full
resolution). Orientation detection is optional and uses Tesseract OSD when available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import cv2
import numpy as np
from PIL import Image

from docintel.preprocessing import denoising, deskew, enhancement
from docintel.preprocessing.image_loader import pil_to_array

ANALYSIS_MAX_SIDE = 1600
Severity = Literal["info", "warning", "problem"]


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    message: str
    step: Optional[str] = None  # preprocessing step that addresses it


@dataclass(frozen=True)
class ImageAnalysis:
    width: int
    height: int
    text_height: Optional[float]
    skew_angle: float
    rotation: Optional[int]        # clockwise degrees to make upright; None = unknown/not checked
    blur_score: float
    noise_level: float
    contrast: float                 # spread between dark and light pixels (5th..95th percentile), 0-255
    illumination_variation: float   # 0 = evenly lit background, 1 = extreme shadow
    dark_background: bool           # light text on a dark background (e.g. dark-mode screenshot)
    dark_border: bool
    colorful: bool
    kind: str                       # "screenshot", "scan" or "photo" (heuristic)
    findings: list[Finding] = field(default_factory=list)

    @property
    def problems(self) -> list[Finding]:
        return [f for f in self.findings if f.severity != "info"]


def _downscaled_gray(arr: np.ndarray) -> tuple[np.ndarray, float]:
    gray = enhancement.to_grayscale(arr)
    scale = min(1.0, ANALYSIS_MAX_SIDE / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray, scale


def estimate_background(gray: np.ndarray, text_height: Optional[float]) -> np.ndarray:
    """Background brightness map: text is removed with a max filter, then smoothed."""
    k = int(max(15, 2.5 * (text_height or 12))) | 1
    background = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return cv2.medianBlur(background, min(255, k if k % 2 else k + 1))


def illumination_variation(gray: np.ndarray, text_height: Optional[float]) -> float:
    bg = estimate_background(gray, text_height).astype(np.float32)
    lo, hi = np.percentile(bg, 5), np.percentile(bg, 95)
    return float((hi - lo) / max(hi, 1.0))


def smart_plan(a: ImageAnalysis) -> tuple[tuple[str, ...], list[str]]:
    """Choose preprocessing steps from the analysis. Returns (steps, human-readable reasons).

    Thresholds were tuned on the bundled synthetic benchmark (data/samples/hard); see
    docs/EVALUATION.md. Steps that measured as harmful when not needed (border cropping on
    shadowed photos, contrast enhancement on noisy print) are only used when their condition holds.
    """
    steps: list[str] = []
    reasons: list[str] = []

    def add(step: str, why: str) -> None:
        steps.append(step)
        reasons.append(f"{step}: {why}")

    if a.rotation not in (None, 0):
        add("orientation", f"page is rotated; turn {a.rotation}° clockwise")
    if a.dark_border:
        add("crop_borders", "dark scanner border detected")
    add("grayscale", "OCR works on brightness only")
    if a.dark_background:
        add("invert", "light text on a dark background")
    if a.illumination_variation > 0.25:
        add("flatten_illumination", f"uneven lighting ({a.illumination_variation:.0%} background variation)")
    if a.noise_level > 4.0:
        add("denoise", f"noise level {a.noise_level:.1f}")
    if abs(a.skew_angle) >= 0.3:
        add("deskew", f"text tilted {a.skew_angle:+.1f}°")
    if a.text_height is not None and a.text_height < 20:
        add("upscale", f"small text (~{a.text_height:.0f} px)")
    if a.contrast < 60 and a.noise_level < 4.0 and "flatten_illumination" not in steps:
        add("contrast", f"very low contrast ({a.contrast:.0f}/255)")
    if a.blur_score < 100 and a.noise_level < 4.0:
        add("sharpen", f"blurry (sharpness {a.blur_score:.0f})")
    return tuple(steps), reasons


def analyze(image: Image.Image, orientation_detector: Callable[[Image.Image], Optional[int]] | None = None) -> ImageAnalysis:
    arr = pil_to_array(image)
    h, w = arr.shape[:2]
    gray, scale = _downscaled_gray(arr)
    dark_background = float(np.median(gray)) < 110
    work = 255 - gray if dark_background else gray

    text_height = enhancement.estimate_text_height(255 - enhancement.to_grayscale(arr) if dark_background else arr)
    th_small = text_height * scale if text_height else None
    skew = deskew.estimate_skew_angle(work)
    blur = enhancement.blur_score(work)
    noise = denoising.estimate_noise(work)
    p5, p95 = np.percentile(work, 5), np.percentile(work, 95)
    contrast = float(p95 - p5)
    illum = illumination_variation(work, th_small)
    box = enhancement.find_content_box(work, margin=0)
    dark_border = False
    if box is not None:
        bx, by, bw, bh = box
        border_dark = np.concatenate([work[:max(1, by), :].ravel(), work[by + bh:, :].ravel(),
                                      work[:, :max(1, bx)].ravel(), work[:, bx + bw:].ravel()])
        dark_border = border_dark.size > 0 and float(np.median(border_dark)) < 80 and bw * bh < 0.95 * work.size
    colorful = arr.ndim == 3 and float(cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[:, :, 1].mean()) > 40
    rotation = None
    if orientation_detector:
        # Orientation detection suffers from the same problems as OCR: shadows, inverted text and
        # tiny glyphs. Detect on a cleaned copy (dark-on-light, lighting flattened, enlarged if small).
        clean = enhancement.to_grayscale(arr)
        if dark_background:
            clean = 255 - clean
        if illum > 0.25:
            clean, _ = enhancement.flatten_illumination(clean, text_height)
        probe = Image.fromarray(clean)
        if text_height is not None and text_height < 24:
            factor = enhancement.compute_upscale_factor(w, h, text_height, 32.0, 3.0, 25_000_000)
            if factor > 1.05:
                probe = probe.resize((int(w * factor), int(h * factor)), Image.BICUBIC)
        rotation = orientation_detector(probe)

    if noise < 2.0 and blur > 500 and illum < 0.1:
        kind = "screenshot"
    elif illum > 0.25 or (abs(skew) > 1 and noise > 4):
        kind = "photo"
    else:
        kind = "scan"

    findings: list[Finding] = []
    if rotation not in (None, 0):
        findings.append(Finding("rotation", "problem", f"Page appears rotated; needs {rotation}° clockwise", "orientation"))
    if dark_border:
        findings.append(Finding("dark_border", "warning", "Dark scanner border around the page", "crop_borders"))
    if dark_background:
        findings.append(Finding("dark_background", "warning", "Light text on a dark background", "invert"))
    if illum > 0.25:
        findings.append(Finding("uneven_lighting", "problem" if illum > 0.4 else "warning",
                                f"Uneven lighting or shadow (variation {illum:.0%})", "flatten_illumination"))
    if noise > 4.0:
        findings.append(Finding("noise", "warning", f"Visible noise (level {noise:.1f})", "denoise"))
    if abs(skew) >= 0.3:
        findings.append(Finding("skew", "warning" if abs(skew) < 3 else "problem", f"Text is tilted {skew:+.1f}°", "deskew"))
    if text_height is not None and text_height < 20:
        findings.append(Finding("small_text", "warning", f"Small text (about {text_height:.0f} px high)", "upscale"))
    if text_height is None:
        findings.append(Finding("no_text_found", "warning", "Could not measure text size; the image may contain little text"))
    if contrast < 90:
        findings.append(Finding("low_contrast", "warning", f"Low contrast ({contrast:.0f}/255)", "contrast"))
    if blur < 100:
        findings.append(Finding("blur", "warning", f"Image looks blurry (sharpness {blur:.0f})", "sharpen"))
    if not findings:
        findings.append(Finding("good", "info", "No quality problems detected"))
    return ImageAnalysis(width=w, height=h, text_height=text_height, skew_angle=skew, rotation=rotation,
                         blur_score=blur, noise_level=noise, contrast=contrast, illumination_variation=illum,
                         dark_background=dark_background, dark_border=dark_border, colorful=colorful,
                         kind=kind, findings=findings)


def alternative_plans(plan: tuple[str, ...]) -> list[tuple[str, tuple[str, ...]]]:
    """Fallback plans tried only when the first OCR result looks weak (orientation is kept)."""
    keep = ("orientation",) if "orientation" in plan else ()
    alternatives: list[tuple[str, tuple[str, ...]]] = []
    if "adaptive_threshold" not in plan:
        alternatives.append(("plan + adaptive threshold", plan + ("adaptive_threshold",)))
    plain = keep + ("grayscale",)
    if plan != plain:
        alternatives.append(("grayscale only", plain))
    toggled = (tuple(s for s in plan if s != "flatten_illumination") if "flatten_illumination" in plan
               else tuple(plan) + ("flatten_illumination",))
    alternatives.append(("plan with lighting correction toggled", toggled))
    return alternatives


def analysis_metrics(a: ImageAnalysis) -> dict:
    return {
        "image": f"{a.width} x {a.height} px", "kind": a.kind,
        "text height (px)": round(a.text_height, 1) if a.text_height else None,
        "skew (deg)": a.skew_angle, "rotation needed (deg)": a.rotation,
        "sharpness": round(a.blur_score), "noise": round(a.noise_level, 1), "contrast": round(a.contrast),
        "lighting variation": f"{a.illumination_variation:.0%}", "dark background": a.dark_background,
        "dark border": a.dark_border,
    }
