"""Image preprocessing tests (no OCR engine required)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from docintel.evaluation import synthetic_data as synth
from docintel.preprocessing import denoising, deskew, enhancement
from docintel.preprocessing.image_loader import iter_frames, pil_to_array
from docintel.preprocessing.preprocessing_pipeline import PRESETS, PreprocessConfig, run_pipeline


@pytest.fixture
def page() -> Image.Image:
    return synth.render_text_image([f"Line number {i} with some sample text" for i in range(8)], font_size=28)


def test_original_preset_returns_identical_copy(page):
    result = run_pipeline(page, PreprocessConfig.from_preset("original"))
    assert result.steps == []
    assert np.array_equal(pil_to_array(result.image), pil_to_array(page))
    assert result.image is not page


def test_input_image_is_not_modified(page):
    before = pil_to_array(page).copy()
    run_pipeline(page, PreprocessConfig.from_preset("grayscale_threshold"))
    assert np.array_equal(pil_to_array(page), before)


@pytest.mark.parametrize("preset", list(PRESETS))
def test_every_preset_runs(page, preset):
    result = run_pipeline(page, PreprocessConfig.from_preset(preset))
    assert result.image.size[0] > 0


def test_grayscale_and_thresholds_are_binary(page):
    for preset in ("grayscale_threshold", "adaptive_threshold"):
        arr = pil_to_array(run_pipeline(page, PreprocessConfig.from_preset(preset)).image)
        assert arr.ndim == 2
        assert set(np.unique(arr)) <= {0, 255}


def test_unknown_step_rejected():
    with pytest.raises(ValueError):
        PreprocessConfig(steps=("magic",))


@pytest.mark.parametrize("angle", [-6.0, -2.5, 3.0, 7.0])
def test_skew_estimation(page, angle):
    rotated = synth.degrade(page, rotate=angle)
    estimate = deskew.estimate_skew_angle(np.asarray(rotated))
    # Rotating by -estimate should undo the skew; PIL rotates counter-clockwise for positive angles.
    assert estimate == pytest.approx(-angle, abs=0.5)


def test_deskew_step_skips_straight_image(page):
    result = run_pipeline(page, PreprocessConfig.from_preset("deskewed"))
    record = next(s for s in result.steps if s.step == "deskew")
    assert not record.applied


def test_deskew_step_applies_on_rotated_image(page):
    result = run_pipeline(synth.degrade(page, rotate=5), PreprocessConfig.from_preset("deskewed"))
    record = next(s for s in result.steps if s.step == "deskew")
    assert record.applied


def test_upscale_only_for_low_resolution(page):
    small = synth.degrade(page, scale=0.3)
    config = PreprocessConfig(steps=("upscale",))
    upscaled = run_pipeline(small, config)
    assert upscaled.scale_factor > 1 and upscaled.image.width > small.width

    big = page.resize((page.width * 3, page.height * 3))
    unchanged = run_pipeline(big, config)
    assert unchanged.scale_factor == 1.0
    assert not unchanged.steps[0].applied


def test_upscale_respects_pixel_budget():
    factor = enhancement.compute_upscale_factor(400, 300, text_height=5, target_height=40, max_scale=10, max_pixels=1_000_000)
    assert 400 * 300 * factor**2 <= 1_000_000 * 1.0001


def test_upscale_ignores_wide_short_images_with_large_text():
    # A one-line strip has a tiny short side but perfectly readable text: do not upscale it.
    strip = synth.render_text_image(["A single line of large text on a narrow strip of paper"], font_size=40)
    assert min(strip.size) < 200
    result = run_pipeline(strip, PreprocessConfig(steps=("upscale",)))
    assert not result.steps[0].applied and "sufficient" in result.steps[0].detail


def test_text_height_estimate(page):
    small = synth.degrade(page, scale=0.4)
    assert enhancement.estimate_text_height(np.asarray(small)) < enhancement.estimate_text_height(np.asarray(page))
    assert enhancement.estimate_text_height(np.full((100, 100), 255, np.uint8)) is None


def test_sharpen_skipped_on_sharp_image(page):
    result = run_pipeline(page, PreprocessConfig(steps=("sharpen",)))
    assert not result.steps[0].applied
    blurry = synth.degrade(page, blur=3)
    assert run_pipeline(blurry, PreprocessConfig(steps=("sharpen",))).steps[0].applied


def test_denoise_reduces_noise(page):
    noisy = synth.degrade(page, noise=35, seed=1)
    before = denoising.estimate_noise(np.asarray(noisy))
    after = denoising.estimate_noise(denoising.denoise(np.asarray(noisy)))
    assert after < before


def test_auto_denoise_skips_clean_image(page):
    result = run_pipeline(page, PreprocessConfig.from_preset("auto"))
    record = next(s for s in result.steps if s.step == "denoise")
    assert not record.applied


def test_contrast_enhancement_increases_spread():
    flat = np.full((200, 200), 120, dtype=np.uint8)
    flat[50:150, 50:150] = 135
    out = enhancement.enhance_contrast(flat)
    assert int(out.max()) - int(out.min()) > 15


def test_crop_borders_removes_dark_frame(page):
    arr = np.asarray(page.convert("L")).copy()
    framed = np.zeros((arr.shape[0] + 120, arr.shape[1] + 120), dtype=np.uint8)
    framed[60:-60, 60:-60] = arr
    cropped, applied = enhancement.crop_borders(framed)
    assert applied
    assert cropped.shape[0] < framed.shape[0] and cropped.shape[1] < framed.shape[1]


def test_orientation_step_uses_detector(page):
    upside_down = page.rotate(180)
    result = run_pipeline(upside_down, PreprocessConfig(steps=("orientation",)), orientation_detector=lambda img: 180)
    assert result.steps[0].applied
    assert np.array_equal(pil_to_array(result.image), pil_to_array(page))


def test_orientation_step_without_detector_is_skipped(page):
    result = run_pipeline(page, PreprocessConfig(steps=("orientation",)))
    assert not result.steps[0].applied
    assert "no orientation detector" in result.steps[0].detail


def test_rgba_is_flattened_onto_white():
    rgba = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    data = synth.image_bytes(rgba, "PNG")
    _, frame = next(iter_frames(data))
    assert frame.mode == "RGB" and frame.getpixel((10, 10)) == (255, 255, 255)
