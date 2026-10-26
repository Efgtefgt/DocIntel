"""Suspicious-text review, OCR quality evaluation, image analysis and the smart plan."""

from __future__ import annotations

import pytest

from docintel.core.result_models import BoundingBox, OCRResult, OCRWord
from docintel.evaluation import synthetic_data as synth
from docintel.extraction.ocr_review import apply_suggestion, review
from docintel.ocr import quality
from docintel.preprocessing.analysis import alternative_plans, analyze, smart_plan
from docintel.preprocessing.preprocessing_pipeline import PreprocessConfig, run_pipeline


def _ocr(tokens: list[tuple[str, float]], text: str | None = None) -> OCRResult:
    words = [OCRWord(text=t, confidence=c, bbox=BoundingBox(x=i * 50, y=0, width=40, height=20), line=0,
                     low_confidence=c < 60) for i, (t, c) in enumerate(tokens)]
    return OCRResult(engine="fake", languages=["eng"], text=text or " ".join(t for t, _ in tokens), words=words,
                     mean_confidence=sum(c for _, c in tokens) / len(tokens) if tokens else None)


# --- suspicious text -----------------------------------------------------------------------------
def test_review_flags_confusions_with_suggestions():
    flags = {f.text: f for f in review(_ocr([("Inv0ice", 90), ("number", 95), ("48Z13", 90), ("he||o", 40)]))}
    assert flags["Inv0ice"].suggestion == "Invoice"
    assert flags["48Z13"].suggestion == "48213"
    assert "low OCR confidence" in flags["he||o"].reasons
    assert "number" not in flags


@pytest.mark.parametrize("token", ["INC-1024", "A4", "3rd", "10kg", "14:35", "2024-05-17", "XR-7", "www"])
def test_review_does_not_flag_legitimate_codes(token):
    assert review(_ocr([(token, 95)])) == []


def test_review_keeps_punctuation_around_suggestion():
    flag = review(_ocr([("(Inv0ice),", 90)]))[0]
    assert flag.suggestion == "(Invoice),"


def test_apply_suggestion_only_whole_words():
    text = "Inv0ice no. 5 and Inv0ices"
    assert apply_suggestion(text, "Inv0ice", "Invoice") == "Invoice no. 5 and Inv0ices"


# --- quality evaluation ---------------------------------------------------------------------------
def test_glyph_estimate_close_to_character_count():
    lines = ["The quick brown fox jumps over the lazy dog", "Invoice number 48213 was paid"]
    expected = sum(c.isalnum() for c in "".join(lines))
    estimate = quality.estimate_glyph_count(synth.render_text_image(lines))
    assert estimate is not None and 0.8 * expected <= estimate <= 1.25 * expected


def test_glyph_estimate_robust_to_shadow():
    lines = synth.FORM_LINES
    plain = quality.estimate_glyph_count(synth.render_scene(lines))
    shadowed = quality.estimate_glyph_count(synth.photo_effects(synth.render_scene(lines), shadow=0.75))
    assert abs(shadowed - plain) <= 0.15 * plain


def test_verdict_detects_missing_text_despite_high_confidence():
    ocr = _ocr([("Location:", 96), ("MG", 95)])  # high confidence, but most of the page is missing
    verdict, notes = quality.verdict(ocr, expected=120)
    assert verdict == "poor" and any("visible characters" in n for n in notes)
    assert quality.verdict(_ocr([]), expected=50)[0] == "poor"


def test_score_prefers_complete_output():
    full = _ocr([("word", 90)] * 20, text="word " * 20)
    partial = _ocr([("word", 95)] * 5, text="word " * 5)
    assert quality.score(full, 80) > quality.score(partial, 80)


# --- analysis + smart plan ------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name, expected_step",
    [("shadow_photo", "flatten_illumination"), ("dark_mode_screenshot", "invert"),
     ("scanner_border_skew", "crop_borders"), ("small_noisy_print", "upscale"), ("phone_photo", "deskew")],
)
def test_smart_plan_reacts_to_conditions(name, expected_step):
    image = synth.render_hard_sample(synth.HARD_SAMPLES[name])
    steps, reasons = smart_plan(analyze(image))
    assert expected_step in steps, (steps, reasons)
    assert len(reasons) == len(steps)


def test_smart_plan_leaves_clean_images_alone():
    steps, _ = smart_plan(analyze(synth.render_hard_sample(synth.HARD_SAMPLES["clean_form"])))
    corrective = {"flatten_illumination", "crop_borders", "invert", "denoise", "deskew", "orientation", "contrast"}
    assert steps[0] == "grayscale" and not corrective & set(steps), steps


def test_analysis_uses_orientation_detector_on_cleaned_copy():
    seen = []
    analysis = analyze(synth.render_scene(synth.FORM_LINES), orientation_detector=lambda img: seen.append(img.mode) or 180)
    assert analysis.rotation == 180 and seen == ["L"]
    assert any(f.code == "rotation" for f in analysis.findings)


def test_alternative_plans_keep_orientation():
    alts = alternative_plans(("orientation", "grayscale", "flatten_illumination"))
    assert all("orientation" in steps for _, steps in alts)
    assert {label for label, _ in alts} >= {"grayscale only", "plan + adaptive threshold"}


def test_new_steps_run():
    dark = synth.render_scene(["Light text on dark"], text_color=(220, 220, 220), background=(25, 25, 30))
    result = run_pipeline(dark, PreprocessConfig(steps=("grayscale", "invert", "flatten_illumination")))
    gray = result.image.convert("L")
    assert gray.getpixel((5, 5)) > 200  # background is now light
    assert all(s.applied for s in result.steps)
