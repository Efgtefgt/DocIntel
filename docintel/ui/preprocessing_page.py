"""Image enhancement: before/after preview, individual enhancement controls and OCR comparison.

Preprocessing runs only for the variants and steps selected here (results are cached per
page and configuration). OCR runs only when the comparison button is pressed. The original
image and existing OCR results are never modified.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from docintel.core.exceptions import DocIntelError, OCREngineUnavailableError
from docintel.evaluation.metrics import cer, wer
from docintel.ocr.base_engine import OCROptions
from docintel.preprocessing.analysis import analysis_metrics, analyze, smart_plan
from docintel.preprocessing.preprocessing_pipeline import PRESET_DESCRIPTIONS, PRESETS, PreprocessConfig, run_pipeline
from docintel.ui import state
from docintel.ui.components import badge, card, document_picker, engine_choices, page_header, page_image, upload_panel

# Individual controls (label, step name, help). Thresholding is a separate choice below.
CONTROLS = [
    ("Rotation correction", "orientation", "Fix pages scanned sideways or upside down (Tesseract OSD)"),
    ("Crop borders", "crop_borders", "Remove dark scanner edges and empty margins"),
    ("Grayscale", "grayscale", "Convert to shades of gray"),
    ("Denoising", "denoise", "Reduce speckle and sensor noise (slower)"),
    ("Deskewing", "deskew", "Straighten slightly rotated text; skipped below 0.3°"),
    ("Upscale small text", "upscale", "Enlarge only when measured character height is small"),
    ("Contrast enhancement", "contrast", "Local contrast (CLAHE) for faint text"),
    ("Sharpening", "sharpen", "Only applied when the image is measurably blurry"),
]
THRESHOLDS = {"None": None, "Adaptive": "adaptive_threshold", "Otsu": "otsu_threshold"}


@st.cache_data(max_entries=12, show_spinner="Preprocessing...")
def _preprocess(source_key: str, _image, config: PreprocessConfig, _detector):
    """Cached per (source page, config); arguments starting with '_' are not hashed."""
    return run_pipeline(_image, config, orientation_detector=_detector)


@st.cache_data(max_entries=12, show_spinner="Analysing the image...")
def _analysis(source_key: str, _image, _detector):
    """Image analysis and the smart plan for one page (cached per page)."""
    analysis = analyze(_image, _detector)
    steps, reasons = smart_plan(analysis)
    return analysis, steps, reasons


def _config_for(preset: str, source_key: str, image, detector, max_pixels: int):
    """PreprocessConfig for a preset; 'smart' runs the analysis and returns its plan."""
    if preset == "smart":
        analysis, steps, _ = _analysis(source_key, image, detector)
        rotation = analysis.rotation
        det = (lambda _img, r=rotation: r) if rotation else None
        return PreprocessConfig(steps=steps, max_pixels=max_pixels), det
    return PreprocessConfig.from_preset(preset, max_pixels=max_pixels), detector


def _detector():
    try:
        return state.engine_factory().get("tesseract").detect_orientation
    except OCREngineUnavailableError:
        return None


def _describe(result) -> None:
    applied = [f"{s.step}" + (f" ({s.detail})" if s.detail else "") for s in result.steps if s.applied]
    skipped = [f"{s.step}: {s.detail}" for s in result.steps if not s.applied]
    st.caption("Applied: " + (", ".join(applied) if applied else "none"))
    if skipped:
        st.caption("Skipped (not needed): " + "; ".join(skipped))


def _custom_config(settings) -> PreprocessConfig:
    st.caption("Steps run in a fixed, sensible order regardless of the order you tick them.")
    cols = st.columns(2)
    steps = []
    for i, (label, step, help_text) in enumerate(CONTROLS):
        if cols[i % 2].checkbox(label, key=f"enh_ctl_{step}", value=step == "grayscale", help=help_text):
            steps.append(step)
    threshold = st.segmented_control("Thresholding", list(THRESHOLDS), key="enh_threshold", default="None", required=True,
                                     help="Adaptive suits uneven lighting and shadows; Otsu suits evenly lit scans")
    if THRESHOLDS[threshold]:
        steps.append(THRESHOLDS[threshold])
    return PreprocessConfig(steps=tuple(steps), max_pixels=settings.max_image_pixels)


def render() -> None:
    page_header("Image enhancement", "Preview preprocessing before OCR and compare variants. "
                "The original image and earlier OCR results are never replaced.")
    upload_panel("enhancement")
    settings = state.settings()
    c1, c2 = st.columns([3, 1])
    with c1:
        doc = document_picker("Source file", "enh_doc", prefer=[st.session_state.get("last_added", ""),
                                                                 st.session_state.get("last_processed", "")])
    if doc is None:
        st.info("Add an image or PDF above to try enhancements.", icon=":material/info:")
        return
    page_count = doc.info.page_count
    number = c2.number_input("Page", 1, page_count, 1, key=f"enh_page_{doc.file_id}") if page_count > 1 else 1
    original = page_image(doc.data, doc.info.kind.value, int(number), dpi=settings.pdf_render_dpi)
    source_key = f"{doc.file_id}:{number}:{settings.pdf_render_dpi}"
    detector = _detector()
    _analysis_card(source_key, original, detector)

    with card("Before and after", "adjust the enhancements and compare with the original"):
        mode = st.segmented_control("Configure with", ["Preset", "Individual controls"], key="enh_mode",
                                    default="Preset", required=True)
        if mode == "Preset":
            preset = st.selectbox("Preset", list(PRESETS), index=list(PRESETS).index("smart"), key="enh_preset",
                                  format_func=lambda p: f"{p} · {PRESET_DESCRIPTIONS[p]}")
            config, det = _config_for(preset, source_key, original, detector, settings.max_image_pixels)
        else:
            config, det = _custom_config(settings), detector
        result = _preprocess(source_key, original, config, det)
        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown("**Original**")
            st.image(original, width="stretch", alt="Original page")
        with right:
            st.markdown("**Processed**")
            st.image(result.image, width="stretch", alt="Processed page")
            _describe(result)

    with card("Compare engines", "run each available OCR engine on the processed image above"):
        _engine_comparison(result.image, settings, f"{source_key}:{'+'.join(config.steps)}")

    with card("Compare variants", f"OCR several presets on this page (max {settings.max_preprocessing_variants})"):
        variants = st.multiselect("Presets to compare", list(PRESETS), default=["original", "smart"],
                                  max_selections=settings.max_preprocessing_variants, key="enh_variants",
                                  format_func=lambda p: f"{p} · {PRESET_DESCRIPTIONS[p]}")
        if not variants:
            return
        processed = {}
        for v in variants:
            cfg, det = _config_for(v, source_key, original, detector, settings.max_image_pixels)
            processed[v] = _preprocess(source_key, original, cfg, det)
        cols = st.columns(len(processed))
        for col, (variant, res) in zip(cols, processed.items()):
            with col:
                st.markdown(f"**{variant}**")
                st.image(res.image, width="stretch", alt=f"{variant} preview")
        _comparison(processed, settings, source_key)


SEVERITY_BADGE = {"problem": "bad", "warning": "warn", "info": "ok"}


def _analysis_card(source_key: str, image, detector) -> None:
    analysis, steps, reasons = _analysis(source_key, image, detector)
    with card("Image analysis", f"looks like a {analysis.kind}; measured before any OCR"):
        with st.container(horizontal=True, wrap=True, gap="small"):
            for finding in analysis.findings:
                badge(finding.message, SEVERITY_BADGE[finding.severity])
        c1, c2 = st.columns([1, 1.2], gap="medium")
        with c1:
            st.markdown("**Smart preprocessing plan**")
            for reason in reasons:
                st.markdown(f"- {reason}")
            if detector is None:
                st.caption("Rotation was not checked: Tesseract orientation data (osd) is not available.")
        with c2:
            st.markdown("**Measurements**")
            st.dataframe(pd.DataFrame(analysis_metrics(analysis).items(), columns=["Measure", "Value"]).astype(str),
                         hide_index=True)


def _languages_for(engine, settings, key: str) -> list[str]:
    langs = [lang for lang in engine.available_languages() if lang != "osd"]
    default = [lang for lang in settings.language_list if lang in langs] or langs[:1]
    if engine.supports_multiple_languages:
        return st.multiselect(f"{engine.display_name} languages", langs, default=default, key=key)
    return [st.selectbox(f"{engine.display_name} language", langs, index=langs.index(default[0]) if default else 0, key=key)]


def _engine_comparison(image, settings, source_key: str) -> None:
    statuses = state.engine_factory().statuses()
    available = [s for s in statuses if s.available]
    missing = [s for s in statuses if not s.available]
    if len(available) < 2:
        names = ", ".join(f"{s.display_name} ({s.reason})" for s in missing) or "none"
        st.info(f"Only {available[0].display_name if available else 'no engine'} is available in this Python, so there is "
                f"nothing to compare against. Unavailable: {names}", icon=":material/info:")
        if not available:
            return
    cols = st.columns(max(1, len(available)))
    chosen = {}
    for col, status in zip(cols, available):
        with col:
            engine = state.engine_factory().get(status.name)
            chosen[status.name] = (engine, _languages_for(engine, settings, f"enh_cmp_lang_{status.name}"))
    reference = st.text_area("Reference text (optional)", height=80, key="enh_cmp_reference",
                             placeholder="Paste the verified text of this page to compute CER and WER")
    result_key = f"cmp_engines_{source_key}"
    if st.button("Run all engines", icon=":material/compare:", key="enh_cmp_run",
                 disabled=not all(langs for _, langs in chosen.values())):
        rows, texts = [], {}
        with st.spinner("Running OCR engines one after another..."):
            for name, (engine, langs) in chosen.items():
                try:
                    ocr = engine.recognize(image, OCROptions(languages=tuple(langs), psm=3 if engine.supports_psm else None,
                                                             timeout_seconds=settings.ocr_timeout_seconds,
                                                             low_confidence_threshold=settings.low_confidence_threshold))
                    texts[engine.display_name] = ocr.text
                    row = {"Engine": engine.display_name, "Words": len(ocr.words), "Mean confidence": ocr.mean_confidence,
                           "Confidence scale": ocr.confidence_scale, "Seconds": ocr.duration_seconds}
                    if reference.strip():
                        row["CER"] = round(cer(reference, ocr.text), 4)
                        row["WER"] = round(wer(reference, ocr.text), 4)
                    rows.append(row)
                except DocIntelError as exc:
                    rows.append({"Engine": engine.display_name, "Error": exc.user_message})
        st.session_state[result_key] = {"rows": rows, "texts": texts}
    saved = st.session_state.get(result_key)
    if saved:
        st.caption("Confidence values are on each engine's own scale and are not comparable between engines; "
                   "use CER/WER against a reference to compare accuracy.")
        st.dataframe(pd.DataFrame(saved["rows"]), hide_index=True)
        cols = st.columns(max(1, len(saved["texts"])))
        for col, (name, text) in zip(cols, saved["texts"].items()):
            with col:
                st.markdown(f"**{name}**")
                st.code(text, language=None, wrap_lines=True)


def _comparison(processed: dict, settings, source_key: str) -> None:
    available, _ = engine_choices()
    if not available:
        st.warning("No OCR engine is available; see Settings.", icon=":material/warning:")
        return
    e1, e2 = st.columns(2)
    engine_name = e1.selectbox("Engine", available, key="enh_engine")
    engine = state.engine_factory().get(engine_name)
    langs = [lang for lang in engine.available_languages() if lang != "osd"]
    default_langs = [lang for lang in settings.language_list if lang in langs] or langs[:1]
    languages = e2.multiselect("Languages", langs, default=default_langs, key="enh_langs",
                               max_selections=None if engine.supports_multiple_languages else 1)
    reference = st.text_area("Reference text (optional)", height=90, key="enh_reference",
                             placeholder="Paste the verified text of this page to compute CER and WER")
    result_key = f"cmp_{source_key}"  # results belong to one document page only
    if st.button("Run OCR on selected variants", type="primary", disabled=not languages, icon=":material/play_arrow:",
                 key="enh_run"):
        rows, texts = [], {}
        progress = st.progress(0.0, text="Running OCR...")
        for i, (variant, result) in enumerate(processed.items()):
            try:
                ocr = engine.recognize(result.image, OCROptions(languages=tuple(languages), timeout_seconds=settings.ocr_timeout_seconds,
                                                                low_confidence_threshold=settings.low_confidence_threshold))
                texts[variant] = ocr.text
                row = {"Variant": variant, "Words": len(ocr.words), "Mean confidence": ocr.mean_confidence,
                       "Low-confidence words": len(ocr.low_confidence_words), "Seconds": ocr.duration_seconds}
                if reference.strip():
                    row["CER"] = round(cer(reference, ocr.text), 4)
                    row["WER"] = round(wer(reference, ocr.text), 4)
                rows.append(row)
            except DocIntelError as exc:
                rows.append({"Variant": variant, "Error": exc.user_message})
            progress.progress((i + 1) / len(processed), text=f"Variant {i + 1} of {len(processed)}")
        st.session_state[result_key] = {"rows": rows, "texts": texts, "engine": engine.display_name}
        st.session_state["cmp_results"] = st.session_state[result_key]

    saved = st.session_state.get(result_key)
    if saved:
        st.caption(f"Engine: {saved['engine']}. Confidence is engine-specific; lower CER/WER is better when a reference is given.")
        st.dataframe(pd.DataFrame(saved["rows"]), hide_index=True)
        cols = st.columns(max(1, len(saved["texts"])))
        for col, (variant, text) in zip(cols, saved["texts"].items()):
            with col:
                st.markdown(f"**{variant}**")
                st.code(text, language=None, wrap_lines=True)
