"""Reusable Streamlit UI pieces: upload panel, processing options, progress, previews, status."""

from __future__ import annotations

import html
from contextlib import contextmanager
from itertools import groupby
from pathlib import Path
from typing import Iterable, Iterator, Literal, Optional

import streamlit as st
from PIL import Image

from docintel.core.pipeline import PipelineOptions, PipelineResult
from docintel.core.result_models import OCRResult
from docintel.documents.document_processor import PDFMode
from docintel.documents.pdf_renderer import render_page_from_bytes
from docintel.extraction.table_extractor import TableMethod, available_pdf_methods
from docintel.extraction.templates import TemplateStore
from docintel.extraction.text_normalizer import NormalizationOptions
from docintel.ocr.base_engine import OCROptions
from docintel.ocr.tesseract_engine import PSM_DESCRIPTIONS
from docintel.preprocessing.image_loader import iter_frames
from docintel.preprocessing.preprocessing_pipeline import PRESET_DESCRIPTIONS, PRESETS, STEP_DESCRIPTIONS, STEP_ORDER
from docintel.core.file_detector import SUPPORTED_EXTENSIONS
from docintel.core.security import display_path
from docintel.ui import state, theme

PDF_MODE_LABELS = {
    PDFMode.AUTO: "Auto: embedded text when usable, OCR for scanned pages",
    PDFMode.EMBEDDED_ONLY: "Embedded text only (no OCR)",
    PDFMode.OCR_ALL: "OCR every page (ignore embedded text)",
}
TABLE_METHOD_LABELS = {
    TableMethod.AUTO: "Auto",
    TableMethod.PYMUPDF: "PyMuPDF (lines and text alignment)",
    TableMethod.CAMELOT_LATTICE: "Camelot lattice (ruled tables)",
    TableMethod.CAMELOT_STREAM: "Camelot stream (whitespace-separated)",
}


@st.cache_data(max_entries=24, show_spinner=False)
def page_image(data: bytes, kind: str, page_number: int, dpi: int = 110) -> Image.Image:
    """Preview image for a page/frame (cached in memory for this server process)."""
    if kind == "pdf":
        return render_page_from_bytes(data, page_number, dpi=dpi, max_pixels=12_000_000)
    _, frame = next(iter_frames(data, [page_number]))
    return frame


def engine_choices() -> tuple[list[str], dict[str, str]]:
    """Return (available engine names, name -> reason for unavailable engines)."""
    available, unavailable = [], {}
    for status in state.engine_factory().statuses():
        if status.available:
            available.append(status.name)
        else:
            unavailable[status.display_name] = status.reason
    return available, unavailable


def processing_options_form(key: str, *, show_tables: bool = True, show_template: bool = True,
                            show_page_range: bool = True) -> Optional[PipelineOptions]:
    """Render processing options. Returns None (with a message) if no engine is usable."""
    settings = state.settings()
    saved: dict = st.session_state["processing_options"]
    available, unavailable = engine_choices()
    if not available:
        st.error("No OCR engine is available. Embedded PDF text can still be read with "
                 "'Embedded text only' mode. See Settings for details.")
        for name, reason in unavailable.items():
            st.caption(f"{name}: {reason}")

    col1, col2, col3 = st.columns(3)
    with col1:
        engine_names = available or ["tesseract"]
        default_engine = saved.get("engine", settings.default_engine)
        engine = st.selectbox("OCR engine", engine_names,
                              index=engine_names.index(default_engine) if default_engine in engine_names else 0,
                              key=f"{key}_engine", disabled=not available)
        langs_available: list[str] = []
        supports_psm, single_lang = False, False
        if available:
            eng = state.engine_factory().get(engine)
            langs_available = [lang for lang in eng.available_languages() if lang != "osd"]
            supports_psm, single_lang = eng.supports_psm, not eng.supports_multiple_languages
        wanted = [lang for lang in saved.get("languages", settings.language_list) if lang in langs_available]
        if not wanted and langs_available:
            wanted = [langs_available[0]]
        if single_lang:
            languages = [st.selectbox("Language", langs_available, index=langs_available.index(wanted[0]) if wanted else 0, key=f"{key}_lang1")]
        else:
            languages = st.multiselect("Languages (installed only)", langs_available, default=wanted, key=f"{key}_langs",
                                       help="Only language data found on this computer is listed.")
    with col2:
        presets = list(PRESETS) + ["custom"]
        preset = st.selectbox("Preprocessing", presets, index=presets.index(saved.get("preset", "smart")) if saved.get("preset", "smart") in presets else 0,
                              format_func=lambda p: p if p == "custom" else f"{p} - {PRESET_DESCRIPTIONS[p]}",
                              key=f"{key}_preset")
        custom_steps = None
        if preset == "custom":
            custom_steps = tuple(st.multiselect("Steps (run in canonical order)", STEP_ORDER, default=list(saved.get("custom_steps") or ["grayscale"]),
                                                format_func=lambda s: f"{s} - {STEP_DESCRIPTIONS[s]}", key=f"{key}_steps"))
        psm_options = list(PSM_DESCRIPTIONS)
        psm = st.selectbox("Page segmentation (Tesseract)", psm_options, index=psm_options.index(saved.get("psm", 3)),
                           format_func=lambda p: f"{p} - {PSM_DESCRIPTIONS[p]}", key=f"{key}_psm", disabled=not supports_psm)
    with col3:
        modes = list(PDFMode)
        pdf_mode = st.selectbox("PDF handling", modes, index=modes.index(PDFMode(saved.get("pdf_mode", "auto"))),
                                format_func=lambda m: PDF_MODE_LABELS[m], key=f"{key}_pdfmode")
        page_range = st.text_input("Pages (e.g. 1-3, 5; empty = all)", saved.get("page_range", ""), key=f"{key}_pages") if show_page_range else ""
        threshold = st.slider("Low-confidence threshold", 0, 100, int(saved.get("threshold", settings.low_confidence_threshold)), key=f"{key}_thr",
                              help="Words below this engine-specific score are highlighted for review. It is not a probability of correctness.")

    with st.expander("Text normalization, tables and structured extraction"):
        n1, n2 = st.columns(2)
        with n1:
            collapse = st.checkbox("Collapse repeated spaces", saved.get("collapse", True), key=f"{key}_collapse")
            nfkc = st.checkbox("Fold compatibility characters (NFKC, e.g. ligatures)", saved.get("nfkc", False), key=f"{key}_nfkc")
            hyphen = st.checkbox("Join words hyphenated across lines", saved.get("hyphen", False), key=f"{key}_hyphen")
        with n2:
            extract_tables = st.checkbox("Extract tables", saved.get("extract_tables", False), key=f"{key}_tables") if show_tables else False
            methods = available_pdf_methods()
            saved_method = TableMethod(saved.get("table_method", "auto"))
            table_method = st.selectbox("Table method for digital PDFs", methods, index=methods.index(saved_method) if saved_method in methods else 0,
                                        format_func=lambda m: TABLE_METHOD_LABELS[m], key=f"{key}_tmethod", disabled=not extract_tables)
            template = None
            if show_template:
                names = ["(none)"] + TemplateStore(settings.templates_dir).list_names()
                tname = st.selectbox("Structured extraction template", names,
                                     index=names.index(saved.get("template", "(none)")) if saved.get("template") in names else 0, key=f"{key}_tmpl")
                if tname != "(none)":
                    template = TemplateStore(settings.templates_dir).load(tname)

    saved.update(engine=engine, languages=languages, preset=preset, custom_steps=custom_steps, psm=psm, pdf_mode=pdf_mode.value,
                 page_range=page_range, threshold=threshold, collapse=collapse, nfkc=nfkc, hyphen=hyphen,
                 extract_tables=extract_tables, table_method=table_method.value,
                 template=template.name if template else "(none)")
    if available and not languages:
        st.warning("Select at least one installed language.")
        return None
    if not available and pdf_mode is not PDFMode.EMBEDDED_ONLY:
        # Without an engine only embedded PDF text can be read; don't offer a run that can only fail.
        return None
    return PipelineOptions(
        engine=engine,
        ocr=OCROptions(languages=tuple(languages) or ("eng",), psm=psm if supports_psm else None,
                       timeout_seconds=settings.ocr_timeout_seconds, low_confidence_threshold=float(threshold)),
        preprocess_preset=preset if preset != "custom" else "original",
        custom_steps=custom_steps,
        pdf_mode=pdf_mode,
        page_range=page_range,
        normalization=NormalizationOptions(collapse_spaces=collapse, unicode_form="NFKC" if nfkc else "NFC", join_hyphenated_lines=hyphen),
        extract_tables=extract_tables,
        table_method=table_method,
        template=template,
    )


def run_with_progress(doc: state.UploadedDoc, options: PipelineOptions) -> PipelineResult:
    bar = st.progress(0.0, text=f"{doc.name}: starting")

    def report(fraction: float, message: str) -> None:
        bar.progress(min(1.0, max(0.0, fraction)), text=f"{doc.name}: {message}")

    result = state.pipeline().process(doc.data, doc.name, options, progress=report)
    state.record_result(doc.file_id, result)
    bar.progress(1.0, text=f"{doc.name}: {'done' if result.succeeded else 'finished with errors'} in {result.duration_seconds:.1f}s")
    return result


def default_options() -> Optional[PipelineOptions]:
    """Processing options from the session's saved choices (or settings), without widgets.

    Used for one-click processing (dashboard, "process now" buttons). Returns None when no
    OCR engine with an installed language is available.
    """
    settings = state.settings()
    saved: dict = st.session_state["processing_options"]
    available, _ = engine_choices()
    if not available:
        return None
    engine_name = saved.get("engine") if saved.get("engine") in available else (
        settings.default_engine if settings.default_engine in available else available[0])
    engine = state.engine_factory().get(engine_name)
    installed = [lang for lang in engine.available_languages() if lang != "osd"]
    languages = [lang for lang in saved.get("languages", settings.language_list) if lang in installed] or installed[:1]
    if not engine.supports_multiple_languages:
        languages = languages[:1]
    if not languages:
        return None
    preset = saved.get("preset", "smart")
    return PipelineOptions(
        engine=engine_name,
        ocr=OCROptions(languages=tuple(languages), psm=saved.get("psm", 3) if engine.supports_psm else None,
                       timeout_seconds=settings.ocr_timeout_seconds,
                       low_confidence_threshold=float(saved.get("threshold", settings.low_confidence_threshold))),
        preprocess_preset=preset if preset in PRESETS else "smart",
        custom_steps=tuple(saved["custom_steps"]) if preset == "custom" and saved.get("custom_steps") else None,
        pdf_mode=PDFMode(saved.get("pdf_mode", "auto")),
    )


@st.cache_data(ttl=60, show_spinner=False)
def _health(settings_json: str, statuses_key: str, _statuses: list, has_osd: bool) -> list:
    from docintel.config.settings import Settings
    from docintel.ui import health

    return health.collect(Settings.model_validate_json(settings_json), _statuses, has_osd)


def health_items() -> list:
    """Dependency health (cached for 60 s; recomputed when settings or engine status change)."""
    settings = state.settings()
    statuses = state.engine_factory().statuses()
    has_osd = False
    tess = next((s for s in statuses if s.name == "tesseract" and s.available), None)
    if tess is not None:
        has_osd = "osd" in state.engine_factory().get("tesseract").available_languages()
    key = "|".join(f"{s.name}:{s.available}:{s.version}:{len(s.languages)}" for s in statuses)
    return _health(settings.model_dump_json(), key, statuses, has_osd)


HEALTH_BADGES: dict[str, StatusKind] = {"ready": "ok", "optional": "off", "setup": "warn", "unavailable": "bad", "error": "bad"}


def health_panel(compact: bool = False) -> None:
    for item in health_items():
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown(f"**{item.name}**")
            st.space("stretch")
            badge(item.display_label, HEALTH_BADGES[item.status])
        if not compact or item.status != "ready":
            st.caption(display_path(item.detail))


def low_confidence_html(ocr: OCRResult) -> str:
    """OCR words with low-confidence ones highlighted (all text HTML-escaped).

    Custom HTML is used here because no native element highlights individual words with a
    tooltip. Colours come from the active theme tokens.
    """
    t = theme.tokens()
    mark = (f"background:{t['low_bg']};color:{t['low_fg']};border-bottom:2px dotted {t['low_bd']};"
            "padding:0 2px;border-radius:2px")
    lines_html = []
    for _, block in groupby(ocr.words, key=lambda w: (w.block, w.paragraph, w.line)):
        parts = []
        for word in block:
            text = html.escape(word.text)
            if word.low_confidence:
                conf = f"{word.confidence:.0f}" if word.confidence is not None else "?"
                parts.append(f'<mark style="{mark}" title="confidence {conf}">{text}</mark>')
            else:
                parts.append(text)
        lines_html.append(" ".join(parts))
    box = (f"font-family:ui-monospace,Consolas,monospace;font-size:0.9rem;line-height:1.7;color:{t['ink']};"
           f"background:{t['raised']};border:1px solid {t['line']};border-radius:8px;padding:0.8rem;"
           "max-height:520px;overflow:auto;overflow-wrap:anywhere")
    return f'<div style="{box}">' + "<br>".join(lines_html) + "</div>"


StatusKind = Literal["ok", "warn", "bad", "info", "off"]
_BADGES: dict[str, tuple[str, str]] = {
    "ok": (":material/check_circle:", "green"),
    "warn": (":material/warning:", "orange"),
    "bad": (":material/error:", "red"),
    "info": (":material/info:", "blue"),
    "off": (":material/remove_circle_outline:", "gray"),
}


def badge(text: str, kind: StatusKind) -> None:
    """Native status badge: icon + label, so state is never conveyed by colour alone."""
    icon, color = _BADGES[kind]
    st.badge(text, icon=icon, color=color)


def page_header(title: str, subtitle: str) -> None:
    """Consistent page title block."""
    st.title(title, anchor=False)
    st.caption(subtitle)


@contextmanager
def card(title: str = "", hint: str = "", *, key: Optional[str] = None) -> Iterator[None]:
    """A bordered section with an optional title and one-line hint."""
    with st.container(border=True, key=key):
        if title:
            st.markdown(f"**{title}**" + (f" &nbsp;:gray[{hint}]" if hint else ""))
        yield


def empty_state(icon: str, title: str, text: str) -> None:
    with st.container(border=True):
        st.markdown(f"### {icon} {title}")
        st.caption(text)


# --- shared upload panel ------------------------------------------------------------------------
UploadKind = Literal["any", "image", "pdf"]
_IMAGE_EXTS = [e for e in SUPPORTED_EXTENSIONS if e != "pdf"]
SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"
SAMPLES = {"image": SAMPLE_DIR / "incident_report.png", "pdf": SAMPLE_DIR / "pdf" / "mixed.pdf"}


def _types(kind: UploadKind) -> list[str]:
    return {"any": list(SUPPORTED_EXTENSIONS), "image": _IMAGE_EXTS, "pdf": ["pdf"]}[kind]


def upload_panel(key: str, *, kind: UploadKind = "any", title: str = "Add documents",
                 show_sample: bool = True) -> list[state.UploadedDoc]:
    """Upload widget shared by every page that works on documents.

    Files are validated by content (not extension) and kept in session memory, so a file added
    on any page is available on all pages. Validation errors are shown right below the widget.
    Returns the documents added through this widget in the current run.
    """
    settings = state.settings()
    what = {"any": "Images (JPG, PNG, TIFF, BMP, WEBP) or PDFs", "image": "Images (JPG, PNG, TIFF, BMP, WEBP)",
            "pdf": "PDF documents"}[kind]
    with card(title, f"{what}, up to {settings.max_file_size_mb:.0f} MB each", key=f"upload_{key}"):
        files = st.file_uploader("Drag and drop files here, or browse", type=_types(kind), accept_multiple_files=True,
                                 key=f"uploader_{key}", label_visibility="collapsed")
        added = [state.register_upload(f.name, f.getvalue()) for f in files or []]
        sample_kind = "pdf" if kind == "pdf" else "image"
        sample = SAMPLES.get(sample_kind)
        row = st.container(horizontal=True, vertical_alignment="center")
        with row:
            st.caption(f"Limits: {settings.max_pdf_pages} PDF pages · {settings.max_image_dimension:,} px per side. "
                       "Checked by content, kept in memory for this session only.")
            if show_sample and sample is not None and sample.is_file():
                if st.button("Try a sample", key=f"sample_{key}", icon=":material/science:", type="tertiary",
                             help=f"Adds {sample.name} from data/samples"):
                    added.append(state.register_upload(sample.name, sample.read_bytes()))
                    st.session_state["last_added"] = added[-1].file_id
        for doc in added:
            if doc.error:
                st.error(f"**{doc.name}**: {doc.error}", icon=":material/block:")
            else:
                for warning in doc.info.warnings:
                    st.warning(f"**{doc.name}**: {warning}", icon=":material/warning:")
    return added


def document_picker(label: str, key: str, *, kind: Optional[str] = None, prefer: Iterable[str] = ()) -> Optional[state.UploadedDoc]:
    """Select one valid uploaded document (optionally only images or only PDFs)."""
    choices = state.valid_upload_choices(kind)
    if not choices:
        return None
    ids = list(choices)
    default = next((fid for fid in prefer if fid in ids), ids[-1])
    fid = st.selectbox(label, ids, index=ids.index(default), format_func=lambda i: choices[i], key=key)
    return state.uploads()[fid]
