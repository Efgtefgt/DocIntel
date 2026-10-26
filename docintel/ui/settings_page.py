"""Settings: engines, defaults, processing limits, resources, housekeeping and optional features.

Changes apply to this browser session. Permanent defaults use the existing mechanism: the
``.env`` file (see ``.env.example``). Documents are never persisted by settings.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pandas as pd
import streamlit as st

from docintel.core.security import display_path
from docintel.core.temp_files import cleanup_stale_temp_dirs, count_temp_dirs
from docintel.documents.searchable_pdf import check_searchable_pdf_support
from docintel.extraction.table_extractor import camelot_available
from docintel.ocr.tesseract_engine import clear_tesseract_caches, resolve_tesseract_cmd
from docintel.preprocessing.preprocessing_pipeline import PRESET_DESCRIPTIONS, PRESETS
from docintel.ui import state
from docintel.ui.components import badge, card, page_header
from docintel.ui.health import PADDLE_INSTALLER


def _engines(statuses) -> None:
    settings = state.settings()
    with card("OCR engines", "detected on this computer"):
        for s in statuses:
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(f"**{s.display_name}**")
                st.space("stretch")
                if s.available:
                    badge("Ready", "ok")
                elif "Disabled" in s.reason:
                    badge("Disabled", "off")
                else:
                    badge("Unavailable", "bad")
            st.caption(f"{s.version} · languages: {', '.join(s.languages[:12]) or 'none'}" if s.available else display_path(s.reason))
        cmd = resolve_tesseract_cmd(settings.tesseract_cmd)
        st.caption(f"Tesseract executable: {display_path(cmd or 'not found')} (DOCINTEL_TESSERACT_CMD in .env). "
                   f"Language data: {display_path(settings.tessdata_dir or 'Tesseract default folder')}.")
        if st.button("Re-check engines and installed languages", icon=":material/refresh:", key="set_recheck"):
            clear_tesseract_caches()
            state.update_settings()
            st.rerun()


def _form(statuses) -> None:
    settings = state.settings()
    saved = st.session_state["processing_options"]
    available = [s for s in statuses if s.available]
    with card("Defaults and limits", "applied to this session; use .env to make them permanent"):
        with st.form("defaults", border=False):
            t_defaults, t_limits, t_resources = st.tabs(["Defaults", "Processing limits", "CPU and memory"])
            with t_defaults:
                names = [s.name for s in available] or [settings.default_engine]
                engine = st.selectbox("Default OCR engine", names,
                                      index=names.index(settings.default_engine) if settings.default_engine in names else 0)
                all_langs = sorted({lang for s in available for lang in s.languages})
                langs = st.multiselect("Default languages", all_langs,
                                       default=[lang for lang in settings.language_list if lang in all_langs],
                                       help="Only language data installed on this computer is listed")
                presets = list(PRESETS)
                preset = st.selectbox("Default preprocessing", presets,
                                      index=presets.index(saved.get("preset", "smart")) if saved.get("preset", "smart") in presets else 0,
                                      format_func=lambda p: f"{p} · {PRESET_DESCRIPTIONS[p]}")
                threshold = st.slider("Low-confidence threshold", 0, 100, int(settings.low_confidence_threshold),
                                      help="Words below this engine-specific score are highlighted for review")
            with t_limits:
                c1, c2 = st.columns(2)
                ocr_timeout = c1.number_input("OCR timeout per page (s)", 5, 600, int(settings.ocr_timeout_seconds))
                doc_timeout = c2.number_input("Time limit per document (s)", 30, 7200, int(settings.document_timeout_seconds))
                st.caption(f"Upload limits come from .env: {settings.max_file_size_mb:.0f} MB per file, "
                           f"{settings.max_pdf_pages} PDF pages, {settings.max_image_dimension:,} px per image side.")
            with t_resources:
                try:
                    import psutil

                    vm = psutil.virtual_memory()
                    st.caption(f"System memory: {vm.available / 2**30:.1f} GB free of {vm.total / 2**30:.1f} GB · "
                               f"DocIntel process: {psutil.Process().memory_info().rss / 2**20:.0f} MB")
                except ImportError:
                    pass
                c1, c2 = st.columns(2)
                dpi = c1.slider("PDF render resolution (DPI)", 100, 400, settings.pdf_render_dpi, step=25,
                                help="300 DPI helps small text but uses about 2.3× the memory and time of 200 DPI")
                megapixels = c2.slider("Max image size (megapixels)", 5, 100, settings.max_image_pixels // 1_000_000,
                                       help="Larger images are rejected or rendered at lower resolution")
                workers = c1.slider("Parallel workers (batch and API)", 1, 4, settings.max_workers,
                                    help="1 = sequential, recommended on 8 GB RAM; each worker runs its own Tesseract process")
                variants = c2.slider("Max preprocessing variants to compare", 1, 6, settings.max_preprocessing_variants)
                attempts = c1.slider("Max OCR attempts per page (smart preprocessing)", 1, 4, settings.max_ocr_attempts,
                                     help="Extra attempts run only when the first result looks weak; each one is a full OCR run")
            submitted = st.form_submit_button("Apply settings", type="primary", icon=":material/check:")
        if submitted:
            state.update_settings(default_engine=engine, default_languages="+".join(langs) or settings.default_languages,
                                  low_confidence_threshold=float(threshold), pdf_render_dpi=dpi,
                                  max_image_pixels=megapixels * 1_000_000, ocr_timeout_seconds=float(ocr_timeout),
                                  document_timeout_seconds=float(doc_timeout), max_workers=workers,
                                  max_preprocessing_variants=variants, max_ocr_attempts=attempts)
            st.session_state["processing_options"] = {"preset": preset}
            st.success("Settings applied for this session.", icon=":material/check_circle:")


def _paddle_addon() -> None:
    """Installed app only: start the PaddleOCR add-on installer (a console window) on request."""
    if not PADDLE_INSTALLER.is_file() or importlib.util.find_spec("paddleocr") is not None:
        return
    with card("PaddleOCR engine", "optional download · deep-learning OCR for photos and difficult images"):
        st.caption("Reads phone photos and hard images better than Tesseract, but is slower and needs 1-2 GB more "
                   "memory. Downloads about 700 MB once (about 1.3 GB on disk). DocIntel closes while it installs "
                   "and reopens when it is done; afterwards choose PaddleOCR as the engine.")
        if st.button("Install PaddleOCR…", icon=":material/download:", key="set_paddle_install"):
            python = sys.executable.replace("pythonw.exe", "python.exe")  # console Python, so progress is visible
            subprocess.Popen([python, str(PADDLE_INSTALLER)], cwd=str(PADDLE_INSTALLER.parent),  # noqa: S603
                             creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
            st.info("An installer window has opened. Confirm there to start the download.", icon=":material/info:")


def render() -> None:
    page_header("Settings", "OCR engines, defaults, limits and housekeeping. Theme: use the selector in the sidebar "
                "(DOCINTEL_UI_THEME in .env sets the default).")
    statuses = state.engine_factory().statuses()
    _engines(statuses)
    _paddle_addon()
    _form(statuses)
    settings = state.settings()
    c1, c2 = st.columns(2, gap="medium")
    with c1, card("Temporary files", "normally deleted automatically"):
        st.caption(f"{count_temp_dirs(settings)} DocIntel temporary folder(s) present.")
        if st.button("Remove leftover temporary files", icon=":material/delete_sweep:", key="set_cleanup"):
            removed = cleanup_stale_temp_dirs(settings, older_than_seconds=300)
            st.success(f"Removed {removed} folder(s) older than 5 minutes.")
    with c2, card("Session data", "held in memory only"):
        st.caption(f"{len(state.uploads())} uploaded file(s) and {len(state.results())} result(s) in memory.")
        if st.button("Clear uploaded files and results", icon=":material/clear_all:", key="set_clear"):
            state.clear_documents()
            st.success("Session documents cleared.")

    pdf_support = check_searchable_pdf_support(settings)
    features = [
        {"Feature": "Camelot table extraction", "Available": camelot_available(),
         "Notes": "" if camelot_available() else "pip install -r requirements-optional.txt"},
        {"Feature": "Searchable PDF (OCRmyPDF + Ghostscript)", "Available": pdf_support.available, "Notes": pdf_support.reason},
        {"Feature": "PaddleOCR engine", "Available": any(s.name == "paddle" and s.available for s in statuses),
         "Notes": next((s.reason for s in statuses if s.name == "paddle"), "")},
        {"Feature": "REST API (FastAPI)", "Available": importlib.util.find_spec("fastapi") is not None,
         "Notes": "python -m uvicorn docintel.api.main:app --host 127.0.0.1 --port 8000"},
        {"Feature": "Local AI assistant (Ollama)", "Available": False,
         "Notes": "Disabled: not set up and no models downloaded. DocIntel never calls a language model."},
    ]
    with card("Optional features"):
        st.dataframe(pd.DataFrame(features), hide_index=True,
                     column_config={"Available": st.column_config.CheckboxColumn("Available", width="small")})
