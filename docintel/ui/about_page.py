"""About: what DocIntel does, privacy, component versions and licences."""

from __future__ import annotations

import importlib.metadata
import sys

import pandas as pd
import streamlit as st

from docintel import __version__
from docintel.ui.components import card, health_panel, page_header

COMPONENTS = [
    ("streamlit", "Apache-2.0"), ("PyMuPDF", "AGPL-3.0 / commercial"), ("opencv-python-headless", "Apache-2.0"),
    ("opencv-python", "Apache-2.0"), ("pillow", "MIT-CMU"), ("numpy", "BSD-3-Clause"), ("pandas", "BSD-3-Clause"),
    ("pydantic", "MIT"), ("fastapi", "MIT"), ("camelot-py", "MIT"), ("ocrmypdf", "MPL-2.0"), ("pikepdf", "MPL-2.0"),
    ("paddleocr", "Apache-2.0"), ("paddlepaddle", "Apache-2.0"),
]


def _version(dist: str) -> str:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return ""


def render() -> None:
    page_header("About", f"DocIntel {__version__} · local image-to-text and document intelligence")
    c1, c2 = st.columns([1.3, 1], gap="medium")
    with c1, card("What it does"):
        st.markdown(
            "- Extracts text from images and PDFs with **Tesseract** (optional **PaddleOCR**)\n"
            "- Uses a PDF's embedded text where it exists and OCRs only scanned pages\n"
            "- Improves recognition with optional preprocessing (deskew, denoise, thresholds...)\n"
            "- Extracts tables (CSV/XLSX) and template fields (JSON/CSV) without guessing missing values\n"
            "- Creates searchable PDFs with OCRmyPDF and Ghostscript"
        )
    with c2, card("Privacy"):
        st.markdown(
            ":material/lock: Everything runs on this computer. Documents stay in memory for this session and are "
            "never stored unless you press a save button. The server listens on 127.0.0.1 only and Streamlit usage "
            "statistics are turned off. Logs contain metadata only, never document text."
        )
    with card("System health", "checked on this computer"):
        health_panel()
        st.markdown("**Local AI assistant (Ollama)**")
        st.badge("Disabled", icon=":material/remove_circle_outline:", color="gray")
        st.caption("Not set up and no models are downloaded. Summaries, questions about documents and other "
                   "language-model features stay disabled; OCR and extraction never depend on them.")
    with card("Components", "installed versions and licences"):
        rows = [{"Component": "Python", "Version": sys.version.split()[0], "Licence": "PSF"}]
        rows += [{"Component": name, "Version": v, "Licence": lic} for name, lic in COMPONENTS if (v := _version(name))]
        rows += [{"Component": "Tesseract OCR (native)", "Version": "see System health", "Licence": "Apache-2.0"},
                 {"Component": "Ghostscript (native)", "Version": "see System health", "Licence": "AGPL-3.0"}]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Review the AGPL terms (PyMuPDF, Ghostscript) before distributing DocIntel or offering it as a network service.")
    with card("Documentation"):
        st.markdown("README.md (installation, usage, API) · docs/ARCHITECTURE.md · docs/EVALUATION.md · "
                    "docs/TROUBLESHOOTING.md · docs/LIMITATIONS.md, all in the project folder.")
        st.caption("OCR output can contain errors. Confidence values are engine-specific indicators, not probabilities.")
