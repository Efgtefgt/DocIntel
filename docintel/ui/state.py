"""Session state for the Streamlit app.

Uploaded files and results live only in this browser session's memory. Nothing is
written to disk unless the user presses an explicit "save" button.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import streamlit as st

from docintel.config.settings import Settings, get_settings
from docintel.core.exceptions import InputValidationError
from docintel.core.file_validator import validate_upload
from docintel.core.pipeline import DocumentPipeline, PipelineResult
from docintel.core.result_models import FileInfo
from docintel.ocr.engine_factory import EngineFactory

if TYPE_CHECKING:
    from docintel.batch.runner import BatchRunner



@dataclass
class UploadedDoc:
    file_id: str
    name: str
    data: bytes
    info: Optional[FileInfo] = None
    error: Optional[str] = None


@dataclass
class SessionStats:
    files_processed: int = 0
    pages_processed: int = 0
    succeeded: int = 0
    failed: int = 0
    activity: list[dict] = field(default_factory=list)


def _init() -> None:
    ss = st.session_state
    ss.setdefault("settings", get_settings())
    ss.setdefault("uploads", {})       # file_id -> UploadedDoc
    ss.setdefault("results", {})       # file_id -> PipelineResult
    ss.setdefault("stats", SessionStats())
    ss.setdefault("processing_options", {})
    ss.setdefault("corrections", {})


def settings() -> Settings:
    _init()
    return st.session_state["settings"]


def update_settings(**changes: object) -> None:
    st.session_state["settings"] = settings().model_copy(update=changes)
    # Engines depend on settings; drop the cached factory so it is rebuilt lazily.
    factory: Optional[EngineFactory] = st.session_state.pop("engine_factory", None)
    if factory is not None:
        factory.close_all()


def engine_factory() -> EngineFactory:
    _init()
    if "engine_factory" not in st.session_state:
        st.session_state["engine_factory"] = EngineFactory(settings())
    return st.session_state["engine_factory"]


def batch_runner() -> "BatchRunner":
    """The process-wide batch worker (shared by all sessions; tests may put their own in session state)."""
    _init()
    if "batch_runner" in st.session_state:
        return st.session_state["batch_runner"]
    from docintel.batch.runner import get_runner

    return get_runner(settings())


def pipeline() -> DocumentPipeline:
    return DocumentPipeline(settings(), engine_factory())


def uploads() -> dict[str, UploadedDoc]:
    _init()
    return st.session_state["uploads"]


def results() -> dict[str, PipelineResult]:
    _init()
    return st.session_state["results"]


def stats() -> SessionStats:
    _init()
    return st.session_state["stats"]


def file_id_for(name: str, data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16] + "_" + str(len(data))


def register_upload(name: str, data: bytes) -> UploadedDoc:
    """Validate an upload once and keep it in session memory."""
    fid = file_id_for(name, data)
    store = uploads()
    if fid not in store:
        doc = UploadedDoc(file_id=fid, name=name, data=data)
        try:
            doc.info = validate_upload(data, name, settings()).info
        except InputValidationError as exc:
            doc.error = exc.user_message
        store[fid] = doc
    return store[fid]


def activity_entry(result: PipelineResult, file_id: str = "") -> dict:
    """One dashboard/activity row describing a processed file (metadata only, no text)."""
    doc_pages = result.document.pages if result.document else []
    confidences = [p.ocr.mean_confidence for p in doc_pages if p.ocr and p.ocr.mean_confidence is not None]
    return {
        "file_id": file_id,
        "time": time.strftime("%H:%M:%S"),
        "file": result.filename,
        "type": result.document.file_info.format if result.document else "-",
        "status": "success" if result.succeeded else "failed",
        "pages": len(doc_pages),
        "embedded pages": sum(1 for p in doc_pages if p.succeeded and p.source.value == "embedded_text"),
        "ocr pages": sum(1 for p in doc_pages if p.succeeded and p.source.value == "ocr"),
        "failed pages": len(result.document.failed_pages) if result.document else 0,
        "engine": next((p.ocr.engine for p in doc_pages if p.ocr), "-"),
        "mean confidence": round(sum(confidences) / len(confidences), 1) if confidences else None,
        "seconds": round(result.duration_seconds, 2),
        "first error": result.errors[0].message if result.errors else "",
    }


def add_to_stats(s: SessionStats, result: PipelineResult, file_id: str = "") -> None:
    s.files_processed += 1
    s.pages_processed += len(result.document.pages) if result.document else 0
    if result.succeeded:
        s.succeeded += 1
    else:
        s.failed += 1
    s.activity.insert(0, activity_entry(result, file_id))
    del s.activity[50:]


def record_result(file_id: str, result: PipelineResult) -> None:
    results()[file_id] = result
    corrections().pop(file_id, None)  # a new result invalidates edits made to the previous one
    add_to_stats(stats(), result, file_id)
    st.session_state["last_processed"] = file_id


def clear_documents() -> None:
    st.session_state["uploads"] = {}
    st.session_state["results"] = {}
    st.session_state["corrections"] = {}
    for key in [k for k in st.session_state if str(k).startswith(("cmp_", "tables_", "structured_", "searchable_"))]:
        del st.session_state[key]


# Page files registered with st.navigation in app.py (paths relative to app.py).
PAGE_FILES = {
    "dashboard": "app_pages/dashboard.py",
    "image_ocr": "app_pages/image_ocr.py",
    "pdf_tools": "app_pages/pdf_tools.py",
    "tables": "app_pages/tables.py",
    "structured": "app_pages/structured.py",
    "batch": "app_pages/batch.py",
    "intelligence": "app_pages/intelligence.py",
    "results": "app_pages/results.py",
    "enhancement": "app_pages/enhancement.py",
    "settings": "app_pages/settings.py",
    "about": "app_pages/about.py",
}


def go(page: str, *, focus: Optional[str] = None) -> None:
    """Switch to a page (optionally selecting a document there first)."""
    if focus:
        st.session_state["last_processed"] = focus
        st.session_state["last_added"] = focus
    st.switch_page(PAGE_FILES[page])


def corrections() -> dict[str, dict[int, str]]:
    """User-edited text per file and page, kept apart from the OCR output."""
    _init()
    return st.session_state["corrections"]


def processed_choices() -> dict[str, str]:
    """Map file_id -> label for files that have results."""
    docs = uploads()
    return {fid: docs[fid].name if fid in docs else res.filename for fid, res in results().items()}


def valid_upload_choices(kind: Optional[str] = None) -> dict[str, str]:
    return {
        fid: doc.name
        for fid, doc in uploads().items()
        if doc.info is not None and (kind is None or doc.info.kind.value == kind)
    }
