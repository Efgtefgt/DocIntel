"""Image OCR: upload, validate, choose processing options and run OCR on one or more files.

Built to stay short with many files: the files are one compact, scrolling table (actions act on
the file chosen below it), the options fold into one summary line, and processing shows a single
overall progress bar followed by a results table.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from docintel.core.pipeline import PipelineOptions, PipelineResult
from docintel.core.result_models import FileKind
from docintel.ui import state
from docintel.ui.components import card, engine_choices, page_header, page_image, processing_options_form, upload_panel
from docintel.ui.key_details import key_details_panel

MANY_FILES = 10        # suggest Batch processing from this many files
TABLE_ROW_PX = 35      # st.dataframe row height, used to cap the table at about 8 rows


def _status(doc: state.UploadedDoc) -> str:
    if doc.info is None:
        return "✗ rejected"
    result = state.results().get(doc.file_id)
    if result is None:
        return "ready"
    return "✓ processed" if result.succeeded else "⚠ processed with errors"


def _details(doc: state.UploadedDoc) -> str:
    info = doc.info
    if info is None:
        return doc.error or "Rejected"
    if info.kind is FileKind.IMAGE:
        frames = f" · {info.page_count} frames" if info.page_count > 1 else ""
        return f"{info.format} · {info.width} × {info.height} px{frames}"
    return f"PDF · {info.page_count} page(s)"


def _files(docs: list[state.UploadedDoc]) -> None:
    valid = sum(d.info is not None for d in docs)
    done = sum(d.file_id in state.results() for d in docs)
    with card("Files in this session", f"{len(docs)} file(s) · {valid} valid · {done} processed"):
        frame = pd.DataFrame([{"#": i + 1, "File": d.name, "Status": _status(d), "Details": _details(d),
                               "Size (KB)": round(len(d.data) / 1024)} for i, d in enumerate(docs)])
        st.dataframe(frame, hide_index=True, height=min(len(docs), 8) * TABLE_ROW_PX + 38, column_config={
            "#": st.column_config.NumberColumn("#", width="small"),
            "Status": st.column_config.TextColumn("Status", help="ready · ✓ processed · ⚠ processed with errors · "
                                                  "✗ rejected (see Details)"),
        })
        ids = [d.file_id for d in docs]
        labels = {d.file_id: f"{i + 1}. {d.name}" for i, d in enumerate(docs)}
        if st.session_state.get("ocr_file_pick") not in labels:  # file removed or list changed
            last = st.session_state.get("last_added", "")
            st.session_state["ocr_file_pick"] = last if last in labels else ids[0]
        with st.container(horizontal=True, vertical_alignment="bottom", wrap=True):
            fid = st.selectbox("File", ids, key="ocr_file_pick", format_func=lambda i: labels.get(i, str(i)),
                               help="Choose a file for Preview, Open or Remove")
            doc = state.uploads()[fid]
            preview = st.toggle("Preview", key="ocr_preview", disabled=doc.info is None)
            if st.button("Open in OCR results", key="ocr_open", icon=":material/visibility:",
                         disabled=fid not in state.results(), help="Available after processing"):
                state.go("results", focus=fid)
            if st.button("Remove", key="ocr_remove", icon=":material/close:", help="Remove this file from the session"):
                _remove([fid])
            if len(docs) > 1 and st.button("Remove all", key="ocr_remove_all", icon=":material/delete_sweep:",
                                           type="tertiary", help="Remove every file and result from this session"):
                _remove(ids)
        if preview and doc.info is not None:
            try:
                st.image(page_image(doc.data, doc.info.kind.value, 1), caption=f"{doc.name}, page 1", width=520)
            except Exception as exc:  # a preview must never break the page
                st.warning(f"Preview unavailable ({type(exc).__name__}).")


def _remove(file_ids: list[str]) -> None:
    for fid in file_ids:
        state.uploads().pop(fid, None)
        state.results().pop(fid, None)
    st.session_state.pop("ocr_file_pick", None)
    st.rerun()


def _options_summary() -> str:
    saved: dict = st.session_state.get("processing_options", {})
    settings = state.settings()
    parts = [saved.get("engine", settings.default_engine), "+".join(saved.get("languages") or settings.language_list),
             f"{saved.get('preset', 'smart')} preprocessing"]
    if saved.get("page_range"):
        parts.append(f"pages {saved['page_range']}")
    if saved.get("extract_tables"):
        parts.append("tables")
    if saved.get("template") not in (None, "(none)"):
        parts.append(f"template {saved['template']}")
    return " · ".join(parts)


def _process(selected: list[state.UploadedDoc], options: PipelineOptions) -> list[tuple[state.UploadedDoc, PipelineResult]]:
    """Process the files one after another with one overall progress bar."""
    bar = st.progress(0.0, text="Starting")
    outcomes = []
    total = len(selected)
    for index, doc in enumerate(selected):
        def report(fraction: float, message: str, index: int = index, name: str = doc.name) -> None:
            overall = (index + min(1.0, max(0.0, fraction))) / total
            bar.progress(overall, text=f"File {index + 1} of {total}: {name} · {message}")

        result = state.pipeline().process(doc.data, doc.name, options, progress=report)
        state.record_result(doc.file_id, result)
        outcomes.append((doc, result))
    bar.progress(1.0, text=f"Finished {total} file(s)")
    return outcomes


def _outcome(file_ids: list[str]) -> None:
    """Summary of the last run (kept across reruns until the next run)."""
    outcomes = [(state.uploads()[f], state.results()[f]) for f in file_ids
                if f in state.uploads() and f in state.results()]
    if not outcomes:
        return
    ok = sum(r.succeeded for _, r in outcomes)
    if ok == len(outcomes):
        st.success(f"Processed {ok} file(s).", icon=":material/check_circle:")
    else:
        st.warning(f"{ok} of {len(outcomes)} file(s) processed without errors. Details below.", icon=":material/warning:")
    rows = []
    for doc, result in outcomes:
        problems = "; ".join(f"{'page ' + str(e.page_number) if e.page_number else e.stage}: {e.message}"
                             for e in result.errors[:3])
        rows.append({"File": doc.name, "Result": "✓ done" if result.succeeded else "⚠ check",
                     "Pages": len(result.document.pages) if result.document else 0,
                     "Seconds": round(result.duration_seconds, 1), "Problems": problems})
    st.dataframe(pd.DataFrame(rows), hide_index=True, height=min(len(rows), 6) * TABLE_ROW_PX + 38)
    for doc, result in outcomes:  # errors also as messages, so screen readers announce them
        for error in result.errors[:2]:
            where = f"page {error.page_number}" if error.page_number else error.stage
            st.error(f"**{doc.name}** ({where}): {error.message}", icon=":material/error:")
    st.page_link(state.PAGE_FILES["results"], label="Open OCR results", icon=":material/arrow_forward:")


def render() -> None:
    page_header("Image OCR", "Upload images or PDFs, choose how to process them, then run OCR. "
                "Multiple files are processed one after another to keep memory use low.")
    upload_panel("image_ocr")
    docs = list(state.uploads().values())
    if not docs:
        return
    _files(docs)
    valid = [d for d in docs if d.info is not None]
    if not valid:
        return
    if len(valid) >= MANY_FILES:
        st.info("Many files? **Batch processing** runs them in the background, can pause and resume, and gives "
                "you one ZIP with a summary.", icon=":material/stacks:")
        st.page_link(state.PAGE_FILES["batch"], label="Go to Batch processing", icon=":material/arrow_forward:")

    available, _ = engine_choices()
    with st.expander(f"Processing options: {_options_summary()}", icon=":material/tune:", expanded=not available):
        options: Optional[PipelineOptions] = processing_options_form("upload")
    pending = [d for d in valid if d.file_id not in state.results()]
    with st.container(horizontal=True, vertical_alignment="bottom", wrap=True):
        targets = st.segmented_control("Process", ["pending", "all"], key="ocr_targets",
                                       default="pending" if pending else "all", required=True,
                                       format_func=lambda v: f"New files only ({len(pending)})" if v == "pending"
                                       else f"All files ({len(valid)})")
        selected = pending if targets == "pending" else valid
        clicked = st.button(f"Process {len(selected)} file(s)", type="primary", icon=":material/play_arrow:",
                            disabled=options is None or not selected, key="ocr_process")
    st.caption("Each page is preprocessed, recognised and normalised; PDFs use embedded text where it exists. "
               "Results open in **OCR results** with the page image next to the text.")
    if clicked and options is not None:
        with card("Progress"):
            outcomes = _process(selected, options)
        st.session_state["ocr_last_run"] = [doc.file_id for doc, _ in outcomes]
        st.rerun()  # redraw the file table above with the new statuses
    if st.session_state.get("ocr_last_run"):
        with card("Last run"):
            _outcome(st.session_state["ocr_last_run"])
    key_details_panel("ocr", [d.file_id for d in valid])
