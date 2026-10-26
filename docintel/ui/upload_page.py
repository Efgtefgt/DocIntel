"""Image OCR: upload, validate, choose processing options and run OCR on one or more files."""

from __future__ import annotations

import streamlit as st

from docintel.core.result_models import FileKind
from docintel.ui import state
from docintel.ui.components import badge, card, page_header, page_image, processing_options_form, run_with_progress, upload_panel


def _file_row(doc: state.UploadedDoc, index: int) -> None:
    info = doc.info
    with st.container(horizontal=True, vertical_alignment="center"):
        with st.container():
            st.markdown(f"**{doc.name}**")
            if info is None:
                st.caption(doc.error or "Rejected")
            elif info.kind is FileKind.IMAGE:
                frames = f" · {info.page_count} frames" if info.page_count > 1 else ""
                st.caption(f"{info.format} · {info.width} × {info.height} px{frames} · {info.size_bytes / 1024:.0f} KB")
            else:
                st.caption(f"PDF · {info.page_count} page(s) · {info.size_bytes / 1024:.0f} KB")
        st.space("stretch")
        if info is None:
            badge("Rejected", "bad")
        elif doc.file_id in state.results():
            result = state.results()[doc.file_id]
            badge("Processed" if result.succeeded else "Processed with errors", "ok" if result.succeeded else "warn")
            if st.button("Open", key=f"ocr_open_{index}", icon=":material/visibility:", type="tertiary"):
                state.go("results", focus=doc.file_id)
        else:
            badge("Ready", "info")
        if st.button("Remove", key=f"ocr_remove_{index}", icon=":material/close:", type="tertiary",
                     help="Remove this file from the session"):
            state.uploads().pop(doc.file_id, None)
            state.results().pop(doc.file_id, None)
            st.rerun()
    if info is not None:
        with st.expander("Preview", icon=":material/image:"):
            try:
                st.image(page_image(doc.data, info.kind.value, 1), caption="Page 1", width="stretch")
            except Exception as exc:  # a preview must never break the page
                st.warning(f"Preview unavailable ({type(exc).__name__}).")


def render() -> None:
    page_header("Image OCR", "Upload images or PDFs, choose how to process them, then run OCR. "
                "Multiple files are processed one after another to keep memory use low.")
    upload_panel("image_ocr")
    docs = list(state.uploads().values())
    if not docs:
        return
    valid = [d for d in docs if d.info is not None]
    with card("Files in this session", f"{len(valid)} valid · {len(docs) - len(valid)} rejected"):
        for i, doc in enumerate(docs):
            _file_row(doc, i)
    if not valid:
        return

    with card("Processing options", "remembered for this session"):
        options = processing_options_form("upload")
    pending = [d for d in valid if d.file_id not in state.results()]
    targets = st.segmented_control("Process", ["pending", "all"], key="ocr_targets", default="pending" if pending else "all",
                                   format_func=lambda v: f"New files only ({len(pending)})" if v == "pending" else f"All files ({len(valid)})",
                                   required=True)
    selected = pending if targets == "pending" else valid
    st.caption("Each page is preprocessed, recognised and normalised; PDFs use embedded text where it exists. "
               "Results open in **OCR results** with the page image next to the text.")
    clicked = st.button(f"Process {len(selected)} file(s)", type="primary", icon=":material/play_arrow:",
                        disabled=options is None or not selected, key="ocr_process")
    if clicked and options is not None:
        with card("Progress"):
            outcomes = [(doc, run_with_progress(doc, options)) for doc in selected]
            ok = sum(r.succeeded for _, r in outcomes)
            if ok == len(outcomes):
                st.success(f"Processed {ok} file(s).", icon=":material/check_circle:")
            else:
                st.warning(f"{ok} of {len(outcomes)} file(s) processed without errors.", icon=":material/warning:")
            for doc, result in outcomes:
                for error in result.errors[:5]:
                    where = f"page {error.page_number}" if error.page_number else error.stage
                    st.error(f"**{doc.name}** ({where}): {error.message}", icon=":material/error:")
            st.page_link(state.PAGE_FILES["results"], label="Open OCR results", icon=":material/arrow_forward:")
