"""PDF tools: inspect PDFs, process page ranges, and create searchable PDFs."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from docintel.core.exceptions import DocIntelError
from docintel.core.security import display_path
from docintel.documents.pdf_reader import analyze_page, open_pdf, read_metadata
from docintel.documents.searchable_pdf import OCRMode, check_searchable_pdf_support, create_searchable_pdf
from docintel.export.file_naming import export_filename
from docintel.export.txt_exporter import document_to_txt_bytes
from docintel.ui import state
from docintel.ui.components import badge, card, document_picker, page_header, processing_options_form, run_with_progress, upload_panel
from docintel.ui.key_details import key_details_panel

MODE_LABELS = {
    OCRMode.SKIP_TEXT: "Skip pages that already have text",
    OCRMode.REDO_OCR: "Redo OCR (replace old OCR text, keep digital text)",
    OCRMode.FORCE_OCR: "Force OCR on every page (rasterizes pages)",
}


@st.cache_data(max_entries=8, show_spinner="Analysing pages...")
def _analyze(file_id: str, data: bytes) -> tuple[dict, list[dict]]:
    with open_pdf(data) as doc:
        meta = read_metadata(doc)
        rows = []
        for page in doc:
            info = analyze_page(page)
            rows.append({"Page": info.page_number, "Text found": "Embedded text" if not info.needs_ocr else "Needs OCR",
                         "Characters": info.meaningful_chars, "Image coverage": round(info.image_coverage * 100),
                         "Reason": info.reason})
    return meta, rows


def _overview(fid: str, doc: state.UploadedDoc) -> None:
    meta, rows = _analyze(fid, doc.data)
    need = sum(r["Text found"] == "Needs OCR" for r in rows)
    with st.container(horizontal=True, gap="medium"):
        st.metric("Pages", meta["page_count"], border=True, icon=":material/auto_stories:")
        st.metric("Embedded-text pages", len(rows) - need, border=True, icon=":material/text_snippet:",
                  help="Pages whose own text layer is usable; read directly without OCR")
        st.metric("Pages needing OCR", need, border=True, icon=":material/document_scanner:",
                  help="Scanned pages or pages without usable embedded text")
        st.metric("Size", f"{doc.info.size_bytes / 1024:.0f} KB", border=True, icon=":material/inventory_2:")
    if need == 0:
        st.caption("This PDF already has embedded text on every page; processing reads it directly (fast, no OCR).")
    elif need == len(rows):
        st.caption("Every page looks scanned, so all pages will go through OCR.")
    else:
        st.caption("Mixed PDF: embedded text is used where it exists and OCR runs only on the scanned pages.")
    c1, c2 = st.columns([1, 2], gap="medium")
    with c1, card("Metadata"):
        shown = {k.replace("_", " ").capitalize(): str(v) for k, v in meta.items() if v not in ("", None)}
        st.table(pd.DataFrame(shown.items(), columns=["Field", "Value"]).set_index("Field"))
    with c2, card("Page analysis", "how each page will be read"):
        st.dataframe(pd.DataFrame(rows), hide_index=True, height=min(360, 40 + 35 * len(rows)),
                     column_config={"Image coverage": st.column_config.ProgressColumn("Image coverage", min_value=0,
                                                                                      max_value=100, format="%d%%")})


def _process(fid: str, doc: state.UploadedDoc) -> None:
    with card("Extract text", "choose pages and how they are read"):
        options = processing_options_form("pdf", show_template=False)
        go = st.button("Process selected pages", type="primary", icon=":material/play_arrow:",
                       disabled=options is None, key="pdf_process")
    if go and options is not None:
        result = run_with_progress(doc, options)
        for error in result.errors:
            st.error(f"{'Page ' + str(error.page_number) if error.page_number else error.stage}: {error.message}",
                     icon=":material/error:")
    result = state.results().get(fid)
    if not (result and result.document):
        return
    with card("Latest result", f"{len(result.document.pages)} page(s) processed in {result.duration_seconds:.1f} s"):
        for page in result.document.pages:
            source = page.source.value.replace("_", " ")
            label = f"Page {page.page_number} · {source}" + (" · failed" if page.error else "")
            with st.expander(label, icon=":material/error:" if page.error else ":material/description:"):
                if page.error:
                    st.error(page.error.message)
                st.code(page.normalized_text or "(no text)", language=None, wrap_lines=True)
        with st.container(horizontal=True):
            st.download_button("Text by page (TXT)", document_to_txt_bytes(result.document),
                               export_filename(doc.name, "text", "txt"), "text/plain", icon=":material/download:",
                               key="pdf_dl_txt", on_click="ignore")
            st.page_link(state.PAGE_FILES["results"], label="Review in OCR results", icon=":material/visibility:")
    key_details_panel("pdf", [fid])


def _searchable(fid: str, doc: state.UploadedDoc) -> None:
    with card("Searchable PDF", "adds an invisible, selectable text layer with OCRmyPDF"):
        support = check_searchable_pdf_support(state.settings())
        if not support.available:
            badge("Unavailable", "bad")
            st.caption(f"{display_path(support.reason)} Searchable PDFs need the OCRmyPDF package plus the native Tesseract and "
                       "Ghostscript programs (README, section 11).")
            return
        st.caption("Needs OCRmyPDF, Tesseract and Ghostscript, all detected on this computer. Long documents can take a "
                   "few seconds per page.")
        c1, c2, c3 = st.columns([2, 1, 1])
        mode = c1.selectbox("Mode", list(OCRMode), format_func=lambda m: MODE_LABELS[m], key="pdf_sp_mode",
                            help="Skip: OCR only pages without text. Redo: remove an existing OCR layer and OCR again; "
                                 "digital text is kept. Force: rasterize and OCR everything.")
        deskew = c2.checkbox("Deskew pages", key="pdf_sp_deskew",
                             help="Needs scans of roughly 200 DPI or more; low-resolution pages may not be corrected.")
        rotate = c3.checkbox("Auto-rotate pages", key="pdf_sp_rotate", help="Uses Tesseract orientation data (osd)")
        languages = st.session_state["processing_options"].get("languages") or state.settings().language_list
        st.caption(f"OCR languages: {'+'.join(languages)} (set in the options above).")
        if st.button("Create searchable PDF", icon=":material/find_in_page:", key="pdf_sp_create"):
            with st.spinner("Running OCRmyPDF..."):
                try:
                    st.session_state[f"searchable_{fid}"] = create_searchable_pdf(
                        doc.data, state.settings(), languages=languages, mode=mode, deskew=deskew, rotate_pages=rotate)
                except DocIntelError as exc:
                    st.error(exc.user_message, icon=":material/error:")
        output = st.session_state.get(f"searchable_{fid}")
        if output:
            st.success(f"Searchable PDF ready ({len(output) / 1024:.0f} KB).", icon=":material/check_circle:")
            st.download_button("Download searchable PDF", output, export_filename(doc.name, "searchable", "pdf"),
                               "application/pdf", icon=":material/download:", key="pdf_sp_download", on_click="ignore")


def render() -> None:
    page_header("PDF tools", "Read embedded text, OCR scanned pages, process page ranges and create searchable PDFs.")
    upload_panel("pdf_tools", kind="pdf", title="Add PDFs")
    doc = document_picker("PDF document", "pdf_doc", kind="pdf",
                          prefer=[st.session_state.get("last_added", ""), st.session_state.get("last_processed", "")])
    if doc is None:
        st.info("Add a PDF above to inspect and process it.", icon=":material/info:")
        return
    _overview(doc.file_id, doc)
    _process(doc.file_id, doc)
    _searchable(doc.file_id, doc)
