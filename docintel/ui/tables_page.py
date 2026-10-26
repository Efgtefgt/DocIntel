"""Table extraction: detect, preview and export tables as CSV/XLSX."""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

from docintel.export.csv_exporter import table_to_csv_bytes
from docintel.export.excel_exporter import tables_to_xlsx_bytes
from docintel.export.file_naming import export_filename
from docintel.extraction.table_extractor import table_with_header
from docintel.ui import state
from docintel.ui.components import (
    badge,
    card,
    document_picker,
    page_header,
    page_image,
    processing_options_form,
    run_with_progress,
    upload_panel,
)


def _results(fid: str, doc: state.UploadedDoc, header: bool) -> None:
    result = st.session_state.get(f"tables_{fid}")
    if result is None:
        return
    failed = [e for e in result.errors if e.stage in ("tables", "validation", "page_processing")]
    if failed:
        with card("Extraction failed"):
            badge("Failed", "bad")
            for error in failed:
                st.error(f"{error.stage}: {error.message}", icon=":material/error:")
        return
    if not result.tables:
        with card("No table found"):
            badge("No table structure detected", "off")
            st.caption("No table structure could be determined reliably, so nothing was guessed or reconstructed. "
                       "Scanned tables work best with clear ruling lines; the recognised text is shown below.")
            if result.document:
                st.code(result.document.combined_text(), language=None, wrap_lines=True)
        return

    tables = [table_with_header(t) if header else t for t in result.tables]
    uncertain = sum(t.uncertain for t in tables)
    with st.container(horizontal=True, vertical_alignment="center"):
        if uncertain:
            badge(f"{len(tables)} table(s) found · {uncertain} need review", "warn")
        else:
            badge(f"{len(tables)} table(s) found", "ok")
        st.space("stretch")
        st.download_button("All tables (XLSX)", tables_to_xlsx_bytes(tables), export_filename(doc.name, "tables", "xlsx"),
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           icon=":material/download:", key=f"tbl_xlsx_{fid}", on_click="ignore")
    for table in tables:
        method = table.method + (f" · quality {table.quality_score}" if table.quality_score is not None else "")
        with card(f"Table {table.table_index} · page {table.page_number}", method):
            if table.uncertain:
                badge("Uncertain structure, review before use", "warn")
            else:
                badge("Structure detected", "ok")
            for note in table.notes:
                st.caption(note)
            st.dataframe(table.to_dataframe(), hide_index=True)
            st.download_button(f"CSV · table {table.table_index}", table_to_csv_bytes(table),
                               export_filename(doc.name, f"table{table.table_index}", "csv"), "text/csv",
                               key=f"csv_{fid}_{table.table_index}", icon=":material/download:", on_click="ignore")


def render() -> None:
    page_header("Table extraction", "Digital PDFs use the PDF's own text and lines (PyMuPDF or Camelot). Tables on scans are "
                "rebuilt from OCR word positions and always flagged for review.")
    upload_panel("tables")
    doc = document_picker("Document", "tbl_doc", prefer=[st.session_state.get("last_added", ""),
                                                        st.session_state.get("last_processed", "")])
    if doc is None:
        st.info("Add an image or PDF above to look for tables.", icon=":material/info:")
        return
    fid = doc.file_id
    left, right = st.columns([1, 1.4], gap="medium")
    with left, card("Source", f"page 1 of {doc.info.page_count}"):
        st.image(page_image(doc.data, doc.info.kind.value, 1), width="stretch", alt=f"First page of {doc.name}")
    with right, card("Detect tables"):
        header = st.checkbox("Use first row as column names", value=True, key="tbl_header")
        with st.expander("OCR and page options", icon=":material/tune:"):
            options = processing_options_form("tables", show_tables=False, show_template=False)
        st.caption("Ruled tables on scans are detected from their lines; the lines are hidden from OCR so they are not "
                   "read as characters. Without lines, columns are inferred from spacing.")
        go = st.button("Detect tables", type="primary", icon=":material/table_chart:", disabled=options is None,
                       key="tbl_detect")
    if go and options is not None:
        st.session_state[f"tables_{fid}"] = run_with_progress(doc, replace(options, extract_tables=True))
    _results(fid, doc, header)
