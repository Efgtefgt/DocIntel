"""Dashboard: header, session KPIs, upload panel, quick actions, system health, recent activity."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from docintel.documents.searchable_pdf import check_searchable_pdf_support
from docintel.export.file_naming import export_filename
from docintel.export.txt_exporter import document_to_txt_bytes
from docintel.ui import state, theme
from docintel.ui.components import (
    badge,
    card,
    default_options,
    engine_choices,
    health_panel,
    run_with_progress,
    upload_panel,
)


def _header() -> None:
    with st.container(horizontal=True, vertical_alignment="bottom"):
        with st.container():
            st.title("Dashboard", anchor=False)
            st.caption("This session's document processing at a glance. Files are kept in memory only, never stored.")
        available, _ = engine_choices()
        if len(available) > 1:
            saved = st.session_state["processing_options"]
            current = saved.get("engine", state.settings().default_engine)
            engine = st.selectbox("OCR engine", available, index=available.index(current) if current in available else 0,
                                  key="dash_engine", width=200, help="Engine used for new processing on every page")
            saved["engine"] = engine
        if st.button("Settings", icon=":material/settings:", key="dash_settings"):
            state.go("settings")


def _kpis() -> None:
    # Figures describe each file's latest result (re-processing a file replaces its result).
    results = [r for r in state.results().values() if r.document is not None]
    pages = sum(len(r.document.pages) for r in results)
    succeeded = sum(r.succeeded for r in results)
    seconds = sum(r.duration_seconds for r in results)
    ocr_conf = [p.ocr.mean_confidence for r in results for p in r.document.pages if p.ocr and p.ocr.mean_confidence is not None]
    cols = st.columns(5)
    cols[0].metric("Files processed", len(results), border=True, icon=":material/description:",
                   help="Files with a result in this browser session")
    cols[1].metric("Pages processed", pages, border=True, icon=":material/auto_stories:")
    cols[2].metric("OK / failed", f"{succeeded} / {len(results) - succeeded}" if results else "—",
                   border=True, icon=":material/rule:", help="Files processed without errors / with errors")
    cols[3].metric("Time per page", f"{seconds / pages:.2f} s" if pages else "—", border=True,
                   icon=":material/schedule:", help="Average processing time per page, over the latest results")
    cols[4].metric("OCR confidence", f"{sum(ocr_conf) / len(ocr_conf):.0f}" if ocr_conf else "—", border=True,
                   icon=":material/verified:",
                   help="Average engine-specific word confidence of OCR pages. Not a probability of correctness; "
                        "shown as — until an OCR page has been processed.")


def _upload() -> None:
    upload_panel("dashboard", title="Upload documents")
    pending = [d for d in state.uploads().values() if d.info is not None and d.file_id not in state.results()]
    if not pending:
        return
    options = default_options()
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown(f"**{len(pending)} file(s) ready:** " + ", ".join(d.name for d in pending[:4])
                    + (" ..." if len(pending) > 4 else ""))
        go = st.button(f"Process {len(pending)} file(s)", type="primary", icon=":material/play_arrow:",
                       disabled=options is None, key="dash_process")
    if options is None:
        st.error("No OCR engine is available. See Settings.", icon=":material/error:")
        return
    st.caption(f"Uses your current processing options ({options.engine}, {'+'.join(options.ocr.languages)}, "
               f"preprocessing: {options.preprocess_preset}). PDFs use embedded text where it exists and OCR for scanned "
               "pages. Results open in **OCR results**; change options on the Image OCR page.")
    if go:
        for doc in pending:
            run_with_progress(doc, options)
        state.go("results", focus=pending[-1].file_id)


QUICK_ACTIONS = [
    ("image_ocr", ":material/document_scanner:", "Image to text", "OCR photos and scans with preprocessing"),
    ("pdf_tools", ":material/picture_as_pdf:", "PDF to text", "Embedded text first, OCR for scanned pages"),
    ("pdf_tools", ":material/find_in_page:", "Searchable PDF", "Add a selectable text layer with OCRmyPDF"),
    ("tables", ":material/table_chart:", "Table extraction", "Export tables to CSV or XLSX"),
    ("structured", ":material/data_object:", "Structured extraction", "Fill template fields; nothing is guessed"),
    ("intelligence", ":material/manage_search:", "Find values", "Document type, e-mails, phones, amounts and IDs"),
    ("enhancement", ":material/compare:", "Compare OCR results", "Try preprocessing variants side by side"),
]


def _quick_actions() -> None:
    searchable = check_searchable_pdf_support(state.settings()).available
    actions = [a for a in QUICK_ACTIONS if a[2] != "Searchable PDF" or searchable]
    st.subheader("Quick actions", anchor=False)
    for row_start in range(0, len(actions), 3):
        cols = st.columns(3)
        for col, (page, icon, title, text) in zip(cols, actions[row_start:row_start + 3]):
            with col, st.container(border=True, height="stretch"):
                st.markdown(f"**{icon} {title}**")
                st.caption(text)
                if st.button("Open", key=f"qa_{title}", icon=":material/arrow_forward:", type="tertiary"):
                    state.go(page)


def _activity() -> None:
    s = state.stats()
    st.subheader("Recent activity", anchor=False)
    if not s.activity:
        with st.container(border=True):
            st.markdown(":material/history: **No documents processed yet**")
            st.caption("Upload a file above or try a sample. Processed files appear here with links to their results.")
        return
    results = state.results()
    with st.container(border=True):
        for i, entry in enumerate(s.activity[:8]):
            fid = entry.get("file_id", "")
            with st.container(horizontal=True, vertical_alignment="center"):
                with st.container():
                    st.markdown(f"**{entry['file']}**")
                    pages = f"{entry['pages']} page" + ("s" if entry["pages"] != 1 else "")
                    st.caption(f"{entry['type']} · {pages} · {entry['seconds']:.2f} s · {entry['time']}"
                               + (f" · {entry['first error']}" if entry["first error"] else ""))
                st.space("stretch")
                badge("Succeeded" if entry["status"] == "success" else "Failed", "ok" if entry["status"] == "success" else "bad")
                result = results.get(fid)
                if result is not None and result.document is not None:
                    if st.button("View", key=f"act_view_{i}", icon=":material/visibility:", type="tertiary"):
                        state.go("results", focus=fid)
                    st.download_button("TXT", document_to_txt_bytes(result.document),
                                       export_filename(entry["file"], "text", "txt"), "text/plain",
                                       key=f"act_txt_{i}", icon=":material/download:", type="tertiary",
                                       on_click="ignore")
            if i < min(len(s.activity), 8) - 1:
                st.divider()
    if len(s.activity) > 1:
        with st.expander("Processing time per document", icon=":material/bar_chart:"):
            st.altair_chart(_time_chart(pd.DataFrame(s.activity[:20])), width="stretch")


def _time_chart(df: pd.DataFrame) -> alt.Chart:
    t = theme.tokens()
    data = df.assign(label=df["file"].str.slice(0, 32), per_page=(df["seconds"] / df["pages"].clip(lower=1)).round(2))
    axis = dict(labelColor=t["muted"], titleColor=t["muted"], gridColor=t["grid"], domainColor=t["axis"], tickColor=t["axis"])
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=4, size=14, color=t["series_1"]).encode(
        x=alt.X("seconds:Q", title="Seconds", axis=alt.Axis(tickCount=5, **axis)),
        y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(grid=False, labelLimit=220, labelOverlap=False, **axis)),
        tooltip=[alt.Tooltip("file:N", title="File"), alt.Tooltip("seconds:Q", title="Total s", format=".2f"),
                 alt.Tooltip("pages:Q", title="Pages"), alt.Tooltip("per_page:Q", title="s / page", format=".2f")],
    )
    labels = bars.mark_text(align="left", dx=4, color=t["ink_2"], fontSize=11).encode(text=alt.Text("seconds:Q", format=".1f"))
    return (bars + labels).properties(height=max(120, 34 * len(data)), padding={"left": 4, "right": 28, "top": 4, "bottom": 4})


def _health() -> None:
    with card("System health", "checked on this computer"):
        health_panel(compact=True)
        try:
            import psutil

            vm = psutil.virtual_memory()
            st.caption(f"System memory: {vm.percent:.0f}% used of {vm.total / 2**30:.1f} GB")
            st.progress(min(1.0, vm.percent / 100))
            st.caption(f"DocIntel process: {psutil.Process().memory_info().rss / 2**20:.0f} MB")
        except ImportError:
            pass
        if st.button("Details", key="dash_health_details", icon=":material/info:", type="tertiary"):
            state.go("about")


def render() -> None:
    _header()
    _kpis()
    main, side = st.columns([2.2, 1], gap="large")
    with main:
        _upload()
        _quick_actions()
        _activity()
    with side:
        _health()
        st.caption("OCR output can contain errors. Confidence values are engine-specific indicators, not probabilities.")
