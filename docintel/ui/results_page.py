"""OCR workspace: document viewer next to the recognised text, with review, edits and exports.

Nothing here runs OCR. Reruns caused by viewer controls (page, zoom, rotation, tabs) only
re-render cached images and stored results.
"""

from __future__ import annotations

import json
from itertools import groupby

import pandas as pd
import streamlit as st

from docintel.core.exceptions import ExportError, OCREngineUnavailableError
from docintel.core.pipeline import PipelineResult
from docintel.core.result_models import PageResult, PageSource
from docintel.core.security import display_path
from docintel.evaluation.metrics import EvalNormalization, cer, wer
from docintel.extraction.ocr_review import apply_suggestion, review
from docintel.export.csv_exporter import pages_to_csv_bytes, table_to_csv_bytes
from docintel.export.file_naming import export_filename, save_export
from docintel.export.json_exporter import document_to_json_bytes
from docintel.export.txt_exporter import document_to_txt_bytes
from docintel.preprocessing.preprocessing_pipeline import PreprocessConfig, run_pipeline
from docintel.ui import state
from docintel.ui.components import badge, card, low_confidence_html, page_header, page_image, upload_panel
from docintel.ui.key_details import key_details_panel

ZOOM = {"Fit": None, "125%": 1.25, "150%": 1.5, "200%": 2.0}
BASE_WIDTH = 640  # px at 100%


@st.cache_data(max_entries=8, show_spinner="Rebuilding the processed image...")
def _processed(source_key: str, _image, steps: tuple[str, ...], max_pixels: int, _detector):
    """Re-apply the preprocessing steps that were used for OCR (cached per page and steps)."""
    return run_pipeline(_image, PreprocessConfig(steps=steps, max_pixels=max_pixels), orientation_detector=_detector).image


def _step_page(key: str, delta: int, count: int) -> None:
    st.session_state[key] = min(count, max(1, int(st.session_state.get(key, 1)) + delta))


def _rotate(key: str, delta: int) -> None:
    st.session_state[key] = (st.session_state.get(key, 0) + delta) % 360


def _viewer(fid: str, doc: state.UploadedDoc | None, result: PipelineResult, page: PageResult, page_key: str,
            position: int, count: int) -> None:
    document = result.document
    with card("Document", f"page {page.page_number} of {document.file_info.page_count}"):
        rot_key = f"ws_rot_{fid}"
        with st.container(horizontal=True, vertical_alignment="center", wrap=True, gap="small"):
            st.button("Previous", icon=":material/chevron_left:", key=f"ws_prev_{fid}", type="tertiary",
                      on_click=_step_page, args=(page_key, -1, count), disabled=position <= 1, help="Previous page")
            st.button("Next", icon=":material/chevron_right:", key=f"ws_next_{fid}", type="tertiary",
                      on_click=_step_page, args=(page_key, 1, count), disabled=position >= count, help="Next page")
            st.button("Rotate left", icon=":material/rotate_left:", key=f"ws_rl_{fid}", type="tertiary",
                      on_click=_rotate, args=(rot_key, 90), help="Rotates the preview only")
            st.button("Rotate right", icon=":material/rotate_right:", key=f"ws_rr_{fid}", type="tertiary",
                      on_click=_rotate, args=(rot_key, -90), help="Rotates the preview only")
        processed_ok = page.source is PageSource.OCR and bool(page.preprocessing) and doc is not None
        with st.container(horizontal=True, vertical_alignment="bottom", wrap=True):
            view = st.segmented_control("View", ["Original", "Processed"], key=f"ws_view_{fid}", default="Original",
                                        required=True, disabled=not processed_ok,
                                        help="Processed = the preprocessed image the OCR engine read" if processed_ok
                                        else "Only available for pages that went through OCR preprocessing")
            zoom = st.segmented_control("Zoom", list(ZOOM), key=f"ws_zoom_{fid}", default="Fit", required=True)
        if doc is None:
            st.info("The original file is no longer in this session.", icon=":material/info:")
            return
        kind = document.file_info.kind.value
        if view == "Processed" and processed_ok:
            source = page_image(doc.data, kind, page.page_number, dpi=state.settings().pdf_render_dpi)
            steps = tuple(s.step for s in page.preprocessing if s.applied and s.step != "remove_table_lines")
            detector = None
            if "orientation" in steps:
                try:
                    detector = state.engine_factory().get("tesseract").detect_orientation
                except OCREngineUnavailableError:
                    steps = tuple(s for s in steps if s != "orientation")
            image = _processed(f"{fid}:{page.page_number}:{state.settings().pdf_render_dpi}", source, steps,
                               state.settings().max_image_pixels, detector)
            caption = "Processed: " + (", ".join(steps) if steps else "no changes")
        else:
            image = page_image(doc.data, kind, page.page_number)
            caption = "Original"
        rotation = st.session_state.get(f"ws_rot_{fid}", 0)
        if rotation:
            image = image.rotate(rotation, expand=True, fillcolor="white")
            caption += f" · rotated {(-rotation) % 360}° for viewing"
        scale = ZOOM[zoom]
        with st.container(height=640, border=False):
            st.image(image, caption=caption, width="stretch" if scale is None else int(BASE_WIDTH * scale),
                     alt=f"Page {page.page_number} of {document.file_info.filename}")
        with st.expander("File information", icon=":material/info:"):
            info = document.file_info
            rows = {"File": info.filename, "Format": info.format, "Size": f"{info.size_bytes / 1024:.1f} KB",
                    "Pages": str(info.page_count)}
            if info.width:
                rows["Dimensions"] = f"{info.width} × {info.height} px"
            if page.detection_reason:
                rows["Text detection (this page)"] = page.detection_reason
            st.table(pd.DataFrame(rows.items(), columns=["Property", "Value"]).set_index("Property"))


def _text_tab(fid: str, page: PageResult) -> None:
    edits = state.corrections().setdefault(fid, {})
    edited = edits.get(page.page_number)
    if page.error:
        st.error(f"This page was not processed: {page.error.message}", icon=":material/error:")
    elif page.source is PageSource.NONE:
        st.warning("No text was found on this page.", icon=":material/warning:")
    if page.quality and page.quality.verdict != "good":
        (st.error if page.quality.verdict == "poor" else st.warning)(
            " ".join(page.quality.notes) + " See the **Quality** and **Review** tabs.", icon=":material/rule:")
    elif page.ocr and page.ocr.low_confidence_words:
        st.warning(f"{len(page.ocr.low_confidence_words)} word(s) have low OCR confidence. Check them in the "
                   "**Confidence** tab before relying on this text.", icon=":material/rule:")
    source_label = {"ocr": "OCR output", "embedded_text": "Embedded PDF text", "none": "No text"}[page.source.value]
    with st.container(horizontal=True, vertical_alignment="center"):
        badge(source_label, "info")
        if edited is not None:
            badge("Edited by you", "warn")
    edit_key = f"ws_edit_{fid}_{page.page_number}"
    if st.toggle("Edit text", key=f"ws_edit_toggle_{fid}_{page.page_number}",
                 help="Your edits are stored separately; the OCR output is never changed"):
        st.session_state.setdefault(edit_key, edited if edited is not None else page.normalized_text)
        st.text_area("Your corrected text", key=edit_key, height=360)
        with st.container(horizontal=True):
            if st.button("Save edits", key=f"ws_save_{fid}_{page.page_number}", icon=":material/save:", type="primary"):
                edits[page.page_number] = st.session_state[edit_key]
                st.toast("Edits saved for this session.", icon=":material/check_circle:")
            if edited is not None and st.button("Discard edits", key=f"ws_discard_{fid}_{page.page_number}",
                                                icon=":material/undo:"):
                edits.pop(page.page_number, None)
                st.session_state.pop(edit_key, None)
                st.rerun()
    else:
        if edited is not None:
            choice = st.segmented_control("Show", ["Your edits", "OCR output"], key=f"ws_show_{fid}_{page.page_number}",
                                          default="Your edits", required=True)
            text = edited if choice == "Your edits" else page.normalized_text
        else:
            text = page.normalized_text
        st.caption("Use the copy icon at the top-right of the box to copy the text.")
        st.code(text or "", language=None, wrap_lines=True)
    with st.expander("Raw output (exactly as returned by the engine or PDF)", icon=":material/notes:"):
        st.code(page.raw_text or "", language=None, wrap_lines=True)
        if page.normalization_changes:
            st.caption("Normalization applied: " + ", ".join(page.normalization_changes))
    _accuracy(fid, page, edited)


def _accuracy(fid: str, page: PageResult, edited: str | None) -> None:
    with st.expander("Check accuracy against a reference", icon=":material/fact_check:"):
        st.caption("Paste the verified text of this page. Whitespace differences are ignored; case and "
                   "punctuation count (docs/EVALUATION.md).")
        reference = st.text_area("Reference text", key=f"ws_ref_{fid}_{page.page_number}", height=120,
                                 label_visibility="collapsed", placeholder="Verified text of this page")
        if reference.strip():
            loose = EvalNormalization(lowercase=True, remove_punctuation=True)
            cols = st.columns(3 if edited is not None else 2)
            cols[0].metric("CER (OCR output)", f"{cer(reference, page.normalized_text):.3f}", border=True,
                           help="Character error rate: 0 = identical")
            cols[1].metric("WER (OCR output)", f"{wer(reference, page.normalized_text):.3f}", border=True,
                           help=f"Ignoring case and punctuation: CER {cer(reference, page.normalized_text, loose):.3f}")
            if edited is not None:
                cols[2].metric("CER (your edits)", f"{cer(reference, edited):.3f}", border=True)


def _confidence_tab(page: PageResult) -> None:
    ocr = page.ocr
    assert ocr is not None
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean confidence", f"{ocr.mean_confidence:.0f}" if ocr.mean_confidence is not None else "—", border=True)
    c2.metric("Words", len(ocr.words), border=True)
    c3.metric("Low-confidence words", len(ocr.low_confidence_words), border=True)
    st.caption(f"Highlighted words scored below {ocr.options.get('low_confidence_threshold')} ({ocr.confidence_scale}). "
               "They are flagged for review, never changed.")
    st.markdown(low_confidence_html(ocr), unsafe_allow_html=True)
    lines = []
    for key, group in groupby(ocr.words, key=lambda w: (w.block, w.paragraph, w.line)):
        words = list(group)
        confs = [w.confidence for w in words if w.confidence is not None]
        lines.append({"Line": len(lines) + 1, "Text": " ".join(w.text for w in words)[:80],
                      "Mean confidence": round(sum(confs) / len(confs)) if confs else None,
                      "Lowest": round(min(confs)) if confs else None,
                      "Low-confidence words": sum(w.low_confidence for w in words)})
    if lines:
        st.markdown("**Confidence by line**")
        st.dataframe(pd.DataFrame(lines), hide_index=True, column_config={
            "Mean confidence": st.column_config.ProgressColumn("Mean confidence", min_value=0, max_value=100, format="%d"),
            "Lowest": st.column_config.NumberColumn("Lowest word", format="%d")})
    with st.expander("Engine details", icon=":material/tune:"):
        rows = {"Engine": f"{ocr.engine} {ocr.engine_version}".strip(), "Languages": "+".join(ocr.languages),
                "Page segmentation": str(ocr.options.get("psm")), "OCR image": f"{ocr.image_width} × {ocr.image_height} px",
                "Recognition time": f"{ocr.duration_seconds:.2f} s"}
        st.table(pd.DataFrame(rows.items(), columns=["Property", "Value"]).set_index("Property"))
        if page.preprocessing:
            st.dataframe(pd.DataFrame([s.model_dump() for s in page.preprocessing]), hide_index=True)


VERDICT_BADGE = {"good": ("Looks reliable", "ok"), "review": ("Review recommended", "warn"), "poor": ("Likely incomplete or wrong", "bad")}
SEVERITY_BADGE = {"problem": "bad", "warning": "warn", "info": "ok"}


def _quality_tab(page: PageResult) -> None:
    q = page.quality
    assert q is not None
    label, kind = VERDICT_BADGE.get(q.verdict, ("Review recommended", "warn"))
    badge(label, kind)
    for note in q.notes:
        st.caption(note)
    c1, c2, c3 = st.columns(3)
    c1.metric("Text coverage", f"{q.coverage:.0%}" if q.coverage is not None else "\u2014", border=True,
              help="Recognised characters / characters estimated on the page image (independent of the OCR engine)")
    c2.metric("Recognised", q.recognised_characters, border=True, help="Letters and digits recognised")
    c3.metric("Estimated", q.estimated_characters if q.estimated_characters is not None else "\u2014", border=True)
    if q.findings:
        st.markdown("**Image analysis**")
        with st.container(horizontal=True, wrap=True, gap="small"):
            for finding in q.findings:
                badge(finding.message, SEVERITY_BADGE.get(finding.severity, "info"))
    if q.plan:
        st.markdown("**Preprocessing chosen automatically**")
        for reason in q.plan:
            st.markdown(f"- {reason}")
    if q.attempts:
        st.markdown("**OCR attempts**")
        st.dataframe(pd.DataFrame([{"Plan": a.label, "Steps": ", ".join(a.steps), "Mean confidence": a.mean_confidence,
                                    "Coverage": f"{a.coverage:.0%}" if a.coverage is not None else "",
                                    "Score": a.score, "Seconds": a.seconds, "Kept": a.chosen} for a in q.attempts]),
                     hide_index=True)
        st.caption("Score = confidence x coverage. Extra attempts run only when the first result looks weak.")
    if q.metrics:
        with st.expander("Measurements", icon=":material/straighten:"):
            st.dataframe(pd.DataFrame(q.metrics.items(), columns=["Measure", "Value"]).astype(str), hide_index=True)


def _accept(fid: str, page_number: int, base: str, token: str, suggestion: str) -> None:
    new_text = apply_suggestion(base, token, suggestion)
    state.corrections().setdefault(fid, {})[page_number] = new_text
    st.session_state[f"ws_edit_{fid}_{page_number}"] = new_text


def _review_tab(fid: str, page: PageResult) -> None:
    assert page.ocr is not None
    flags = review(page.ocr)
    if not flags:
        badge("No suspicious words found", "ok")
        st.caption("Checks: low confidence, letters and digits mixed in a word, unusual characters, repeated characters.")
        return
    edited = state.corrections().get(fid, {}).get(page.page_number)
    base = edited if edited is not None else page.normalized_text
    st.caption(f"{len(flags)} word(s) look suspicious. Nothing is changed automatically: accepting a suggestion "
               "writes it into **your edits**, and the OCR output stays as it is.")
    for i, flag in enumerate(flags[:60]):
        with st.container(horizontal=True, vertical_alignment="center"):
            conf = f" · confidence {flag.confidence:.0f}" if flag.confidence is not None else ""
            st.markdown(f"`{flag.text}` :gray[line {flag.line + 1}{conf} · {', '.join(flag.reasons)}]")
            st.space("stretch")
            if flag.suggestion:
                already = flag.suggestion in base and flag.text not in base
                st.button(f"Use \u201c{flag.suggestion}\u201d", key=f"ws_sugg_{fid}_{page.page_number}_{i}", type="tertiary",
                          icon=":material/check:", disabled=already or flag.text not in base,
                          on_click=_accept, args=(fid, page.page_number, base, flag.text, flag.suggestion),
                          help="Applies to your edited copy only")
    if len(flags) > 60:
        st.caption(f"... and {len(flags) - 60} more.")


def _tables_tab(fid: str, result: PipelineResult, page: PageResult) -> None:
    tables = [t for t in result.tables if t.page_number == page.page_number] or result.tables
    for table in tables:
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f"**Table {table.table_index}** · page {table.page_number} · {table.method}")
            badge("Uncertain, review" if table.uncertain else "Structure detected", "warn" if table.uncertain else "ok")
        for note in table.notes:
            st.caption(note)
        st.dataframe(table.to_dataframe(), hide_index=True)
        st.download_button(f"CSV · table {table.table_index}", table_to_csv_bytes(table),
                           export_filename(result.filename, f"table{table.table_index}", "csv"), "text/csv",
                           key=f"ws_tcsv_{fid}_{table.table_index}", icon=":material/download:", on_click="ignore")


def _structured_tab(result: PipelineResult) -> None:
    structured = result.structured
    assert structured is not None
    st.caption(f"Template: {structured.template_name}. Missing fields stay empty; nothing is guessed.")
    df = pd.DataFrame(structured.to_flat_records())[["field", "value", "status", "messages"]]
    st.dataframe(df, hide_index=True)
    st.page_link(state.PAGE_FILES["structured"], label="Review and correct fields", icon=":material/edit_note:")


def _exports(fid: str, result: PipelineResult) -> None:
    document = result.document
    name = document.file_info.filename
    edits = state.corrections().get(fid, {})
    with card("Export", "all pages, with page boundaries"):
        with st.container(horizontal=True, wrap=True):
            txt = document_to_txt_bytes(document)
            st.download_button("TXT", txt, export_filename(name, "text", "txt"), "text/plain", key="ws_dl_txt",
                               icon=":material/description:", on_click="ignore", help="Normalized text")
            st.download_button("TXT (raw)", document_to_txt_bytes(document, normalized=False),
                               export_filename(name, "raw", "txt"), "text/plain", key="ws_dl_raw",
                               icon=":material/notes:", on_click="ignore")
            if edits:
                corrected = document.model_copy(deep=True)
                for p in corrected.pages:
                    if p.page_number in edits:
                        p.normalized_text = edits[p.page_number]
                st.download_button("TXT (with your edits)", document_to_txt_bytes(corrected),
                                   export_filename(name, "edited", "txt"), "text/plain", key="ws_dl_edited",
                                   icon=":material/edit_note:", on_click="ignore")
            st.download_button("JSON", document_to_json_bytes(document, include_words=True),
                               export_filename(name, "result", "json"), "application/json", key="ws_dl_json",
                               icon=":material/data_object:", on_click="ignore")
            st.download_button("CSV (per page)", pages_to_csv_bytes(document), export_filename(name, "pages", "csv"),
                               "text/csv", key="ws_dl_csv", icon=":material/table_rows:", on_click="ignore")
            if st.button("Save TXT copy", icon=":material/save:", key="ws_save_copy",
                         help=f"Writes the TXT into {display_path(state.settings().export_dir)} (never overwrites)"):
                try:
                    st.success(f"Saved {display_path(save_export(txt, export_filename(name, 'text', 'txt'), state.settings().export_dir))}")
                except ExportError as exc:
                    st.error(exc.user_message)


def render() -> None:
    page_header("OCR results", "Review the document next to its recognised text, check confidence, correct text and export.")
    choices = state.processed_choices()
    if not choices:
        st.info("No processed documents yet. Upload a file and process it, here or on the Image OCR page.",
                icon=":material/info:")
        upload_panel("results")
        st.page_link(state.PAGE_FILES["image_ocr"], label="Go to Image OCR to process files", icon=":material/arrow_forward:")
        return
    ids = list(choices)
    focus = st.session_state.get("last_processed")
    fid = st.selectbox("Document", ids, index=ids.index(focus) if focus in ids else len(ids) - 1,
                       format_func=lambda i: choices[i], key="ws_doc")
    st.session_state["last_processed"] = fid
    result = state.results()[fid]
    doc = state.uploads().get(fid)
    for error in result.errors:
        if error.page_number is None:
            st.error(f"{error.stage}: {error.message}", icon=":material/error:")
    if result.document is None or not result.document.pages:
        return
    document = result.document
    numbers = [p.page_number for p in document.pages]
    page_key = f"ws_page_{fid}"
    count = len(numbers)
    position = min(count, max(1, int(st.session_state.get(page_key, 1))))
    page = document.pages[position - 1]

    confs = [p.ocr.mean_confidence for p in document.pages if p.ocr and p.ocr.mean_confidence is not None]
    with st.container(horizontal=True, gap="medium"):
        st.metric("Status", "Succeeded" if document.succeeded else "Errors", border=True,
                  icon=":material/check_circle:" if document.succeeded else ":material/error:")
        st.metric("Pages", f"{len(document.pages)}", border=True, icon=":material/auto_stories:",
                  help=f"{sum(p.source is PageSource.OCR for p in document.pages)} via OCR, "
                       f"{sum(p.source is PageSource.EMBEDDED_TEXT for p in document.pages)} embedded text")
        st.metric("Processing time", f"{result.duration_seconds:.2f} s", border=True, icon=":material/schedule:")
        st.metric("Mean confidence", f"{sum(confs) / len(confs):.0f}" if confs else "—", border=True,
                  icon=":material/verified:", help="Engine-specific score of OCR pages; — when no page used OCR")
    if result.analysis is not None:
        c = result.analysis.classification
        strength = {"high": "strong", "medium": "some", "low": "weak"}.get(c.confidence)
        st.badge(f"Type: {c.label}" + (f" ({strength} evidence)" if strength else ""),
                 icon=":material/description:", color="blue" if c.confidence in ("high", "medium") else "gray")
        with st.expander("Key details: phone numbers, e-mails, PIN/ZIP codes, dates, amounts, IDs",
                         icon=":material/manage_search:"):
            key_details_panel("results", [fid])

    if count > 1:
        st.select_slider("Page", options=list(range(1, count + 1)), key=page_key,
                         format_func=lambda n: f"Page {numbers[n - 1]}")
    left, right = st.columns([1, 1.15], gap="medium")
    with left:
        _viewer(fid, doc, result, page, page_key, position, count)
    with right, card("Recognised text", f"page {page.page_number}"):
        tabs = ["Text"]
        if page.ocr and page.ocr.words:
            tabs += ["Review", "Confidence"]
        if page.quality is not None:
            tabs.append("Quality")
        if result.tables:
            tabs.append("Tables")
        if result.structured is not None:
            tabs.append("Structured data")
        tabs.append("JSON")
        for name, tab in zip(tabs, st.tabs(tabs)):
            with tab:
                if name == "Text":
                    _text_tab(fid, page)
                elif name == "Review":
                    _review_tab(fid, page)
                elif name == "Confidence":
                    _confidence_tab(page)
                elif name == "Quality":
                    _quality_tab(page)
                elif name == "Tables":
                    _tables_tab(fid, result, page)
                elif name == "Structured data":
                    _structured_tab(result)
                else:
                    payload = json.loads(document_to_json_bytes(document))
                    st.json(payload["pages"][position - 1], expanded=1)
    _exports(fid, result)
    with st.expander("Processing stages", icon=":material/timeline:"):
        st.dataframe(pd.DataFrame([s.model_dump() for s in result.stages]), hide_index=True)
    upload_panel("results_more", title="Add more documents")
