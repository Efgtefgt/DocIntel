"""Document intelligence: document type, detected entities (numbers, e-mails, IDs...) and key-value pairs."""

from __future__ import annotations

import html
from typing import Optional

import pandas as pd
import streamlit as st

from docintel.core.result_models import DocumentAnalysis, Entity, EntityStatus
from docintel.export.csv_exporter import entities_to_csv_bytes, key_values_to_csv_bytes
from docintel.export.file_naming import export_filename
from docintel.export.json_exporter import model_to_json_bytes
from docintel.extraction.document_analysis import analyze_document, analyze_text, mask_sensitive
from docintel.extraction.entities import ENTITY_TYPES
from docintel.extraction.templates import TemplateStore
from docintel.ui import state, theme
from docintel.ui.components import badge, card, default_options, document_picker, page_header, run_with_progress, upload_panel

CONFIDENCE_BADGES = {"high": ("Strong evidence", "ok"), "medium": ("Some evidence", "info"),
                     "low": ("Weak evidence, check", "warn"), "none": ("No clear type", "off")}
STATUS_HELP = {
    "valid": "Format confirmed by a checksum, a calendar check or a matching label.",
    "uncertain": "Matches the pattern, but nothing confirms it (no label, ambiguous, or low OCR confidence).",
    "invalid": "Has the shape of the type but fails its checksum or calendar check; often an OCR error.",
}


def _type_label(etype: str) -> str:
    return ENTITY_TYPES.get(etype, etype)


def _template_types() -> dict[str, str]:
    return TemplateStore(state.settings().templates_dir).document_types()


def _source() -> Optional[tuple[DocumentAnalysis, str, str]]:
    """Return (analysis, source name, file id or "") for the chosen text, or None."""
    choice = st.segmented_control("Analyse", ["Document", "Paste text"], key="intel_source", default="Document",
                                  required=True)
    if choice == "Paste text":
        text = st.text_area("Text", height=180, key="intel_paste",
                            placeholder="Paste any text: an e-mail, an invoice, a form...")
        if not text.strip():
            st.caption("Paste text above to analyse it.")
            return None
        return analyze_text(text, template_types=_template_types()), "pasted_text", ""
    upload_panel("intelligence", title="Add a document")
    doc = document_picker("Document", "intel_doc", prefer=[st.session_state.get("last_processed", ""),
                                                          st.session_state.get("last_added", "")])
    if doc is None:
        st.caption("Add a document above, or paste text instead.")
        return None
    result = state.results().get(doc.file_id)
    if result is None:
        st.info("This document has not been processed yet.", icon=":material/info:")
        options = default_options()
        if st.button("Process it now", icon=":material/play_arrow:", key="intel_process", disabled=options is None):
            run_with_progress(doc, options)
            st.rerun()
        return None
    if result.document is None or not any(p.succeeded for p in result.document.pages):
        st.error("This document has no text (processing failed).", icon=":material/error:")
        return None
    edits = state.corrections().get(doc.file_id, {})
    use_edits = bool(edits) and st.checkbox("Use my edited text from OCR results", value=True, key="intel_use_edits")
    analysis = analyze_document(result.document, page_texts=edits if use_edits else None,
                                template_types=_template_types())
    return analysis, result.filename, doc.file_id


def _classification(analysis: DocumentAnalysis, file_id: str) -> None:
    c = analysis.classification
    with card("Document type", "rule-based, from keywords and the values found"):
        with st.container(horizontal=True, vertical_alignment="center", wrap=True):
            st.markdown(f"### {c.label}")
            text, kind = CONFIDENCE_BADGES[c.confidence]
            badge(text, kind)  # type: ignore[arg-type]
        for note in c.notes:
            st.caption(note)
        if c.scores:
            top = c.scores[0]
            st.caption("Evidence: " + ", ".join(top.evidence[:10]) + (" …" if len(top.evidence) > 10 else ""))
            with st.expander("All candidate types", icon=":material/leaderboard:"):
                st.dataframe(
                    pd.DataFrame([{"type": s.label, "score": s.score, "evidence": ", ".join(s.evidence)}
                                  for s in c.scores]),
                    hide_index=True,
                    column_config={"score": st.column_config.ProgressColumn(
                        "Score", min_value=0.0, max_value=max(s.score for s in c.scores), format="%.1f",
                        help="Weighted count of matching title phrases, keywords and values. Not a probability.")},
                )
        if c.suggested_templates:
            with st.container(horizontal=True, wrap=True):
                for name in c.suggested_templates:
                    if st.button(f"Extract fields with '{name}'", icon=":material/data_object:", key=f"intel_tmpl_{name}"):
                        st.session_state["tmpl_select"] = name
                        state.go("structured", focus=file_id or None)


def _metrics(analysis: DocumentAnalysis) -> None:
    review = sum(e.status is not EntityStatus.VALID for e in analysis.entities)
    sensitive = sum(e.sensitive for e in analysis.entities)
    with st.container(horizontal=True, gap="medium"):
        st.metric("Values found", len(analysis.entities), border=True, icon=":material/manage_search:",
                  help="E-mails, phone numbers, dates, amounts, IDs and other numbers")
        st.metric("Need review", review, border=True, icon=":material/rule:", help="Uncertain or invalid values")
        st.metric("Labelled fields", sum(not kv.empty for kv in analysis.key_values), border=True,
                  icon=":material/list_alt:", help='"Label: value" pairs written in the document')
        st.metric("Personal IDs", sensitive, border=True, icon=":material/shield_person:",
                  help="PAN, Aadhaar, card, IBAN or bank account numbers")


def _entity_table(analysis: DocumentAnalysis) -> None:
    counts = analysis.entity_counts()
    if not counts:
        st.info("No e-mails, phone numbers, dates, amounts, IDs or numbers were found.", icon=":material/info:")
        return
    order = [t for t in ENTITY_TYPES if t in counts]
    chosen = st.pills("Show", order, selection_mode="multi", default=order, key="intel_types",
                      format_func=lambda t: f"{_type_label(t)} · {counts[t]}")
    status_filter = st.segmented_control("Status", ["All", "Needs review"], default="All", key="intel_status",
                                         required=True)
    rows = [
        {"page": e.page_number, "type": _type_label(e.type), "value": e.value, "normalized": e.normalized or "",
         "status": e.status.value, "label": e.label or "", "notes": " ".join(e.notes), "line": e.line}
        for e in analysis.entities
        if e.type in (chosen or []) and (status_filter == "All" or e.status is not EntityStatus.VALID)
    ]
    if not rows:
        st.caption("Nothing matches the filters.")
        return
    frame = pd.DataFrame(rows)
    if frame["page"].isna().all():
        frame = frame.drop(columns="page")
    st.dataframe(frame, hide_index=True, column_config={
        "page": st.column_config.NumberColumn("Page"),
        "type": st.column_config.TextColumn("Type"),
        "notes": st.column_config.TextColumn("Notes"),
        "value": st.column_config.TextColumn("Value", help="Exactly as written in the text"),
        "normalized": st.column_config.TextColumn("Normalized", help="Standard form; empty when it can't be "
                                                  "determined without guessing (e.g. ambiguous dates)"),
        "status": st.column_config.TextColumn("Status", help=" ".join(f"{k}: {v}" for k, v in STATUS_HELP.items())),
        "label": st.column_config.TextColumn("Label", help="Label written next to the value"),
        "line": st.column_config.TextColumn("Source line"),
    })
    with st.expander("What the statuses mean", icon=":material/help:"):
        for k, v in STATUS_HELP.items():
            st.markdown(f"**{k}**: {v}")


def _key_value_table(analysis: DocumentAnalysis) -> None:
    if not analysis.key_values:
        st.info('No "Label: value" pairs were found.', icon=":material/info:")
        return
    frame = pd.DataFrame([
        {"page": kv.page_number, "label": kv.key, "value": kv.value, "value type": _type_label(kv.value_type)
         if kv.value_type != "text" else "text", "notes": " ".join(kv.notes)}
        for kv in analysis.key_values
    ])
    if frame["page"].isna().all():
        frame = frame.drop(columns="page")
    st.dataframe(frame, hide_index=True, column_config={
        "page": st.column_config.NumberColumn("Page"), "label": st.column_config.TextColumn("Label"),
        "value": st.column_config.TextColumn("Value", help="Exactly as written in the text"),
        "value type": st.column_config.TextColumn("Value type"), "notes": st.column_config.TextColumn("Notes")})


def highlighted_html(text: str, entities: list[Entity]) -> str:
    """Text with entities marked (HTML-escaped). Status is shown by underline style and tooltip, not colour alone."""
    t = theme.tokens()
    styles = {
        EntityStatus.VALID: f"background:{t['ent_bg']};color:{t['ent_fg']};border-bottom:2px solid {t['ent_bd']}",
        EntityStatus.UNCERTAIN: f"background:{t['low_bg']};color:{t['low_fg']};border-bottom:2px dotted {t['low_bd']}",
        EntityStatus.INVALID: f"background:{t['bad_bg']};color:{t['bad_fg']};border-bottom:3px double {t['bad_bd']}",
    }
    parts, pos = [], 0
    for e in sorted(entities, key=lambda x: x.start):
        if e.start < pos:
            continue
        parts.append(html.escape(text[pos: e.start]))
        tip = html.escape(f"{_type_label(e.type)} · {e.status.value}" + (f" · {e.notes[0]}" if e.notes else ""))
        parts.append(f'<mark style="{styles[e.status]};padding:0 2px;border-radius:2px" title="{tip}">'
                     f"{html.escape(text[e.start: e.end])}</mark>")
        pos = e.end
    parts.append(html.escape(text[pos:]))
    box = (f"font-family:ui-monospace,Consolas,monospace;font-size:0.9rem;line-height:1.8;color:{t['ink']};"
           f"background:{t['raised']};border:1px solid {t['line']};border-radius:8px;padding:0.8rem;"
           "max-height:520px;overflow:auto;overflow-wrap:anywhere;white-space:pre-wrap")
    return f'<div style="{box}">' + "".join(parts) + "</div>"


def _highlighted(analysis: DocumentAnalysis, texts: dict[Optional[int], str]) -> None:
    st.caption("Solid underline: valid · dotted: uncertain · double: invalid. Hover a value for its type.")
    for page_number, text in texts.items():
        if page_number is not None and len(texts) > 1:
            st.markdown(f"**Page {page_number}**")
        st.html(highlighted_html(text, [e for e in analysis.entities if e.page_number == page_number]))


def _page_texts(file_id: str) -> dict[Optional[int], str]:
    if not file_id:
        return {None: st.session_state.get("intel_paste", "")}
    result = state.results()[file_id]
    edits = state.corrections().get(file_id, {}) if st.session_state.get("intel_use_edits") else {}
    return {p.page_number: edits.get(p.page_number, p.normalized_text) for p in result.document.pages if p.succeeded}


def render() -> None:
    page_header("Document intelligence", "Finds the document type and every e-mail, phone number, date, amount, ID "
                "and number in the text, plus its \"Label: value\" fields. Rules and checksums only: values are "
                "shown exactly as written and nothing is guessed.")
    with card("Source"):
        source = _source()
    if source is None:
        return
    analysis, source_name, file_id = source
    masked = st.toggle("Mask personal IDs (PAN, Aadhaar, card and account numbers)", value=True, key="intel_mask",
                       help="Applies to this page and its downloads. The processed document itself is unchanged.")
    shown = mask_sensitive(analysis) if masked else analysis
    _classification(shown, file_id)
    _metrics(shown)
    with card("What was found"):
        tabs = st.tabs(["Values", "Labelled fields", "Highlighted text"])
        with tabs[0]:
            _entity_table(shown)
        with tabs[1]:
            _key_value_table(shown)
        with tabs[2]:
            if masked and any(e.sensitive for e in analysis.entities):
                st.caption("Turn off masking to see the highlighted text; it would show personal IDs in full.")
            else:
                _highlighted(analysis, _page_texts(file_id))
        with st.container(horizontal=True, wrap=True):
            st.download_button("Values (CSV)", entities_to_csv_bytes(shown),
                               export_filename(source_name, "entities", "csv"), "text/csv",
                               icon=":material/download:", key="intel_csv", on_click="ignore")
            st.download_button("Labelled fields (CSV)", key_values_to_csv_bytes(shown),
                               export_filename(source_name, "fields", "csv"), "text/csv",
                               icon=":material/download:", key="intel_kv_csv", on_click="ignore")
            st.download_button("Everything (JSON)", model_to_json_bytes(shown),
                               export_filename(source_name, "analysis", "json"), "application/json",
                               icon=":material/data_object:", key="intel_json", on_click="ignore")
