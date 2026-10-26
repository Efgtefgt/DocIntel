"""Structured extraction: templates, field review, corrections and export."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from docintel.core.exceptions import TemplateError
from docintel.core.result_models import FieldStatus, StructuredResult
from docintel.export.csv_exporter import structured_to_csv_bytes
from docintel.export.file_naming import export_filename
from docintel.export.json_exporter import structured_to_json_bytes
from docintel.extraction.structured_extractor import StructuredExtractor
from docintel.extraction.templates import ExtractionTemplate, TemplateStore, parse_template
from docintel.ui import state
from docintel.ui.components import badge, card, default_options, document_picker, page_header, run_with_progress, upload_panel

STATUS_HELP = {
    "found": "Value located next to its label and passed validation.",
    "missing": "Not present in the text; left empty (null).",
    "empty": "Label present but no value after it.",
    "conflict": "Several different values; choose one manually.",
    "invalid": "Value present but failed type validation; kept verbatim.",
    "uncertain": "Weak evidence (no label, ambiguous date or low OCR confidence).",
    "corrected": "Entered or edited by you.",
}
NEW_TEMPLATE = {
    "name": "my_template",
    "document_type": "my_document",
    "description": "Describe the document type.",
    "fields": {"reference_number": {"type": "string", "labels": ["Reference No", "Ref"]}, "date": "date"},
}


def _template_editor(store: TemplateStore, selected: ExtractionTemplate | None) -> None:
    with st.expander("Create or edit templates", icon=":material/edit:"):
        st.caption("Field types: string, text, phone, email, date, number, amount. Optional keys: labels, pattern "
                   "(regex with a 'value' group), required, multiline, search_unlabeled.")
        editor_key = f"structured_json_{selected.name if selected else 'new'}"
        if editor_key not in st.session_state:
            start = selected.model_dump(exclude_defaults=True) if selected else NEW_TEMPLATE
            st.session_state[editor_key] = json.dumps(start, indent=2, ensure_ascii=False)
        if st.button("Start from a blank example", key="tmpl_blank", icon=":material/note_add:"):
            st.session_state[editor_key] = json.dumps(NEW_TEMPLATE, indent=2)
        raw = st.text_area("Template JSON", height=300, key=editor_key)
        overwrite = st.checkbox("Overwrite if a template with this name exists", key="tmpl_overwrite")
        with st.container(horizontal=True):
            validate = st.button("Validate", key="tmpl_validate", icon=":material/rule:")
            save = st.button("Save template", key="tmpl_save", icon=":material/save:")
        if validate:
            try:
                parse_template(raw)
                st.success("Template is valid.", icon=":material/check_circle:")
            except TemplateError as exc:
                st.error(exc.user_message, icon=":material/error:")
        if save:
            try:
                path = store.save(parse_template(raw), overwrite=overwrite)
                st.success(f"Saved {path.name}.", icon=":material/check_circle:")
            except TemplateError as exc:
                st.error(exc.user_message, icon=":material/error:")


def _show_result(extractor: StructuredExtractor, result: StructuredResult, source_name: str) -> None:
    counts = {s: sum(f.status.value == s for f in result.fields.values()) for s in STATUS_HELP}
    review = counts["conflict"] + counts["invalid"] + counts["uncertain"]
    missing = counts["missing"] + counts["empty"]
    with st.container(horizontal=True, gap="medium"):
        st.metric("Fields", len(result.fields), border=True, icon=":material/data_object:")
        st.metric("Found", counts["found"] + counts["corrected"], border=True, icon=":material/check_circle:",
                  help=f"{counts['corrected']} corrected by you")
        st.metric("Need review", review, border=True, icon=":material/rule:", help="Conflict, invalid or uncertain")
        st.metric("Missing or empty", missing, border=True, icon=":material/remove_circle_outline:",
                  help="Left blank, never guessed")
    if review or not result.valid:
        st.warning(f"{review} field(s) need review." + (" " + " ".join(result.issues) if result.issues else ""),
                   icon=":material/rule:")
    elif missing:
        st.info(f"{missing} field(s) were not found in the text and were left empty.", icon=":material/info:")
    else:
        st.success("All fields found and valid.", icon=":material/check_circle:")
    problems = [f for f in result.fields.values()
                if f.status in (FieldStatus.CONFLICT, FieldStatus.INVALID, FieldStatus.UNCERTAIN)]
    if problems:
        with st.container(horizontal=True, wrap=True):
            for field in problems:
                badge(f"{field.name}: {field.status.value}", "bad" if field.status is FieldStatus.INVALID else "warn")

    rows = []
    for f in result.fields.values():
        corrected = f.status is FieldStatus.CORRECTED or f.original_value is not None
        rows.append({
            "field": f.name, "type": f.field_type,
            "extracted value": (f.original_value if corrected else f.value) or "",
            "your correction": (f.value or "") if corrected else "",
            "status": f.status.value, "normalized": f.normalized_value or "", "notes": " ".join(f.messages),
            "candidates": " | ".join(f.candidates), "source text": f.source_text or "",
        })
    edited = st.data_editor(
        pd.DataFrame(rows), hide_index=True,
        disabled=["field", "type", "extracted value", "status", "normalized", "notes", "candidates", "source text"],
        column_config={
            "extracted value": st.column_config.TextColumn("Extracted value", help="Exactly as found in the text"),
            "your correction": st.column_config.TextColumn("Your correction", help="Type a value to correct this field"),
            "status": st.column_config.TextColumn("Status", help=" ".join(f"{k}: {v}" for k, v in STATUS_HELP.items())),
            "source text": st.column_config.TextColumn("Source text", help="The line the value was taken from"),
        },
        key=f"structured_table_{st.session_state.get('structured_version', 0)}",
    )
    with st.expander("What the statuses mean", icon=":material/help:"):
        for k, v in STATUS_HELP.items():
            st.markdown(f"**{k}**: {v}")
    with st.container(horizontal=True):
        if st.button("Apply corrections", icon=":material/edit_note:", key="fields_apply"):
            corrections = {}
            for row in edited.to_dict("records"):
                new = (row["your correction"] or "").strip()
                field = result.fields[row["field"]]
                if new and new != (field.value or ""):
                    corrections[row["field"]] = new
            st.session_state["structured_result"] = extractor.apply_corrections(result, corrections)
            st.session_state["structured_version"] = st.session_state.get("structured_version", 0) + 1
            st.rerun()
        current: StructuredResult = st.session_state["structured_result"]
        st.download_button("JSON", structured_to_json_bytes(current), export_filename(source_name, "fields", "json"),
                           "application/json", icon=":material/data_object:", key="fields_json", on_click="ignore")
        st.download_button("CSV", structured_to_csv_bytes(current), export_filename(source_name, "fields", "csv"),
                           "text/csv", icon=":material/table_rows:", key="fields_csv", on_click="ignore")


def render() -> None:
    page_header("Structured extraction", "Fields are extracted with deterministic rules (labels and patterns). Values are "
                "never invented: anything not found stays empty and is marked as missing.")
    store = TemplateStore(state.settings().templates_dir)
    names = store.list_names()
    selected = None
    with card("Template", "which fields to look for"):
        if names:
            name = st.selectbox("Template", names, key="tmpl_select", label_visibility="collapsed")
            try:
                selected = store.load(name)
                st.caption(selected.description or f"Document type: {selected.document_type}")
                with st.container(horizontal=True, wrap=True, gap="small"):
                    for fname, spec in selected.fields.items():
                        st.badge(f"{fname} · {spec.type}", color="blue" if spec.required else "gray")
            except TemplateError as exc:
                st.error(exc.user_message, icon=":material/error:")
        else:
            st.info("No templates found. Create one below.", icon=":material/info:")
        _template_editor(store, selected)
    if selected is None:
        return

    extractor = StructuredExtractor(selected)
    with card("Source text", "the recognised text the rules run on"):
        source = _source_text()
        text, low_conf, source_name, source_id = source if source else ("", [], "pasted_text", "")
        clicked = st.button("Extract fields", type="primary", disabled=not text.strip(), icon=":material/data_object:",
                            key="fields_extract")
    if clicked:
        st.session_state["structured_result"] = extractor.extract(text, low_conf)
        st.session_state["structured_template"] = selected.name
        st.session_state["structured_source"] = source_id
        st.session_state["structured_version"] = st.session_state.get("structured_version", 0) + 1
    result = st.session_state.get("structured_result")
    if (result is not None and st.session_state.get("structured_template") == selected.name
            and st.session_state.get("structured_source") == source_id):
        with card("Extracted fields", "correct values in the right-hand column, then apply"):
            _show_result(extractor, result, source_name)


def _source_text() -> tuple[str, list[str], str, str] | None:
    """Return (text, low-confidence words, source name, source id), or None when there is no text yet."""
    choice = st.segmented_control("Take text from", ["Document", "Paste text"], key="fields_source", default="Document",
                                  required=True)
    if choice == "Paste text":
        text = st.text_area("Text", height=200, key="fields_paste")
        return text, [], "pasted_text", f"paste:{hash(text)}"
    upload_panel("structured", title="Add a document")
    doc = document_picker("Document", "fields_doc", prefer=[st.session_state.get("last_processed", ""),
                                                           st.session_state.get("last_added", "")])
    if doc is None:
        st.caption("Add a document above, or paste text instead.")
        return None
    result = state.results().get(doc.file_id)
    if result is None:
        st.info("This document has not been processed yet.", icon=":material/info:")
        options = default_options()
        if st.button("Process it now", icon=":material/play_arrow:", key="fields_process", disabled=options is None):
            run_with_progress(doc, options)
            st.rerun()
        return None
    if result.document is None:
        st.error("This document has no text (processing failed).", icon=":material/error:")
        return None
    edits = state.corrections().get(doc.file_id, {})
    use_edits = bool(edits) and st.checkbox("Use my edited text from OCR results", value=True, key="fields_use_edits")
    pages = [(edits[p.page_number] if use_edits and p.page_number in edits else p.normalized_text)
             for p in result.document.pages if p.succeeded]
    text = "\n\n".join(pages)
    low_conf = [w.text for p in result.document.pages if p.ocr for w in p.ocr.low_confidence_words]
    with st.expander("Show source text", icon=":material/notes:"):
        st.code(text, language=None, wrap_lines=True)
    return text, low_conf, result.filename, f"{doc.file_id}:{use_edits}"
