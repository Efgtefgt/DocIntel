"""Key details: a quick view of the main values found in processed documents.

Shown right after processing (Image OCR, PDF tools) and in OCR results. One click on a shortcut
(Phone, E-mail, PIN/ZIP code...) lists those values; the full view is the Document intelligence page.
Values come from the pipeline's rule-based analysis: verbatim text, nothing guessed.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import streamlit as st

from docintel.core.result_models import DocumentAnalysis, EntityStatus
from docintel.export.csv_exporter import entities_to_csv_bytes
from docintel.export.file_naming import export_filename
from docintel.extraction.document_analysis import mask_sensitive
from docintel.extraction.entities import ENTITY_GROUPS
from docintel.ui import state
from docintel.ui.components import card

SHORTCUTS = ENTITY_GROUPS  # shortcut label -> entity types it shows
STATUS_TEXT = {EntityStatus.VALID: "✓ confirmed", EntityStatus.UNCERTAIN: "? check", EntityStatus.INVALID: "✗ looks wrong"}


def _analysed(file_ids: Iterable[str]) -> dict[str, tuple[str, DocumentAnalysis]]:
    results = state.results()
    return {fid: (results[fid].filename, results[fid].analysis) for fid in file_ids
            if fid in results and results[fid].analysis is not None}


def key_details_panel(key: str, file_ids: Iterable[str]) -> None:
    """Shortcut buttons for the main values of one processed document (picked when several are given)."""
    available = _analysed(file_ids)
    if not available:
        return
    with card("Key details", "phone numbers, e-mails, PIN/ZIP codes and more, found in the text"):
        ids = list(available)
        fid = ids[0]
        if len(ids) > 1:
            fid = st.selectbox("Document", ids, key=f"{key}_kd_doc", format_func=lambda i: available[i][0],
                               index=len(ids) - 1)
        name, analysis = available[fid]
        show_ids = st.toggle("Show full ID numbers", value=False, key=f"{key}_kd_unmask",
                             help="PAN, Aadhaar, card and bank account numbers are masked (••••1234) unless switched on.")
        shown = analysis if show_ids else mask_sensitive(analysis)

        counts = {label: sum(e.type in types for e in shown.entities) for label, types in SHORTCUTS.items()}
        offered = [label for label, n in counts.items() if n]
        if not offered:
            st.caption("No phone numbers, e-mails, PIN/ZIP codes, dates, amounts or IDs were found in this document.")
            return
        total = sum(counts.values())
        picked = st.pills("Show", ["All", *offered], default="All", key=f"{key}_kd_pick",
                          format_func=lambda label: f"{label} · {total if label == 'All' else counts[label]}")
        chosen = offered if picked in (None, "All") else [picked]
        types = {t for label in chosen for t in SHORTCUTS[label]}
        rows = [{"What": label, "Value": e.value, "Check": STATUS_TEXT[e.status], "Page": e.page_number,
                 "Label in document": e.label or ""}
                for label, ts in SHORTCUTS.items() for e in shown.entities if e.type in ts and e.type in types]
        if rows:
            frame = pd.DataFrame(rows)
            if frame["Page"].isna().all():
                frame = frame.drop(columns="Page")
            st.dataframe(frame, hide_index=True, column_config={
                "Value": st.column_config.TextColumn("Value", help="Exactly as written; select cells to copy"),
                "Check": st.column_config.TextColumn("Check", help="✓ confirmed by a label or checksum · "
                                                     "? matches the pattern only · ✗ fails its check (often OCR error)"),
            })
        with st.container(horizontal=True, wrap=True, vertical_alignment="center"):
            filtered = shown.model_copy(update={"entities": [e for e in shown.entities if e.type in types]})
            st.download_button("Download (CSV)", entities_to_csv_bytes(filtered),
                               export_filename(name, "key_details", "csv"), "text/csv",
                               key=f"{key}_kd_csv", icon=":material/download:", on_click="ignore")
            if st.button("All values and fields", key=f"{key}_kd_more", icon=":material/manage_search:",
                         type="tertiary"):
                state.go("intelligence", focus=fid)
