"""Batch results package: one ZIP with per-file outputs, a summary and all key details.

    summary.csv / summary.html   one row per file: status, pages, time, confidence, type, values, error
    key_details.csv              every phone, e-mail, PIN/ZIP code, date, amount and ID, with file and page
    README.txt                   what the files and columns mean
    documents/<nnn>_<name>/      <name>.txt, <name>.json, tables_<n>.csv, fields.csv, <name>_searchable.pdf

Personal ID numbers (PAN, Aadhaar, card, IBAN, account) are masked in key_details.csv unless
``mask_ids`` is False. The per-file text and JSON are the recognised text as-is.
"""

from __future__ import annotations

import html
import zipfile
from pathlib import Path, PurePath
from typing import BinaryIO, Union

import pandas as pd

from docintel import __version__
from docintel.batch.models import Batch, ItemState, OutputFormat
from docintel.batch.store import BatchStore
from docintel.core.security import safe_filename
from docintel.export.csv_exporter import dataframe_to_csv_bytes, structured_to_csv_bytes, table_to_csv_bytes
from docintel.export.txt_exporter import document_to_txt_bytes
from docintel.extraction.document_analysis import mask_sensitive
from docintel.extraction.entities import ENTITY_GROUPS, ENTITY_TYPES

STATUS_TEXT = {ItemState.DONE: "done", ItemState.FAILED: "failed", ItemState.CANCELLED: "cancelled",
               ItemState.QUEUED: "not processed", ItemState.RUNNING: "not finished"}
def _group_column(group: str) -> str:
    """"PIN / ZIP code" -> "pin_zip_code" (summary column names)."""
    return group.lower().replace(" / ", "_").replace(" ", "_").replace("-", "")


SUMMARY_COLUMNS = ["file", "status", "pages", "ocr_pages", "failed_pages", "pages_to_review", "mean_confidence",
                   "seconds", "document_type", "type_evidence", *[_group_column(g) for g in ENTITY_GROUPS],
                   "tables", "fields_valid", "error", "warnings"]

README = """DocIntel batch results
======================

summary.csv / summary.html
    One row per file. status: done, failed (see "error"), cancelled or not processed.
    mean_confidence: average engine score of OCR pages (Tesseract 0-100; not a probability and
    not comparable between engines). pages_to_review: OCR pages whose quality check did not say
    "good"; read those pages carefully. Value columns count what was found (see key_details.csv).
key_details.csv
    Every phone number, e-mail, PIN/ZIP code, website, date, amount and ID found, with file and page.
    status: valid (confirmed by a label, checksum or calendar check), uncertain (pattern only),
    invalid (fails its check; often an OCR error). Values are exactly as written; nothing is guessed.
documents/<nnn>_<name>/
    The recognised text (.txt), the full result (.json) and, when chosen, tables, template fields
    and a searchable PDF.

OCR is never 100% accurate. Check important values against the original documents.
"""


def summary_frame(batch: Batch) -> pd.DataFrame:
    rows = []
    for item in batch.items:
        s = item.summary
        counts = s.value_counts if s else {}
        row = {
            "file": item.name, "status": STATUS_TEXT[item.state],
            "pages": s.pages if s else None, "ocr_pages": s.ocr_pages if s else None,
            "failed_pages": s.failed_pages if s else None, "pages_to_review": s.pages_to_review if s else None,
            "mean_confidence": s.mean_confidence if s else None, "seconds": item.duration_seconds,
            "document_type": s.document_type if s else "", "type_evidence": s.type_confidence if s else "",
            "tables": s.tables if s else None,
            "fields_valid": "" if not s or s.fields_valid is None else ("yes" if s.fields_valid else "needs review"),
            "error": item.error, "warnings": "; ".join(item.warnings),
        }
        for group, types in ENTITY_GROUPS.items():
            row[_group_column(group)] = sum(counts.get(t, 0) for t in types) if s else None
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def key_details_frame(store: BatchStore, batch: Batch, *, mask_ids: bool = True) -> pd.DataFrame:
    group_of = {t: g for g, types in ENTITY_GROUPS.items() for t in types}
    rows = []
    for item in batch.items:
        stored = store.load_result(batch.batch_id, item.item_id) if item.state is ItemState.DONE or \
            item.state is ItemState.FAILED else None
        if stored is None or stored.analysis is None:
            continue
        analysis = mask_sensitive(stored.analysis) if mask_ids else stored.analysis
        for e in analysis.entities:
            if e.type in group_of:
                rows.append({"file": item.name, "page": e.page_number, "group": group_of[e.type],
                             "type": ENTITY_TYPES.get(e.type, e.type), "value": e.value,
                             "normalized": e.normalized or "", "status": e.status.value, "label": e.label or ""})
    return pd.DataFrame(rows, columns=["file", "page", "group", "type", "value", "normalized", "status", "label"])


def summary_html(batch: Batch, frame: pd.DataFrame) -> str:
    done, failed = batch.count(ItemState.DONE), batch.count(ItemState.FAILED)
    head = "".join(f"<th>{html.escape(c)}</th>" for c in frame.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{'' if pd.isna(v) else html.escape(str(v))}</td>" for v in row) + "</tr>"
        for row in frame.itertuples(index=False))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>DocIntel batch summary</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px;color:#0f172a;background:#fff}}
table{{border-collapse:collapse;font-size:13px}}th,td{{border:1px solid #cbd5e1;padding:4px 6px;text-align:left}}
th{{background:#eef2f8}}.wrap{{overflow-x:auto}}</style></head><body>
<h1>{html.escape(batch.title)}</h1>
<p>{len(batch.items)} file(s): {done} done, {failed} failed, {batch.count(ItemState.CANCELLED)} cancelled.
Created {batch.created_at:%Y-%m-%d %H:%M} UTC with DocIntel {html.escape(__version__)}.
OCR is never 100% accurate; check important values against the originals.</p>
<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></body></html>"""


def build_zip(store: BatchStore, batch: Batch, target: Union[Path, BinaryIO], *, mask_ids: bool = True) -> int:
    """Write the package to ``target``; returns the number of files in the ZIP."""
    summary = summary_frame(batch)
    written = 0
    root = safe_filename(f"DocIntel_batch_{batch.batch_id}")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        def add(name: str, data: Union[bytes, str]) -> None:
            nonlocal written
            zf.writestr(f"{root}/{name}", data)
            written += 1

        add("README.txt", README)
        add("summary.csv", dataframe_to_csv_bytes(summary))
        add("summary.html", summary_html(batch, summary))
        add("key_details.csv", dataframe_to_csv_bytes(key_details_frame(store, batch, mask_ids=mask_ids)))
        for item in batch.items:
            stored = store.load_result(batch.batch_id, item.item_id)
            if stored is None or stored.document is None:
                continue
            stem = PurePath(safe_filename(PurePath(item.name).name)).stem or "document"
            folder = f"documents/{item.item_id}_{stem}"
            if OutputFormat.TEXT in batch.outputs:
                add(f"{folder}/{stem}.txt", document_to_txt_bytes(stored.document))
            if OutputFormat.JSON in batch.outputs:
                add(f"{folder}/{stem}.json", stored.model_dump_json(
                    indent=2, exclude={"document": {"pages": {"__all__": {"ocr": {"words"}}}}}))
            if OutputFormat.TABLES in batch.outputs:
                for table in stored.tables:
                    add(f"{folder}/tables_{table.table_index}_page{table.page_number}.csv", table_to_csv_bytes(table))
            if OutputFormat.FIELDS in batch.outputs and stored.structured is not None:
                add(f"{folder}/fields.csv", structured_to_csv_bytes(stored.structured))
            if OutputFormat.SEARCHABLE_PDF in batch.outputs and item.has_searchable_pdf:
                pdf = store.load_file(batch.batch_id, item.item_id, ".pdf")
                if pdf:
                    add(f"{folder}/{stem}_searchable.pdf", pdf)
    return written
