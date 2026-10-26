"""CSV export for tables, structured fields and per-page text.

CSV files are written as UTF-8 with a byte-order mark (``utf-8-sig``) so that Excel on
Windows displays non-ASCII characters correctly; other tools read it as normal UTF-8.
"""

from __future__ import annotations

import pandas as pd

from docintel.core.result_models import DocumentAnalysis, DocumentResult, StructuredResult, TableResult

CSV_ENCODING = "utf-8-sig"


def formula_safe(value: object) -> object:
    """Prefix cells that spreadsheet apps would execute as formulas (CSV injection)."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r") and not _is_number(value):
        return "'" + value
    return value


def _is_number(value: str) -> bool:
    try:
        float(value.replace(",", ""))
        return True
    except ValueError:
        return False


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.map(formula_safe).to_csv(index=False).encode(CSV_ENCODING)


def table_to_csv_bytes(table: TableResult) -> bytes:
    return dataframe_to_csv_bytes(table.to_dataframe())


def structured_to_csv_bytes(result: StructuredResult) -> bytes:
    return dataframe_to_csv_bytes(pd.DataFrame(result.to_flat_records()))


def pages_to_csv_bytes(document: DocumentResult) -> bytes:
    records = [
        {
            "page": p.page_number,
            "source": p.source.value,
            "mean_confidence": p.ocr.mean_confidence if p.ocr else None,
            "error": p.error.message if p.error else "",
            "text": p.normalized_text,
        }
        for p in document.pages
    ]
    return dataframe_to_csv_bytes(pd.DataFrame(records))


def entities_to_csv_bytes(analysis: DocumentAnalysis) -> bytes:
    records = [
        {"page": e.page_number, "type": e.type, "value": e.value, "normalized": e.normalized, "status": e.status.value,
         "label": e.label, "notes": "; ".join(e.notes), "line": e.line}
        for e in analysis.entities
    ]
    columns = ["page", "type", "value", "normalized", "status", "label", "notes", "line"]
    return dataframe_to_csv_bytes(pd.DataFrame(records, columns=columns))


def key_values_to_csv_bytes(analysis: DocumentAnalysis) -> bytes:
    records = [
        {"page": kv.page_number, "key": kv.key, "value": kv.value, "value_type": kv.value_type,
         "notes": "; ".join(kv.notes), "line": kv.line}
        for kv in analysis.key_values
    ]
    columns = ["page", "key", "value", "value_type", "notes", "line"]
    return dataframe_to_csv_bytes(pd.DataFrame(records, columns=columns))
