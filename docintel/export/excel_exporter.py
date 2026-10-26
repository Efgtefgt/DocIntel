"""XLSX export (one sheet per table plus a notes sheet) using openpyxl via pandas."""

from __future__ import annotations

import io
from typing import Sequence

import pandas as pd

from docintel.core.exceptions import ExportError
from docintel.core.result_models import StructuredResult, TableResult
from docintel.export.csv_exporter import formula_safe


def tables_to_xlsx_bytes(tables: Sequence[TableResult]) -> bytes:
    if not tables:
        raise ExportError("There are no tables to export.")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        notes = []
        for table in tables:
            sheet = f"p{table.page_number}_t{table.table_index}"[:31]
            table.to_dataframe().map(formula_safe).to_excel(writer, sheet_name=sheet, index=False)
            notes.append(
                {
                    "sheet": sheet,
                    "page": table.page_number,
                    "method": table.method,
                    "uncertain": table.uncertain,
                    "quality_score": table.quality_score,
                    "notes": " ".join(table.notes),
                }
            )
        pd.DataFrame(notes).to_excel(writer, sheet_name="extraction_notes", index=False)
    return buffer.getvalue()


def structured_to_xlsx_bytes(result: StructuredResult) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(result.to_flat_records()).map(formula_safe).to_excel(writer, sheet_name="fields", index=False)
    return buffer.getvalue()
