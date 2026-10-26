"""JSON export of document, table and structured results (UTF-8)."""

from __future__ import annotations

import json
from typing import Sequence

from pydantic import BaseModel

from docintel.core.result_models import DocumentResult, StructuredResult, TableResult


def model_to_json_bytes(model: BaseModel, **dump_kwargs: object) -> bytes:
    return model.model_dump_json(indent=2, **dump_kwargs).encode("utf-8")  # type: ignore[arg-type]


def document_to_json_bytes(document: DocumentResult, *, include_words: bool = False) -> bytes:
    """Serialize a document. Word-level boxes are excluded by default to keep files small."""
    exclude = None if include_words else {"pages": {"__all__": {"ocr": {"words"}}}}
    return document.model_dump_json(indent=2, exclude=exclude).encode("utf-8")


def structured_to_json_bytes(result: StructuredResult, *, include_source_text: bool = False) -> bytes:
    exclude = None if include_source_text else {"source_text"}
    return result.model_dump_json(indent=2, exclude=exclude).encode("utf-8")


def tables_to_json_bytes(tables: Sequence[TableResult]) -> bytes:
    return json.dumps([t.model_dump() for t in tables], indent=2, ensure_ascii=False).encode("utf-8")
