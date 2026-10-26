"""Batch records: one Batch with one BatchItem per file. Stored as JSON (see store.py).

Only metadata and summary numbers live here; each file's full result is a separate JSON file.
File names are kept, but folder inputs record their path relative to the chosen folder, so
summaries and ZIPs never contain the user's home path.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def now() -> datetime:
    return datetime.now(timezone.utc)


class BatchState(str, Enum):
    QUEUED = "queued"            # waiting for the worker
    RUNNING = "running"
    PAUSED = "paused"            # stopped after a file at the user's request; resume continues
    INTERRUPTED = "interrupted"  # the app stopped while it ran; resume continues
    CANCELLED = "cancelled"
    DONE = "done"                # every file finished (some may have failed)


class ItemState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OutputFormat(str, Enum):
    TEXT = "text"                  # <name>.txt
    JSON = "json"                  # <name>.json (full result)
    TABLES = "tables"              # tables_<n>.csv (when tables were extracted)
    FIELDS = "fields"              # fields.csv (when a template was used)
    SEARCHABLE_PDF = "searchable_pdf"  # <name>_searchable.pdf (PDF inputs only; slow)


OUTPUT_LABELS = {
    OutputFormat.TEXT: "Text (TXT)",
    OutputFormat.JSON: "Full result (JSON)",
    OutputFormat.TABLES: "Tables (CSV)",
    OutputFormat.FIELDS: "Template fields (CSV)",
    OutputFormat.SEARCHABLE_PDF: "Searchable PDF (slow)",
}
DEFAULT_OUTPUTS = [OutputFormat.TEXT, OutputFormat.JSON]


class ItemSummary(BaseModel):
    """Numbers shown in the queue and written to summary.csv."""

    pages: int = 0
    ocr_pages: int = 0
    failed_pages: int = 0
    pages_to_review: int = 0          # OCR pages whose quality verdict is not "good"
    mean_confidence: Optional[float] = None
    document_type: str = ""
    type_confidence: str = ""
    value_counts: dict[str, int] = Field(default_factory=dict)   # entity type -> count
    tables: int = 0
    fields_valid: Optional[bool] = None


class BatchItem(BaseModel):
    item_id: str                      # "001", "002", ... (order of the queue)
    name: str                         # display name: file name or path relative to the folder
    size_bytes: int = 0
    stored_input: Optional[str] = None   # file name under <batch>/inputs (uploaded files)
    source_path: Optional[str] = None    # absolute path (folder inputs; never exported)
    state: ItemState = ItemState.QUEUED
    attempts: int = 0
    stronger: bool = False            # retry with heavier settings (higher DPI, more OCR attempts)
    started_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    error: str = ""
    warnings: list[str] = Field(default_factory=list)
    summary: Optional[ItemSummary] = None
    has_searchable_pdf: bool = False


class Batch(BaseModel):
    batch_id: str
    title: str
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)
    finished_at: Optional[datetime] = None
    state: BatchState = BatchState.QUEUED
    source: str = "upload"            # "upload" or "folder"
    options: dict[str, Any] = Field(default_factory=dict)   # serialized PipelineOptions (options.py)
    outputs: list[OutputFormat] = Field(default_factory=lambda: list(DEFAULT_OUTPUTS))
    items: list[BatchItem] = Field(default_factory=list)
    owner_pid: Optional[int] = None   # process running it; used to detect interrupted batches
    message: str = ""

    def count(self, *states: ItemState) -> int:
        return sum(item.state in states for item in self.items)

    @property
    def finished(self) -> int:
        return self.count(ItemState.DONE, ItemState.FAILED, ItemState.CANCELLED)

    @property
    def active(self) -> bool:
        return self.state in (BatchState.QUEUED, BatchState.RUNNING)

    def eta_seconds(self) -> Optional[float]:
        """Remaining time from the average of finished files; None until one file is done."""
        times = [i.duration_seconds for i in self.items if i.state in (ItemState.DONE, ItemState.FAILED)
                 and i.duration_seconds]
        remaining = self.count(ItemState.QUEUED, ItemState.RUNNING)
        if not times or not remaining:
            return None
        return sum(times) / len(times) * remaining
