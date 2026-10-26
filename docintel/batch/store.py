"""On-disk batch storage.

Layout under ``settings.batch_dir`` (default %LOCALAPPDATA%\\DocIntel\\batches)::

    <batch_id>/batch.json            queue state (Batch)
    <batch_id>/inputs/<item>_<name>  copies of uploaded files (folder inputs are read in place)
    <batch_id>/results/<item>.json   full result per file (StoredResult)
    <batch_id>/results/<item>.pdf    searchable PDF, when requested

Everything stays on this computer; deleting a batch removes its folder. One process owns a
running batch (``owner_pid``); a batch left "running" by a process that no longer exists is
marked interrupted and can be resumed.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

from pydantic import BaseModel, Field

from docintel.batch.models import Batch, BatchItem, BatchState, ItemState, OutputFormat, now
from docintel.core.file_detector import SUPPORTED_EXTENSIONS
from docintel.core.pipeline import PipelineResult
from docintel.core.result_models import (
    DocumentAnalysis,
    DocumentResult,
    StageError,
    StageRecord,
    StructuredResult,
    TableResult,
)
from docintel.core.security import ensure_within, safe_filename

logger = logging.getLogger(__name__)
_BATCH_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
_ITEM_ID = re.compile(r"^\d{3,5}$")


class StoredResult(BaseModel):
    """PipelineResult as JSON (it is a dataclass holding pydantic models)."""

    filename: str
    document: Optional[DocumentResult] = None
    tables: list[TableResult] = Field(default_factory=list)
    structured: Optional[StructuredResult] = None
    analysis: Optional[DocumentAnalysis] = None
    stages: list[StageRecord] = Field(default_factory=list)
    errors: list[StageError] = Field(default_factory=list)
    duration_seconds: float = 0.0
    rss_mb: Optional[float] = None

    @classmethod
    def from_result(cls, result: PipelineResult) -> "StoredResult":
        return cls(filename=result.filename, document=result.document, tables=result.tables,
                   structured=result.structured, analysis=result.analysis, stages=result.stages,
                   errors=result.errors, duration_seconds=result.duration_seconds, rss_mb=result.rss_mb)

    def to_result(self) -> PipelineResult:
        return PipelineResult(filename=self.filename, document=self.document, tables=self.tables,
                              structured=self.structured, analysis=self.analysis, stages=self.stages,
                              errors=self.errors, duration_seconds=self.duration_seconds, rss_mb=self.rss_mb)


def scan_folder(folder: Path, *, recursive: bool, max_files: int) -> tuple[list[tuple[str, Path, int]], bool]:
    """Supported files in ``folder`` as (name relative to the folder, path, size), sorted by name.

    Returns (files, truncated). Hidden files and folders are skipped; the type is only guessed
    from the extension here and checked by content when the file is processed.
    """
    extensions = {f".{ext.lower()}" for ext in SUPPORTED_EXTENSIONS}
    pattern = "**/*" if recursive else "*"
    found: list[tuple[str, Path, int]] = []
    for path in sorted(folder.glob(pattern), key=lambda p: str(p).lower()):
        relative = path.relative_to(folder)
        if any(part.startswith(".") for part in relative.parts) or path.suffix.lower() not in extensions:
            continue
        try:
            if not path.is_file():
                continue
            size = path.stat().st_size
        except OSError:
            continue
        if len(found) >= max_files:
            return found, True
        found.append((relative.as_posix(), path, size))
    return found, False


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    if pid == os.getpid():
        return True
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:  # psutil is a core dependency; be conservative without it
        return True


class BatchStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._lock = threading.RLock()
        self._cache: dict[str, Batch] = {}

    # --- paths -------------------------------------------------------------------------
    def folder(self, batch_id: str) -> Path:
        if not _BATCH_ID.match(batch_id):
            raise ValueError("Invalid batch id.")
        return ensure_within(self.root, Path(batch_id))

    def _result_path(self, batch_id: str, item_id: str, suffix: str) -> Path:
        if not _ITEM_ID.match(item_id):
            raise ValueError("Invalid item id.")
        return self.folder(batch_id) / "results" / f"{item_id}{suffix}"

    # --- batches -----------------------------------------------------------------------
    def create(self, *, title: str, source: str, options: dict, outputs: Iterable[OutputFormat],
               uploads: Iterable[tuple[str, bytes]] = (),
               folder_files: Iterable[tuple[str, Path, int]] = ()) -> Batch:
        batch_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
        folder = self.folder(batch_id)
        (folder / "inputs").mkdir(parents=True)
        (folder / "results").mkdir()
        batch = Batch(batch_id=batch_id, title=title, source=source, options=options, outputs=list(outputs))
        for name, data in uploads:
            item_id = f"{len(batch.items) + 1:03d}"
            stored = f"{item_id}_{safe_filename(name)}"
            (folder / "inputs" / stored).write_bytes(data)
            batch.items.append(BatchItem(item_id=item_id, name=name, size_bytes=len(data), stored_input=stored))
        for name, path, size in folder_files:
            item_id = f"{len(batch.items) + 1:03d}"
            batch.items.append(BatchItem(item_id=item_id, name=name, size_bytes=size, source_path=str(path)))
        self.save(batch)
        logger.info("Batch created", extra={"batch": batch_id, "files": len(batch.items), "source": source})
        return batch

    def save(self, batch: Batch) -> None:
        with self._lock:
            batch.updated_at = now()
            target = self.folder(batch.batch_id) / "batch.json"
            tmp = target.with_suffix(".tmp")
            tmp.write_text(batch.model_dump_json(indent=1), encoding="utf-8")
            for attempt in range(20):  # a reader (antivirus, indexer) can hold the file briefly on Windows
                try:
                    os.replace(tmp, target)
                    break
                except PermissionError:
                    time.sleep(0.05 * (attempt + 1))
            else:
                raise PermissionError(f"Could not update {target.name}")
            self._cache[batch.batch_id] = batch.model_copy(deep=True)

    def get(self, batch_id: str) -> Optional[Batch]:
        with self._lock:
            if batch_id not in self._cache:
                path = self.folder(batch_id) / "batch.json"
                try:
                    self._cache[batch_id] = Batch.model_validate_json(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    return None
            return self._cache[batch_id].model_copy(deep=True)

    def update(self, batch_id: str, change: Callable[[Batch], None]) -> Optional[Batch]:
        """Apply ``change`` to the stored batch and save it (atomic under the store lock)."""
        with self._lock:
            batch = self.get(batch_id)
            if batch is None:
                return None
            change(batch)
            self.save(batch)
            return batch

    def list(self) -> list[Batch]:
        if not self.root.is_dir():
            return []
        batches = [b for p in self.root.iterdir() if _BATCH_ID.match(p.name) and (b := self.get(p.name))]
        return sorted(batches, key=lambda b: b.created_at, reverse=True)

    def delete(self, batch_id: str) -> None:
        with self._lock:
            folder = self.folder(batch_id)
            self._cache.pop(batch_id, None)
            shutil.rmtree(folder, ignore_errors=True)
        logger.info("Batch deleted", extra={"batch": batch_id})

    def recover(self) -> list[str]:
        """Mark batches left running by a stopped process as interrupted (their running file restarts)."""
        recovered = []
        for batch in self.list():
            if batch.state in (BatchState.RUNNING, BatchState.QUEUED) and not _pid_alive(batch.owner_pid):
                def change(b: Batch) -> None:
                    b.state = BatchState.INTERRUPTED
                    b.message = "DocIntel stopped while this batch was running. Resume to continue."
                    for item in b.items:
                        if item.state is ItemState.RUNNING:
                            item.state = ItemState.QUEUED
                self.update(batch.batch_id, change)
                recovered.append(batch.batch_id)
        return recovered

    # --- inputs and results --------------------------------------------------------------
    def input_bytes(self, batch: Batch, item: BatchItem) -> bytes:
        if item.stored_input:
            return ensure_within(self.folder(batch.batch_id) / "inputs", Path(item.stored_input)).read_bytes()
        if item.source_path:
            return Path(item.source_path).read_bytes()
        raise FileNotFoundError("No input recorded for this file.")

    def save_result(self, batch_id: str, item_id: str, result: PipelineResult) -> None:
        path = self._result_path(batch_id, item_id, ".json")
        path.write_text(StoredResult.from_result(result).model_dump_json(), encoding="utf-8")

    def load_result(self, batch_id: str, item_id: str) -> Optional[StoredResult]:
        try:
            return StoredResult.model_validate_json(self._result_path(batch_id, item_id, ".json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def save_file(self, batch_id: str, item_id: str, suffix: str, data: bytes) -> None:
        self._result_path(batch_id, item_id, suffix).write_bytes(data)

    def load_file(self, batch_id: str, item_id: str, suffix: str) -> Optional[bytes]:
        try:
            return self._result_path(batch_id, item_id, suffix).read_bytes()
        except OSError:
            return None

    def clear_result(self, batch_id: str, item_id: str) -> None:
        for suffix in (".json", ".pdf"):
            self._result_path(batch_id, item_id, suffix).unlink(missing_ok=True)
