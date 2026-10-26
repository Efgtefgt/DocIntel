"""Background batch worker: one thread per process, one file at a time (8 GB RAM friendly).

* Pause takes effect after the current file; cancel stops between pages of the current file
  (its partial result is discarded) and marks the remaining files cancelled.
* One file failing (corrupt, encrypted, too large, timeout) never stops the batch.
* Before each file, free memory is checked; below ``batch_min_free_memory_mb`` the worker
  waits up to a minute so other programs (or the UI) can release memory.
* Progress is saved after every file, so a stopped app can resume where it was.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from docintel.batch.models import Batch, BatchItem, BatchState, ItemState, ItemSummary, OutputFormat, now
from docintel.batch.options import options_from_dict, options_to_dict, stronger
from docintel.batch.store import BatchStore
from docintel.config.settings import Settings
from docintel.core.exceptions import DocIntelError
from docintel.core.pipeline import DocumentPipeline, PipelineOptions, PipelineResult
from docintel.core.result_models import FileKind, PageSource
from docintel.ocr.engine_factory import EngineFactory

logger = logging.getLogger(__name__)
MEMORY_WAIT_SECONDS = 60


class BatchCancelled(Exception):
    """Raised from the progress callback to stop the current file (not a DocIntelError on purpose,
    so the pipeline does not record it as a page failure and continue)."""


@dataclass
class LiveStatus:
    batch_id: str
    item_id: str
    fraction: float = 0.0
    message: str = "starting"


def summarize(result: PipelineResult) -> ItemSummary:
    document = result.document
    summary = ItemSummary(tables=len(result.tables))
    if document is not None:
        ocr_pages = [p for p in document.pages if p.source is PageSource.OCR and p.ocr]
        confidences = [p.ocr.mean_confidence for p in ocr_pages if p.ocr.mean_confidence is not None]
        summary.pages = len(document.pages)
        summary.ocr_pages = len(ocr_pages)
        summary.failed_pages = len(document.failed_pages)
        summary.pages_to_review = sum(1 for p in ocr_pages if p.quality and p.quality.verdict != "good")
        summary.mean_confidence = round(sum(confidences) / len(confidences), 1) if confidences else None
    if result.analysis is not None:
        summary.document_type = result.analysis.classification.label
        summary.type_confidence = result.analysis.classification.confidence
        summary.value_counts = result.analysis.entity_counts()
    if result.structured is not None:
        summary.fields_valid = result.structured.valid
    return summary


class BatchRunner:
    def __init__(self, settings: Settings, *,
                 factory_builder: Optional[Callable[[Settings], EngineFactory]] = None,
                 recover: bool = True) -> None:
        self.settings = settings
        self.store = BatchStore(settings.batch_dir)
        self._factory_builder = factory_builder or EngineFactory
        self._lock = threading.Lock()
        self._queue: deque[str] = deque()
        self._cancel: set[str] = set()
        self._pause: set[str] = set()
        self._thread: Optional[threading.Thread] = None
        self.live: Optional[LiveStatus] = None
        if recover:
            self.store.recover()

    # --- public API ----------------------------------------------------------------------
    def create(self, *, title: str, options: PipelineOptions, outputs: Iterable[OutputFormat],
               uploads: Iterable[tuple[str, bytes]] = (),
               folder_files: Iterable[tuple[str, Path, int]] = (), start: bool = True) -> Batch:
        uploads, folder_files = list(uploads), list(folder_files)
        total = len(uploads) + len(folder_files)
        if not total:
            raise ValueError("A batch needs at least one file.")
        if total > self.settings.batch_max_files:
            raise ValueError(f"A batch can hold at most {self.settings.batch_max_files} files "
                             "(DOCINTEL_BATCH_MAX_FILES).")
        batch = self.store.create(title=title, source="folder" if folder_files else "upload",
                                  options=options_to_dict(options), outputs=outputs,
                                  uploads=uploads, folder_files=folder_files)
        if start:
            self.start(batch.batch_id)
        return self.store.get(batch.batch_id) or batch

    def start(self, batch_id: str) -> None:
        """Queue a new, paused or interrupted batch (no effect when finished or already queued)."""
        def change(b: Batch) -> None:
            if b.state in (BatchState.QUEUED, BatchState.PAUSED, BatchState.INTERRUPTED, BatchState.CANCELLED,
                           BatchState.DONE) and b.count(ItemState.QUEUED):
                b.state, b.message, b.finished_at = BatchState.QUEUED, "", None
                b.owner_pid = os.getpid()
        batch = self.store.update(batch_id, change)
        if batch is None or batch.state is not BatchState.QUEUED:
            return
        with self._lock:
            self._pause.discard(batch_id)
            self._cancel.discard(batch_id)
            if batch_id not in self._queue and not (self.live and self.live.batch_id == batch_id):
                self._queue.append(batch_id)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, name="docintel-batch", daemon=True)
                self._thread.start()

    def pause(self, batch_id: str) -> None:
        with self._lock:
            if batch_id in self._queue:
                self._queue.remove(batch_id)
                self.store.update(batch_id, lambda b: setattr(b, "state", BatchState.PAUSED))
            else:
                self._pause.add(batch_id)

    def cancel(self, batch_id: str) -> None:
        with self._lock:
            running = self.live is not None and self.live.batch_id == batch_id
            if batch_id in self._queue:
                self._queue.remove(batch_id)
            if running:
                self._cancel.add(batch_id)
                return
        self.store.update(batch_id, self._mark_cancelled)

    def retry_failed(self, batch_id: str, *, stronger_settings: bool = False) -> int:
        """Queue failed and cancelled files again; returns how many."""
        count = 0

        def change(b: Batch) -> None:
            nonlocal count
            for item in b.items:
                if item.state in (ItemState.FAILED, ItemState.CANCELLED):
                    item.state, item.error, item.warnings, item.summary = ItemState.QUEUED, "", [], None
                    item.stronger = item.stronger or stronger_settings
                    count += 1
        if self.busy_with(batch_id):
            return 0
        self.store.update(batch_id, change)
        if count:
            self.start(batch_id)
        return count

    def delete(self, batch_id: str) -> bool:
        if self.busy_with(batch_id):
            return False
        with self._lock:
            if batch_id in self._queue:
                self._queue.remove(batch_id)
        self.store.delete(batch_id)
        return True

    def busy_with(self, batch_id: str) -> bool:
        with self._lock:
            return batch_id in self._queue or (self.live is not None and self.live.batch_id == batch_id)

    @property
    def busy(self) -> bool:
        with self._lock:
            return bool(self._queue) or self.live is not None

    def wait(self, timeout: float = 60.0) -> bool:
        """Block until the worker is idle (tests and the API). True when idle."""
        end = time.monotonic() + timeout
        while self.busy and time.monotonic() < end:
            time.sleep(0.05)
        return not self.busy

    # --- worker --------------------------------------------------------------------------
    def _loop(self) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    self._thread = None
                    return
                batch_id = self._queue.popleft()
                self.live = LiveStatus(batch_id, "", 0.0, "starting")  # busy from the moment it leaves the queue
            try:
                self._run(batch_id)
            except Exception:  # never let one batch kill the worker
                logger.exception("Batch worker error", extra={"batch": batch_id})
                self.store.update(batch_id, lambda b: (setattr(b, "state", BatchState.INTERRUPTED),
                                                       setattr(b, "message", "Unexpected error; resume to retry.")))
            finally:
                with self._lock:
                    self.live = None

    @staticmethod
    def _mark_cancelled(b: Batch) -> None:
        for item in b.items:
            if item.state in (ItemState.QUEUED, ItemState.RUNNING):
                item.state, item.error = ItemState.CANCELLED, "Cancelled before it was processed."
        b.state, b.finished_at, b.message = BatchState.CANCELLED, now(), "Cancelled."

    def _run(self, batch_id: str) -> None:
        batch = self.store.update(batch_id, lambda b: (setattr(b, "state", BatchState.RUNNING),
                                                        setattr(b, "owner_pid", os.getpid())))
        if batch is None:
            return
        options, warnings = options_from_dict(batch.options, self.settings.templates_dir)
        factories: dict[bool, EngineFactory] = {}
        logger.info("Batch started", extra={"batch": batch_id, "files": len(batch.items)})
        try:
            for item in batch.items:
                current = self.store.get(batch_id)
                state = next((i.state for i in current.items if i.item_id == item.item_id), None) if current else None
                if state is not ItemState.QUEUED:
                    continue
                with self._lock:
                    if batch_id in self._cancel:
                        self._cancel.discard(batch_id)
                        self.store.update(batch_id, self._mark_cancelled)
                        return
                    if batch_id in self._pause:
                        self._pause.discard(batch_id)
                        self.store.update(batch_id, lambda b: (setattr(b, "state", BatchState.PAUSED),
                                                               setattr(b, "message", "Paused.")))
                        return
                self._wait_for_memory(batch_id)
                if item.stronger not in factories:
                    factories[item.stronger] = self._factory_builder(self._settings_for(item))
                if not self._process_item(batch_id, item, options, warnings, factories[item.stronger]):
                    self.store.update(batch_id, self._mark_cancelled)
                    return
            self.store.update(batch_id, lambda b: (setattr(b, "state", BatchState.DONE),
                                                   setattr(b, "finished_at", now()),
                                                   setattr(b, "message", "")))
            logger.info("Batch finished", extra={"batch": batch_id})
        finally:
            for factory in factories.values():
                factory.close_all()
            gc.collect()

    def _settings_for(self, item: BatchItem) -> Settings:
        if not item.stronger:
            return self.settings
        return self.settings.model_copy(update={
            "pdf_render_dpi": max(self.settings.pdf_render_dpi, 300),
            "max_ocr_attempts": 4,
            "document_timeout_seconds": self.settings.document_timeout_seconds * 1.5,
        })

    def _wait_for_memory(self, batch_id: str) -> None:
        floor = self.settings.batch_min_free_memory_mb
        if floor <= 0:
            return
        try:
            import psutil
        except ImportError:
            return
        end = time.monotonic() + MEMORY_WAIT_SECONDS
        while psutil.virtual_memory().available / 1_048_576 < floor and time.monotonic() < end:
            with self._lock:
                self.live = LiveStatus(batch_id, "", 0.0, "waiting for free memory")
                if batch_id in self._cancel:
                    return
            gc.collect()
            time.sleep(2)

    def _process_item(self, batch_id: str, item: BatchItem, options: PipelineOptions, warnings: list[str],
                      factory: EngineFactory) -> bool:
        """Process one file and record its outcome. Returns False when the batch was cancelled."""
        with self._lock:
            self.live = LiveStatus(batch_id, item.item_id)

        def mark_running(b: Batch) -> None:
            for i in b.items:
                if i.item_id == item.item_id:
                    i.state, i.started_at, i.attempts = ItemState.RUNNING, now(), i.attempts + 1
        batch = self.store.update(batch_id, mark_running)
        assert batch is not None
        self.store.clear_result(batch_id, item.item_id)

        def progress(fraction: float, message: str) -> None:
            with self._lock:
                self.live = LiveStatus(batch_id, item.item_id, fraction, message)
                if batch_id in self._cancel:
                    raise BatchCancelled

        started = time.perf_counter()
        error, result = "", None
        item_options = stronger(options) if item.stronger else options
        try:
            data = self.store.input_bytes(batch, item)
            pipeline = DocumentPipeline(factory.settings, factory)
            result = pipeline.process(data, Path(item.name).name, item_options, progress=progress)
        except BatchCancelled:
            with self._lock:
                self._cancel.discard(batch_id)
            logger.info("Batch cancelled during a file", extra={"batch": batch_id})
            return False
        except FileNotFoundError:
            error = "The file no longer exists at its original location."
        except OSError as exc:
            error = f"The file could not be read ({type(exc).__name__})."
        except DocIntelError as exc:
            error = exc.user_message
        except Exception:  # an unexpected bug in one file must not stop the batch
            logger.exception("Unexpected error in batch file", extra={"batch": batch_id, "item": item.item_id})
            error = "Unexpected error while processing this file (see the log)."

        searchable = False
        if result is not None:
            self.store.save_result(batch_id, item.item_id, result)
            if not result.succeeded:
                error = "; ".join(dict.fromkeys(e.message for e in result.errors[:3])) or "No text could be read."
            if OutputFormat.SEARCHABLE_PDF in batch.outputs and result.document is not None \
                    and result.document.file_info.kind is FileKind.PDF:
                searchable = self._searchable_pdf(batch_id, item, data, item_options, factory.settings)
        duration = round(time.perf_counter() - started, 2)

        def finish(b: Batch) -> None:
            for i in b.items:
                if i.item_id == item.item_id:
                    i.duration_seconds = duration
                    i.error = error
                    i.summary = summarize(result) if result is not None else None
                    i.state = ItemState.DONE if result is not None and result.succeeded else ItemState.FAILED
                    i.warnings = list(warnings)
                    i.has_searchable_pdf = searchable
        self.store.update(batch_id, finish)
        return True

    def _searchable_pdf(self, batch_id: str, item: BatchItem, data: bytes, options: PipelineOptions,
                        settings: Settings) -> bool:
        from docintel.documents.searchable_pdf import OCRMode, create_searchable_pdf

        with self._lock:  # OCRmyPDF runs as one step; cancel takes effect after it
            self.live = LiveStatus(batch_id, item.item_id, 0.97, "creating searchable PDF")
        try:
            output = create_searchable_pdf(data, settings, languages=options.ocr.languages, mode=OCRMode.SKIP_TEXT)
        except DocIntelError as exc:
            self.store.update(batch_id, lambda b: [i.warnings.append(f"Searchable PDF not created: {exc.user_message}")
                                                   for i in b.items if i.item_id == item.item_id])
            return False
        self.store.save_file(batch_id, item.item_id, ".pdf", output)
        return True


# --- process-wide runner ------------------------------------------------------------------
_RUNNER: Optional[BatchRunner] = None
_RUNNER_LOCK = threading.Lock()


def get_runner(settings: Settings) -> BatchRunner:
    """The runner of this process (created on first use; later settings changes apply to new runs)."""
    global _RUNNER
    with _RUNNER_LOCK:
        if _RUNNER is None or _RUNNER.settings.batch_dir != settings.batch_dir:
            _RUNNER = BatchRunner(settings)
        else:
            _RUNNER.settings = settings
        return _RUNNER


def is_busy() -> bool:
    """True while this process is processing a batch (the desktop launcher stays running)."""
    return _RUNNER is not None and _RUNNER.busy
