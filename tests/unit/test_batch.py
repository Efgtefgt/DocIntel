"""Batch processing: queue, pause/cancel/resume, retry, recovery, folder input and the ZIP package."""

from __future__ import annotations

import io
import os
import zipfile
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from docintel.batch import runner as runner_module
from docintel.batch.models import BatchState, ItemState, OutputFormat
from docintel.batch.options import options_from_dict, options_to_dict, stronger
from docintel.batch.package import build_zip, key_details_frame, summary_frame
from docintel.batch.runner import BatchRunner
from docintel.batch.store import BatchStore, scan_folder
from docintel.core.pipeline import PipelineOptions
from docintel.evaluation import synthetic_data as synth
from docintel.extraction.templates import TemplateStore
from docintel.ocr.base_engine import OCROptions
from tests.conftest import make_factory
from tests.fakes import FakeEngine

TEXT = "Contact: info@example.com\nPhone: 080 1234 5678\nZIP: 62704\nTotal: Rs. 1,250.00"


@pytest.fixture
def batch_settings(settings, tmp_path):
    return settings.model_copy(update={"batch_dir": tmp_path / "batches", "default_engine": "fake",
                                       "batch_min_free_memory_mb": 0})


def _runner(settings, engine: FakeEngine | None = None) -> BatchRunner:
    engine = engine or FakeEngine(text=TEXT)
    return BatchRunner(settings, factory_builder=lambda s: make_factory(s, engine))


def _options(**kw) -> PipelineOptions:
    return PipelineOptions(engine="fake", ocr=OCROptions(languages=("eng",)), preprocess_preset="original", **kw)


def _png() -> bytes:
    return synth.image_bytes(synth.render_text_image(["Hello"]))


def test_batch_processes_all_files_and_one_bad_file_does_not_stop_it(batch_settings):
    runner = _runner(batch_settings)
    files = [("a.png", _png()), ("broken.pdf", b"%PDF-1.4 not really"), ("b.png", _png())]
    batch = runner.create(title="Test", options=_options(), outputs=[OutputFormat.TEXT, OutputFormat.JSON], uploads=files)
    assert runner.wait(30)
    batch = runner.store.get(batch.batch_id)
    assert batch.state is BatchState.DONE and batch.finished_at is not None
    states = [i.state for i in batch.items]
    assert states == [ItemState.DONE, ItemState.FAILED, ItemState.DONE]
    assert batch.items[1].error  # a reason is shown, not a crash
    summary = batch.items[0].summary
    assert summary.pages == 1 and summary.value_counts["email"] == 1 and summary.value_counts["pin_code"] == 1
    assert runner.store.load_result(batch.batch_id, "001").document.pages[0].raw_text
    assert batch.eta_seconds() is None  # nothing left


def test_zip_package_contents(batch_settings):
    runner = _runner(batch_settings)
    batch = runner.create(title="Zip", options=_options(), outputs=[OutputFormat.TEXT, OutputFormat.JSON],
                          uploads=[("scan one.png", _png()), ("bad.png", b"nope")])
    runner.wait(30)
    batch = runner.store.get(batch.batch_id)
    buffer = io.BytesIO()
    count = build_zip(runner.store, batch, buffer)
    names = zipfile.ZipFile(buffer).namelist()
    assert count == len(names)
    root = names[0].split("/")[0]
    assert {f"{root}/summary.csv", f"{root}/summary.html", f"{root}/key_details.csv", f"{root}/README.txt"} <= set(names)
    assert f"{root}/documents/001_scan one/scan one.txt" in names
    assert f"{root}/documents/001_scan one/scan one.json" in names
    assert not any("002_" in n for n in names)  # failed file: listed in the summary only
    summary = pd.read_csv(io.BytesIO(zipfile.ZipFile(buffer).read(f"{root}/summary.csv")))
    assert list(summary["status"]) == ["done", "failed"] and summary.loc[0, "email"] == 1
    details = key_details_frame(runner.store, batch)
    assert {"080 1234 5678", "info@example.com", "62704"} <= set(details["value"])


def test_key_details_mask_personal_ids(batch_settings):
    runner = _runner(batch_settings, FakeEngine(text="Aadhaar: 4991 1866 5246"))
    batch = runner.create(title="IDs", options=_options(), outputs=[], uploads=[("id.png", _png())])
    runner.wait(30)
    batch = runner.store.get(batch.batch_id)
    assert "4991 1866 5246" not in set(key_details_frame(runner.store, batch)["value"])
    assert "4991 1866 5246" in set(key_details_frame(runner.store, batch, mask_ids=False)["value"])


def test_cancel_stops_between_pages(batch_settings):
    engine = FakeEngine(text=TEXT, delay=0.3)
    runner = _runner(batch_settings, engine)
    pdf = synth.make_scanned_pdf([synth.render_text_image(["Page"])] * 6)
    batch = runner.create(title="Cancel", options=_options(pdf_mode=__import__("docintel.documents.document_processor",
                                                                               fromlist=["PDFMode"]).PDFMode.OCR_ALL),
                          outputs=[], uploads=[("long.pdf", pdf), ("next.png", _png())])
    for _ in range(100):
        if engine.calls:
            break
        __import__("time").sleep(0.05)
    runner.cancel(batch.batch_id)
    assert runner.wait(30)
    batch = runner.store.get(batch.batch_id)
    assert batch.state is BatchState.CANCELLED
    assert [i.state for i in batch.items] == [ItemState.CANCELLED, ItemState.CANCELLED]
    assert len(engine.calls) < 6  # stopped before the last page


def test_pause_after_current_file_then_resume(batch_settings):
    engine = FakeEngine(text=TEXT, delay=0.2)
    runner = _runner(batch_settings, engine)
    batch = runner.create(title="Pause", options=_options(), outputs=[], uploads=[(f"{n}.png", _png()) for n in "abc"])
    for _ in range(100):
        if engine.calls:
            break
        __import__("time").sleep(0.02)
    runner.pause(batch.batch_id)
    runner.wait(30)
    paused = runner.store.get(batch.batch_id)
    assert paused.state is BatchState.PAUSED
    assert paused.count(ItemState.DONE) >= 1 and paused.count(ItemState.QUEUED) >= 1
    runner.start(batch.batch_id)
    runner.wait(30)
    assert runner.store.get(batch.batch_id).count(ItemState.DONE) == 3


def test_retry_failed_with_stronger_settings(batch_settings):
    runner = _runner(batch_settings)
    batch = runner.create(title="Retry", options=_options(), outputs=[], uploads=[("bad.png", b"nope"), ("ok.png", _png())])
    runner.wait(30)
    assert runner.retry_failed(batch.batch_id, stronger_settings=True) == 1
    runner.wait(30)
    retried = runner.store.get(batch.batch_id).items[0]
    assert retried.attempts == 2 and retried.stronger and retried.state is ItemState.FAILED
    assert runner._settings_for(retried).pdf_render_dpi >= 300
    assert stronger(_options()).preprocess_preset == "smart"


def test_interrupted_batch_is_recovered_and_resumed(batch_settings):
    runner = _runner(batch_settings)
    batch = runner.create(title="Crash", options=_options(), outputs=[], uploads=[("a.png", _png()), ("b.png", _png())],
                          start=False)

    def crashed(b):
        b.state, b.owner_pid = BatchState.RUNNING, 999_999_999  # a process that no longer exists
        b.items[0].state = ItemState.RUNNING
    runner.store.update(batch.batch_id, crashed)
    fresh = _runner(batch_settings)  # a new app start recovers on creation
    recovered = fresh.store.get(batch.batch_id)
    assert recovered.state is BatchState.INTERRUPTED and recovered.items[0].state is ItemState.QUEUED
    fresh.start(batch.batch_id)
    fresh.wait(30)
    assert fresh.store.get(batch.batch_id).count(ItemState.DONE) == 2


def test_folder_input_scan_and_missing_file(batch_settings, tmp_path):
    folder = tmp_path / "scans"
    (folder / "sub").mkdir(parents=True)
    (folder / "a.png").write_bytes(_png())
    (folder / "sub" / "b.png").write_bytes(_png())
    (folder / "notes.txt").write_text("skip me")
    (folder / ".hidden.png").write_bytes(_png())
    flat, _ = scan_folder(folder, recursive=False, max_files=10)
    deep, _ = scan_folder(folder, recursive=True, max_files=10)
    limited, truncated = scan_folder(folder, recursive=True, max_files=1)
    assert [n for n, _, _ in flat] == ["a.png"]
    assert [n for n, _, _ in deep] == ["a.png", "sub/b.png"]
    assert len(limited) == 1 and truncated

    runner = _runner(batch_settings)
    batch = runner.create(title="Folder", options=_options(), outputs=[], folder_files=deep, start=False)
    (folder / "sub" / "b.png").unlink()
    runner.start(batch.batch_id)
    runner.wait(30)
    batch = runner.store.get(batch.batch_id)
    assert batch.source == "folder" and batch.items[1].name == "sub/b.png"
    assert batch.items[0].state is ItemState.DONE
    assert batch.items[1].state is ItemState.FAILED and "no longer exists" in batch.items[1].error
    assert str(tmp_path) not in summary_frame(batch).to_csv()  # no local paths in the summary


def test_options_round_trip_and_missing_template(batch_settings):
    tmpl = TemplateStore(batch_settings.templates_dir).load("invoice")
    options = replace(_options(extract_tables=True, page_range="1-2"), template=tmpl, custom_steps=("grayscale",))
    data = options_to_dict(options)
    restored, warnings = options_from_dict(data, batch_settings.templates_dir)
    assert restored == options and not warnings
    data["template"] = "no_such_template"
    restored, warnings = options_from_dict(data, batch_settings.templates_dir)
    assert restored.template is None and warnings


def test_limits_delete_and_memory_wait(batch_settings, monkeypatch):
    runner = _runner(batch_settings.model_copy(update={"batch_max_files": 1}))
    with pytest.raises(ValueError):
        runner.create(title="Too many", options=_options(), outputs=[], uploads=[("a.png", _png()), ("b.png", _png())])
    with pytest.raises(ValueError):
        runner.create(title="Empty", options=_options(), outputs=[])

    monkeypatch.setattr(runner_module, "MEMORY_WAIT_SECONDS", 0.1)
    low_memory = _runner(batch_settings.model_copy(update={"batch_min_free_memory_mb": 10**9}))
    batch = low_memory.create(title="Memory", options=_options(), outputs=[], uploads=[("a.png", _png())])
    assert low_memory.wait(30)
    assert low_memory.store.get(batch.batch_id).count(ItemState.DONE) == 1  # waited, then continued
    assert low_memory.delete(batch.batch_id)
    assert low_memory.store.get(batch.batch_id) is None
    assert not os.path.exists(low_memory.store.root / batch.batch_id)


def test_store_rejects_unsafe_ids(tmp_path):
    store = BatchStore(tmp_path)
    for bad in ("..", "../x", "20260101-000000-zzzzzz", "C:\\Windows"):
        with pytest.raises(ValueError):
            store.folder(bad)
    assert store.list() == []
    assert isinstance(Path(store.root), Path)
