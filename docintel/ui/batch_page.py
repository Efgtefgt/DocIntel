"""Batch processing: queue many files (uploads or a folder), follow progress, control the queue,
retry failures and download one ZIP with a summary.

Files are processed one at a time in the background; the page refreshes itself while a batch
runs. Queue state is saved under the batch folder (settings.batch_dir), so a batch survives a
closed window or app restart and can be resumed.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from docintel.batch.models import DEFAULT_OUTPUTS, OUTPUT_LABELS, Batch, BatchState, ItemState, OutputFormat
from docintel.batch.package import build_zip, key_details_frame
from docintel.batch.runner import BatchRunner
from docintel.batch.store import scan_folder
from docintel.core.file_detector import SUPPORTED_EXTENSIONS
from docintel.core.security import display_path
from docintel.export.file_naming import save_export
from docintel.extraction.entities import ENTITY_GROUPS
from docintel.ui import state
from docintel.ui.components import badge, card, page_header, processing_options_form

STATE_BADGES = {
    BatchState.QUEUED: ("Waiting", "info"), BatchState.RUNNING: ("Running", "info"),
    BatchState.PAUSED: ("Paused", "warn"), BatchState.INTERRUPTED: ("Interrupted", "warn"),
    BatchState.CANCELLED: ("Cancelled", "off"), BatchState.DONE: ("Finished", "ok"),
}
ITEM_TEXT = {ItemState.QUEUED: "waiting", ItemState.RUNNING: "running", ItemState.DONE: "✓ done",
             ItemState.FAILED: "✗ failed", ItemState.CANCELLED: "cancelled"}
MAX_LISTED = 15


def _duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes} min {secs:02d} s" if minutes else f"{secs} s"


def _local(moment: datetime) -> str:
    return moment.astimezone().strftime("%d %b %Y, %H:%M")


# --- new batch -------------------------------------------------------------------------------
def _choose_files() -> tuple[str, list[tuple[str, bytes]], list[tuple[str, Path, int]]]:
    """Returns (source, uploads, folder files)."""
    settings = state.settings()
    source = st.segmented_control("Files from", ["Upload files", "A folder on this computer"], key="batch_source",
                                  default="Upload files", required=True)
    if source == "Upload files":
        files = st.file_uploader("Drag and drop files here, or browse", type=list(SUPPORTED_EXTENSIONS),
                                 accept_multiple_files=True, key="batch_uploader", label_visibility="collapsed")
        uploads = [(f.name, f.getvalue()) for f in files or []]
        session = [d for d in state.uploads().values() if d.info is not None]
        if session and st.checkbox(f"Also include the {len(session)} file(s) already added in this session",
                                   key="batch_include_session"):
            uploads += [(d.name, d.data) for d in session]
        if uploads:
            st.caption(f"{len(uploads)} file(s) · {sum(len(d) for _, d in uploads) / 1_048_576:.1f} MB · "
                       f"up to {settings.max_file_size_mb:.0f} MB each. Copies are kept with the batch until you delete it.")
        return "upload", uploads, []

    folder_text = st.text_input("Folder", key="batch_folder", placeholder=r"e.g. D:\Scans\October",
                                help="Images and PDFs in this folder are processed where they are; nothing is copied. "
                                     "Large files and many files are fine: they are read one at a time.")
    recursive = st.checkbox("Include subfolders", key="batch_recursive")
    if not folder_text.strip():
        st.caption("Paste a folder path (in File Explorer: click the address bar and copy it).")
        return "folder", [], []
    folder = Path(folder_text.strip().strip('"')).expanduser()
    if not folder.is_dir():
        st.error("That folder does not exist or cannot be opened.", icon=":material/folder_off:")
        return "folder", [], []
    found, truncated = scan_folder(folder, recursive=recursive, max_files=settings.batch_max_files)
    if not found:
        st.warning("No images or PDFs found in that folder" + ("" if recursive else " (try 'Include subfolders')") + ".",
                   icon=":material/search_off:")
        return "folder", [], []
    st.caption(f"{len(found)} file(s) · {sum(s for _, _, s in found) / 1_048_576:.1f} MB in {display_path(folder)}")
    if truncated:
        st.warning(f"Only the first {settings.batch_max_files} files are included (DOCINTEL_BATCH_MAX_FILES).",
                   icon=":material/warning:")
    with st.expander("Files found", icon=":material/list:"):
        st.dataframe(pd.DataFrame([{"File": n, "Size (KB)": round(s / 1024)} for n, _, s in found]), hide_index=True)
    return "folder", [], found


def _new_batch(runner: BatchRunner) -> None:
    with card("1. Choose files"):
        source, uploads, folder_files = _choose_files()
    with card("2. Processing options", "the same for every file"):
        options = processing_options_form("batch")
    with card("3. Results package (ZIP)", "a summary and key details are always included"):
        outputs = st.pills("Include", list(OutputFormat), selection_mode="multi", default=DEFAULT_OUTPUTS,
                           key="batch_outputs", format_func=lambda o: OUTPUT_LABELS[o])
        if OutputFormat.SEARCHABLE_PDF in (outputs or []):
            st.caption("Searchable PDFs are made for PDF files only and can take as long as the OCR itself.")
        title = st.text_input("Batch name", key="batch_title", placeholder=f"Batch {datetime.now():%d %b %H:%M}")
    count = len(uploads) + len(folder_files)
    if st.button(f"Start batch ({count} file{'s' if count != 1 else ''})", type="primary", icon=":material/play_arrow:",
                 key="batch_start", disabled=not count or options is None):
        try:
            batch = runner.create(title=title.strip() or f"Batch {datetime.now():%d %b %H:%M}", options=options,
                                  outputs=outputs or [], uploads=uploads, folder_files=folder_files)
        except (ValueError, OSError) as exc:
            st.error(str(exc), icon=":material/error:")
            return
        st.session_state["batch_view_next"] = "Batches"
        st.session_state["batch_focus"] = batch.batch_id
        st.rerun()


# --- batch list ------------------------------------------------------------------------------
def _values_text(counts: dict[str, int]) -> str:
    parts = []
    for group, types in ENTITY_GROUPS.items():
        n = sum(counts.get(t, 0) for t in types)
        if n:
            parts.append(f"{n} {group.lower()}")
    return ", ".join(parts)


def _items_table(batch: Batch) -> None:
    rows = []
    for item in batch.items:
        s = item.summary
        rows.append({
            "#": item.item_id, "File": item.name, "Status": ITEM_TEXT[item.state],
            "Pages": s.pages if s else None, "Time": _duration(item.duration_seconds),
            "Confidence": s.mean_confidence if s else None,
            "Review": s.pages_to_review if s else None,
            "Type": s.document_type if s else "", "Values found": _values_text(s.value_counts) if s else "",
            "Note": item.error or "; ".join(item.warnings),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
        "Confidence": st.column_config.NumberColumn("Confidence", format="%.0f",
                                                    help="Mean engine score of OCR pages; not a probability"),
        "Review": st.column_config.NumberColumn("Review", help="OCR pages whose quality check did not say 'good'"),
        "Note": st.column_config.TextColumn("Note", help="Why a file failed, or warnings"),
    })


def _open_item(batch: Batch, runner: BatchRunner) -> None:
    done = [i for i in batch.items if i.state in (ItemState.DONE, ItemState.FAILED)]
    if not done:
        return
    with st.container(horizontal=True, vertical_alignment="bottom", wrap=True):
        pick = st.selectbox("Open one file in OCR results", [i.item_id for i in done], key=f"batch_pick_{batch.batch_id}",
                            format_func=lambda i: next(f"{x.item_id} · {x.name}" for x in done if x.item_id == i))
        if st.button("Open", key=f"batch_open_{batch.batch_id}", icon=":material/visibility:"):
            item = next(i for i in done if i.item_id == pick)
            stored = runner.store.load_result(batch.batch_id, item.item_id)
            try:
                data = runner.store.input_bytes(batch, item)
            except OSError:
                st.error("The original file is no longer available.", icon=":material/error:")
                return
            if stored is None:
                st.error("No result was saved for this file.", icon=":material/error:")
                return
            doc = state.register_upload(Path(item.name).name, data)
            state.record_result(doc.file_id, stored.to_result())
            state.go("results", focus=doc.file_id)


def _key_details(batch: Batch, runner: BatchRunner) -> None:
    show_ids = st.toggle("Show full ID numbers", key=f"batch_unmask_{batch.batch_id}",
                         help="PAN, Aadhaar, card and account numbers are masked unless switched on.")
    frame = key_details_frame(runner.store, batch, mask_ids=not show_ids)
    if frame.empty:
        st.caption("No phone numbers, e-mails, PIN/ZIP codes, dates, amounts or IDs found yet.")
        return
    groups = [g for g in ENTITY_GROUPS if (frame["group"] == g).any()]
    pick = st.pills("Show", ["All", *groups], default="All", key=f"batch_kd_{batch.batch_id}",
                    format_func=lambda g: f"{g} · {len(frame) if g == 'All' else int((frame['group'] == g).sum())}")
    shown = frame if pick in (None, "All") else frame[frame["group"] == pick]
    st.dataframe(shown.drop(columns=["group"]).rename(columns=str.capitalize), hide_index=True)


def _controls(batch: Batch, runner: BatchRunner) -> None:
    bid = batch.batch_id
    busy = runner.busy_with(bid)
    redo = batch.count(ItemState.FAILED, ItemState.CANCELLED)
    with st.container(horizontal=True, wrap=True, vertical_alignment="center"):
        if busy and batch.state is not BatchState.PAUSED:
            if st.button("Pause", key=f"batch_pause_{bid}", icon=":material/pause:",
                         help="Stops after the file being processed now"):
                runner.pause(bid)
                st.rerun()
        if not busy and batch.state in (BatchState.PAUSED, BatchState.INTERRUPTED) and batch.count(ItemState.QUEUED):
            if st.button("Resume", key=f"batch_resume_{bid}", icon=":material/play_arrow:", type="primary"):
                runner.start(bid)
                st.rerun()
        if busy or (batch.state in (BatchState.PAUSED, BatchState.INTERRUPTED) and batch.count(ItemState.QUEUED)):
            if st.button("Cancel", key=f"batch_cancel_{bid}", icon=":material/stop:",
                         help="Stops between pages; files not finished are marked cancelled"):
                runner.cancel(bid)
                st.rerun()
        if not busy and redo:
            stronger = st.checkbox("with stronger settings", key=f"batch_strong_{bid}",
                                   help="Higher PDF resolution (300 DPI), more OCR attempts and the smart preset. Slower.")
            if st.button(f"Retry {redo} failed/cancelled", key=f"batch_retry_{bid}", icon=":material/refresh:"):
                runner.retry_failed(bid, stronger_settings=stronger)
                st.rerun()
        if batch.finished:
            def make_zip(b: Batch = batch) -> bytes:
                buffer = io.BytesIO()
                build_zip(runner.store, runner.store.get(b.batch_id) or b, buffer)
                return buffer.getvalue()
            st.download_button("Download ZIP", make_zip, f"DocIntel_batch_{bid}.zip", "application/zip",
                               key=f"batch_zip_{bid}", icon=":material/download:", on_click="ignore", type="primary"
                               if batch.state is BatchState.DONE else "secondary")
            if st.button("Save ZIP to Documents", key=f"batch_save_{bid}", icon=":material/save:", type="tertiary",
                         help=f"Writes the ZIP into {display_path(state.settings().export_dir)}"):
                buffer = io.BytesIO()
                build_zip(runner.store, batch, buffer)
                path = save_export(buffer.getvalue(), f"DocIntel_batch_{bid}.zip", state.settings().export_dir)
                st.success(f"Saved {display_path(path)}", icon=":material/check_circle:")
        if not busy:
            if st.button("Delete", key=f"batch_delete_{bid}", icon=":material/delete:", type="tertiary",
                         help="Removes this batch, its copied files and results from this computer"):
                st.session_state[f"batch_confirm_{bid}"] = True
    if st.session_state.get(f"batch_confirm_{bid}"):
        st.warning("Delete this batch and its results? Original files in a folder are not touched.",
                   icon=":material/warning:")
        with st.container(horizontal=True):
            if st.button("Yes, delete", key=f"batch_delete_yes_{bid}", type="primary"):
                runner.delete(bid)
                st.session_state.pop(f"batch_confirm_{bid}", None)
                st.rerun()
            if st.button("Keep it", key=f"batch_delete_no_{bid}"):
                st.session_state.pop(f"batch_confirm_{bid}", None)
                st.rerun()


def _batch_card(batch: Batch, runner: BatchRunner) -> None:
    total, done, failed = len(batch.items), batch.count(ItemState.DONE), batch.count(ItemState.FAILED)
    source = "folder" if batch.source == "folder" else "uploaded files"
    with card(batch.title, f"{_local(batch.created_at)} · {total} file(s) from {source}", key=f"batch_card_{batch.batch_id}"):
        text, kind = STATE_BADGES[batch.state]
        with st.container(horizontal=True, vertical_alignment="center", wrap=True):
            badge(text, kind)  # type: ignore[arg-type]
            st.caption(f"{done} done · {failed} failed · {batch.count(ItemState.CANCELLED)} cancelled · "
                       f"{batch.count(ItemState.QUEUED, ItemState.RUNNING)} to go")
        live = runner.live if runner.live and runner.live.batch_id == batch.batch_id else None
        fraction = batch.finished / total if total else 1.0
        label = f"{batch.finished} of {total} files"
        if live is not None:
            current = next((i for i in batch.items if i.item_id == live.item_id), None)
            fraction = (batch.finished + live.fraction) / total if total else 1.0
            label += f" · now: {current.name if current else ''} ({live.message})"
        eta = batch.eta_seconds()
        if eta and batch.active:
            label += f" · about {_duration(eta)} left"
        st.progress(min(1.0, fraction), text=label)
        if batch.message:
            st.caption(batch.message)
        _controls(batch, runner)
        with st.expander("Files", icon=":material/list:", expanded=batch.batch_id == st.session_state.get("batch_focus")):
            _items_table(batch)
            _open_item(batch, runner)
        if done or failed:
            with st.expander("Key details from all files", icon=":material/manage_search:"):
                _key_details(batch, runner)


def _batch_list(runner: BatchRunner) -> None:
    batches = runner.store.list()
    if not batches:
        st.info("No batches yet. Start one under **New batch**.", icon=":material/info:")
        return
    interrupted = [b for b in batches if b.state is BatchState.INTERRUPTED]
    if interrupted:
        st.warning(f"{len(interrupted)} batch(es) stopped when DocIntel closed. Press **Resume** to continue where "
                   "they stopped.", icon=":material/history:")
    for batch in batches[:MAX_LISTED]:
        _batch_card(batch, runner)
    if len(batches) > MAX_LISTED:
        st.caption(f"{len(batches) - MAX_LISTED} older batch(es) not shown; delete finished batches to see them.")


def render() -> None:
    page_header("Batch processing", "Process many images and PDFs, or a whole folder, one after another in the "
                "background. Pause, cancel or retry failed files, then download one ZIP with a summary.")
    runner = state.batch_runner()
    if "batch_view_next" in st.session_state:
        st.session_state["batch_view"] = st.session_state.pop("batch_view_next")
    has_batches = bool(runner.store.list())
    view = st.segmented_control("View", ["Batches", "New batch"], key="batch_view",
                                default="Batches" if has_batches else "New batch", required=True,
                                label_visibility="collapsed")
    if view == "New batch":
        _new_batch(runner)
        return
    refresh = 2 if runner.busy else None  # refresh only while something runs

    @st.fragment(run_every=refresh)
    def live_list() -> None:
        _batch_list(runner)
        if refresh and not runner.busy:
            st.rerun()  # the last batch just finished: one full rerun turns auto-refresh off

    live_list()
    st.caption(f"Batches are kept in {display_path(state.settings().batch_dir)} until you delete them. "
               "OCR is never 100% accurate; check important values against the originals.")
