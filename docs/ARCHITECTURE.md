# Architecture

## Package layout

```
app.py                     Streamlit entry point (navigation only)
docintel/
  config/                  Settings (pydantic-settings, DOCINTEL_* env vars) and JSON logging
  core/                    exceptions, security (safe names, path checks, shell-free subprocess),
                           file_detector (magic bytes), file_validator, result_models,
                           deadline, temp_files, pipeline (orchestration)
  ocr/                     base_engine (interface), tesseract_engine, paddle_engine, engine_factory
  preprocessing/           image_loader, enhancement, denoising, deskew, preprocessing_pipeline
  documents/               pdf_reader, pdf_renderer, document_processor (page loop), searchable_pdf,
                           text_layer (removes invisible OCR text for Redo OCR; pikepdf)
  extraction/              text_normalizer, table_extractor, templates, field_validators,
                           structured_extractor, ocr_review, entities (typed values + checksums),
                           key_values, classifier (document type), document_analysis
  export/                  txt / json / csv / excel exporters, file_naming
  batch/                   models (Batch, BatchItem), options (PipelineOptions <-> JSON), store (on-disk
                           queue state, inputs, results), runner (background worker), package (ZIP + summary)
  evaluation/              metrics (CER/WER), benchmark CLI, synthetic_data generators
  api/                     FastAPI app factory, routes, schemas
  ui/                      one module per page + state, theme, health checks, shared components, assets/
app_pages/                 thin page scripts registered with st.navigation in app.py
scripts/                   check_environment.py, generate_samples.py, install/run PowerShell scripts
tests/unit, tests/integration, tests/fakes.py (fake OCR engine)
data/templates             extraction templates (JSON)
data/samples               synthetic benchmark images with ground truth, sample PDFs
```

The packages live under `docintel/` rather than at the top level, so generic names such as
`config`, `api` or `ocr` cannot collide with installed third-party packages.

## Pipeline

```
upload bytes
  -> validate_upload          size, magic-byte type, Pillow header + full decode, pixel/dimension
                              limits (decompression bombs), PDF open/encryption/page limits
  -> page selection           parse_page_range ("1-3,5", "8-")
  -> per page (sequential)
       PDF page:  analyze_page -> embedded text usable?  yes -> use it
                                                         no  -> render (pixel-capped DPI) -> OCR
       image frame: OCR
       OCR step:  [smart preset] analyze image (preprocessing/analysis.py) -> smart_plan(steps + reasons)
                  -> preprocessing pipeline -> [remove ruling lines if tables requested]
                  -> engine.recognize (timeout = min(OCR timeout, remaining document budget))
                  -> evaluate (ocr/quality.py): coverage = recognised chars / estimated glyphs,
                     verdict good | review | poor
                  -> [smart, verdict not good] up to max_ocr_attempts-1 alternative plans,
                     keep the best score (confidence x coverage); all attempts recorded
                  -> QualityReport on the page; [tables from OCR words of the kept attempt]
  -> normalization            raw_text kept, normalized_text + list of applied rules
  -> tables (optional)        PyMuPDF / Camelot for embedded-text pages, OCR tables collected above
  -> analysis (default on)    per page: entities (regex + checksums, highest-priority type wins,
                              no overlaps) -> key-value pairs; whole text: rule-based document type
                              + matching templates. Failure here never fails the document.
  -> structured (optional)    template rules -> Pydantic validation -> statuses
  -> PipelineResult           document, tables, analysis, structured, per-stage records, errors,
                              timing, RSS
```

### Document intelligence rules

* Values are verbatim spans of the text. `normalized` is filled only when no guess is needed
  (an ambiguous date or an IFSC with the letter O stays empty).
* Status: `valid` needs the format plus a checksum (GSTIN, Aadhaar Verhoeff, card Luhn, IBAN
  mod 97, calendar date) or a matching label; a pattern alone is `uncertain`; failing the
  checksum is `invalid`. Words below the OCR confidence threshold make a value `uncertain`.
* Labels come from the same line. When OCR merges two columns ("Basic: 40,000 HRA: 16,000"),
  the label is taken from the capitalised words before the separator, cut to a known label
  phrase where possible.
* The classifier scores title phrases (first five lines weigh most), keywords and entity hints.
  The score is evidence strength, not a probability; weak or close results say so, and below
  the threshold the type is `unknown`. No language model is involved.

Every stage records a `StageRecord` (status, duration, detail). Failures become `StageError`
objects with a stable `code` and a user-safe message. A failing page does not stop the other
pages. When the document deadline expires, the remaining pages are marked as skipped.

### Batch processing

`BatchRunner` (one per process, `get_runner`) owns a single worker thread that takes queued
batches in order and processes their files **one at a time**, reusing one engine set per batch
(closed afterwards so PaddleOCR memory is released).

* **State on disk.** `BatchStore` writes `batch.json` atomically after every change; uploaded files
  are copied to `inputs/`, folder inputs are read in place; each file's full result is
  `results/<item>.json` (the `PipelineResult` as JSON). Batches survive app restarts.
* **Pause** is honoured after the current file. **Cancel** is raised from the pipeline's progress
  callback (`BatchCancelled`, deliberately not a `DocIntelError`, so the pipeline does not record it
  as a page failure and carry on), so it stops between pages; the partial file is discarded.
* **Failures are per file.** Validation errors, timeouts, unreadable files and unexpected exceptions
  mark only that file failed with a reason. Retry re-queues failed and cancelled files, optionally
  with stronger settings (300 DPI, 4 OCR attempts, smart preset, 1.5x time limit).
* **Recovery.** A batch left `running` by a process that no longer exists (`owner_pid`) is marked
  `interrupted` on the next start; its running file goes back to the queue.
* **Memory.** Before each file, free RAM below `batch_min_free_memory_mb` makes the worker wait (up
  to a minute) instead of risking an out-of-memory crash.
* **Privacy.** Summaries, the ZIP and the API show file names (relative to the chosen folder), never
  absolute paths; personal IDs are masked in key_details.csv by default.
* The desktop launcher does not stop the server while `runner.is_busy()`.

## OCR engine interface

`OCREngine` (in `ocr/base_engine.py`) defines `check_availability`, `available_languages`,
`recognize(image, OCROptions) -> OCRResult` and an optional `detect_orientation`. Each engine
declares `supports_psm`, `supports_multiple_languages`, `is_deep_learning` and `confidence_scale`.

To add an engine, subclass `OCREngine` and add it to `ENGINE_CLASSES` in `engine_factory.py`.
The pipeline, UI and API select engines by name, so nothing else needs to change.

Engines are constructed lazily. `EngineFactory.get()` checks availability and releases any other
loaded deep-learning engine.

Tesseract is called through `run_subprocess` with an argument list and `shell=False`. Language
codes are validated against `^[A-Za-z0-9_]{2,32}$` and against the installed data before they
reach the command line.

## Key decisions

* **Native Tesseract CLI instead of pytesseract:** gives reliable timeouts, paths with spaces,
  one fewer dependency, and TSV output parsed directly into words with boxes and confidences.
* **Smart preprocessing (default):** each page is analysed first (text height, skew, rotation,
  blur, noise, contrast, lighting variation, dark background, dark border). Only the steps the
  analysis calls for are run, each with a recorded reason. Orientation is measured once, on a
  cleaned copy, and reused.
* **Output evaluation does not trust confidence alone.** Expected text is estimated from
  glyph-sized connected components on a shadow-corrected copy of the page, so missing text is
  detected even when the engine reports high confidence.
* **Suspicious-text review** (`extraction/ocr_review.py`) flags low-confidence words, letter/digit
  mix-ups (with suggestions such as `Inv0ice` → `Invoice`), unusual and repeated characters. It
  never changes the OCR output; accepted suggestions go into the user's separately stored edits.
* **Conditional preprocessing:**
  * Upscaling is driven by the measured median glyph height, not by image size.
  * Deskew only runs above 0.3°.
  * Sharpening only runs when the Laplacian variance shows blur.
  * The `auto` preset denoises only when measured noise is high.
  * Denoise runs before upscaling, so noise is measured and removed at native resolution.
  * These rules came out of benchmark measurements (see EVALUATION.md).
* **Embedded text first:** for each PDF page, the embedded text is used unless it is missing,
  garbled, or the page is mostly an image.
* **Tables from scans:** ruled grids are detected with morphology. The lines are hidden before
  OCR, and words are assigned to cells by their box centres. Without visible lines, columns are
  inferred from whitespace and always flagged as uncertain. Empty cells stay empty, and words
  crossing a boundary are reported.
* **Structured extraction:** deterministic label and regex rules only. Conflicting candidates
  are never auto-resolved, and values are never generated.
* **Privacy:** uploads stay in memory. Logs contain metadata only (sizes, counts, durations,
  codes) and never document text. Temp directories are removed in `finally` blocks.

## UI design system

**Navigation.** `app.py` registers the page scripts in `app_pages/` with `st.navigation`, in
grouped sections (Process, Review, System). Each page script is a thin call into its module in
`docintel/ui/`. Switching pages in code uses `state.go()`, which wraps `st.switch_page`.

**Native styling, almost no CSS.** All colours are Streamlit theme options: text, borders,
status badges, chart palette, dataframe headers and the sidebar. Native elements do the rest:
`st.metric(border=True)`, `st.badge`, bordered and horizontal containers, `st.logo`.

The only custom HTML is the low-confidence word highlighting, because there is no native
equivalent. It uses inline styles from the theme tokens and HTML-escaped text. There are no
selectors that depend on Streamlit's DOM structure.

**Theme.** `ui/theme.py` defines both palettes. Their WCAG AA contrast is checked in
`tests/unit/test_ui_theme.py` (4.5:1 for text and status colours, 3:1 for active UI).
`theme.apply()` writes the active palette into Streamlit's `theme.*` options before rendering,
and the app reruns once so the browser receives it.

The sidebar selector writes to a separate plain session key via `on_change`. Widget state is
discarded when `st.rerun()` cuts a run short before the widget is drawn, so the widget's own key
can't be the source of truth.

**Shared components** (`ui/components.py`):

* `upload_panel`: used on every document page, files shared across pages
* `document_picker`, `processing_options_form`, `default_options` (for one-click processing)
* `run_with_progress`, `badge`, `card`, `health_panel`

`ui/health.py` holds the real but lightweight dependency checks behind the dashboard and About
pages. Executables are run once with `--version` and cached; models are never loaded.

**Reruns and caching.** Viewer controls (page, zoom, rotation, original/processed view, tabs)
only re-render stored results and cached images. OCR runs only from explicit buttons, and a test
asserts this. Cached previews are keyed by file ID, page and settings, so documents can't mix.

Dashboard charts use Altair with a palette checked by the dataviz validator.

## Security controls

| Threat | Control |
| --- | --- |
| Spoofed file type | content signature detection; extension only produces a warning |
| Decompression bomb / huge images | dimensions checked from the header before decoding; pixel and side limits; PDF render DPI auto-reduced |
| Corrupt or truncated files | full decode during validation, mapped to `corrupt_file` |
| Shell injection | no `shell=True`; argument lists; validated language codes, PSM and OEM |
| Path traversal | `safe_filename` + `ensure_within` for templates and exports; exclusive-create writes |
| CSV/Excel formula injection | cells starting with `= + - @` (except numbers) are prefixed with `'` |
| Resource exhaustion | file size, page count, frame count, per-OCR and per-document timeouts, bounded workers, API body limit |
| Data leakage | no document content in logs; no telemetry (`gatherUsageStats=false`); localhost binding |
