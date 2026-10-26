# Known limitations and future improvements

## Limitations

**OCR**

* Accuracy depends heavily on scan quality, fonts, language models and layout. No accuracy figure
  for real documents has been measured; the bundled benchmark is synthetic.
* Handwriting is not supported in any useful way by Tesseract.
* Confidence values are engine-specific heuristics. Tesseract's and PaddleOCR's scores are not
  comparable, and neither is a probability of correctness.
* Multi-column layouts rely on Tesseract's page segmentation, and reading order can be wrong on
  complex pages.

**PaddleOCR**

* It needs its own environment, because it replaces the headless OpenCV and downgrades protobuf.
* It reads one language per run, and models are downloaded on first use (needs internet once).
* A timeout cannot stop inference that is already running (Python threads cannot be killed). The
  timed-out call keeps using CPU until it finishes.
* The English model drops some sentence-final full stops and occasionally adds one at a line end.

**Searchable PDF**

* *Redo OCR* is DocIntel's own implementation; OCRmyPDF's `--redo-ocr` doubled every word
  (220 → 440 words on a test page, in OCRmyPDF 16.7 and 16.13). It removes text objects drawn
  entirely in invisible render mode 3, which is how OCR layers are stored. Visible text is never
  removed. A text object mixing visible and invisible glyphs is kept as it is, so in that rare
  case the old invisible part can remain.
* OCRmyPDF deskew reported a 0° angle on a 5°-tilted page rendered at about 120 DPI, but it
  corrected the same tilt on a 300 DPI scan.
* OCRmyPDF 16.7 requires `pikepdf<10`, so pikepdf is pinned to 9.11.0.

**User interface**

* The sidebar theme selector changes Streamlit's `theme.*` options through its internal config
  API, because Streamlit (1.65) has no public way for a widget to switch the theme. The options
  are process-wide, so all open tabs follow the latest choice, and the selector doesn't follow
  the operating-system theme automatically (`DOCINTEL_UI_THEME` sets the default). If a future
  Streamlit release moves that API, the app logs a warning and keeps the light theme.
* AppTest can't simulate clicking `st.navigation` links. The UI tests open pages with
  `switch_page`, and the real navigation clicks were exercised in the Edge browser checks.
* On windows narrower than about 640 px, Streamlit shows navigation as an overlay that has to be
  closed after choosing a page; wide data tables scroll inside their own frame.

**Installation**

* In a Python shared with other projects (no virtual environment), use the constrained install in
  README section 9. Camelot is then usually left out, because it would add a second OpenCV build.
* Streamlit's uploader limit (`.streamlit/config.toml`, `maxUploadSize`) must be raised together
  with `DOCINTEL_MAX_FILE_SIZE_MB`.

**Smart OCR pipeline**

* The analysis thresholds were tuned on synthetic images. On the never-examined synthetic set,
  mean CER is 0.19 (vs. 0.49 without preprocessing); real-world accuracy has not been measured.
  Add your own scans to the benchmark (docs/EVALUATION.md).
* Coverage estimates characters from connected components. Touching or broken glyphs, logos,
  handwriting and non-Latin scripts make it less accurate, so treat the verdict as a review hint.
* Retries cost one full OCR run each and are capped by `DOCINTEL_MAX_OCR_ATTEMPTS` (default 3).
* Suspicious-word suggestions cover common letter/digit confusions only. There is no dictionary or
  language model, so misspelled real words are not detected.
* No local AI is used. The Ollama integration is intentionally disabled, with no models downloaded.

**Document intelligence**

* Document types come from keyword rules for 14 common types (invoice, receipt, purchase order,
  bank statement, payslip, identity document, resume, letter, incident report, medical, utility
  bill, contract, delivery note, certificate). Anything else is `unknown`, or `form` when it has
  several labelled fields. The rules are English-only.
* Accuracy of the type and of the values has been measured only on synthetic documents
  (docs/EVALUATION.md), not on real ones.
* Indian formats are covered (PAN, GSTIN, IFSC, Aadhaar, PIN codes, vehicle numbers, lakh digit
  grouping); other countries' IDs are not, apart from IBAN and payment cards.
* PAN, IFSC and vehicle numbers have no checksum, so without a label they stay `uncertain`. A
  12-digit number passes the Aadhaar checksum by chance 1 time in 10, so without an Aadhaar label
  it stays `uncertain` too.
* "$" is not assumed to be US dollars, and an amount without a currency symbol is reported without one.
* When OCR merges two columns and the second label is not a known phrase, only its last word is
  used as the label ("Blood Group" may become "Group").
* Masking applies to the Document intelligence page and its downloads. The full-document JSON
  export on OCR results contains the text unmasked.

**Preprocessing**

* Orientation correction needs `osd.traineddata` and enough text on the page.
* Deskew handles up to ±15° by default.
* Border cropping is heuristic.

**Tables**

* OCR-based extraction finds at most one table per page.
* Merged cells in scanned grids are reported as boundary-crossing words, not reconstructed.
* Whitespace-inferred tables are always marked uncertain.
* Camelot and PyMuPDF only work on digital PDFs.

**Structured extraction**

* Label-based rules work for "Label: value" style documents. Free-form prose needs custom regex
  patterns.
* Low-confidence detection matches word text, not exact character positions.
* Dates with two-digit years or an ambiguous day/month order are flagged as uncertain, not resolved.

**Platform and session**

* Statistics and documents exist only for the current browser session. Settings changed in the UI
  are not persisted; use `.env` for permanent settings.
* The REST API has no authentication and is meant for localhost only.
* TIFF support covers formats that Pillow can decode. Some compressions (for example certain
  JPEG-in-TIFF or JBIG variants) may be rejected as corrupt.

## Future improvements

* Process-based OCR workers, so timeouts can terminate PaddleOCR inference.
* PaddleOCR 3.x support (new API) and more language models.
* Multi-table detection on scanned pages, and merged-cell reconstruction.
* Layout analysis (columns, headers and footers) for better reading order.
* Per-field confidence that combines OCR word scores with label evidence.
* Persisted session history (opt-in), and saving or deleting templates from the API.
* An optional local LLM for free-text field extraction, disabled by default and only adopted
  if it measurably improves results on a labelled dataset.
