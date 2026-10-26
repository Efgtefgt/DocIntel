# DocIntel: Local Image-to-Text and Document Intelligence

DocIntel extracts text from images and PDFs, improves recognition through image preprocessing,
extracts tables and template-defined fields, and exports TXT, JSON, CSV, XLSX and (optionally)
searchable PDF.

* Runs on **Windows 10/11, CPU only**, tuned for an i5 laptop with 8 GB RAM.
* Uses **open-source software only**: no paid APIs, no API keys, and no internet connection during processing.
* Uploaded documents stay **in memory**. Nothing is saved unless you press a save button.
* OCR output is **never 100% accurate**. Confidence scores are engine-specific indicators,
  missing fields stay empty, and uncertain results are flagged rather than guessed.

| Area | Status |
| --- | --- |
| Basic OCR (Tesseract), languages, PSM, TXT export | Implemented and tested with Tesseract 5.4 |
| Smart OCR pipeline (default): image analysis, automatic preprocessing plan, output evaluation (coverage + confidence), bounded retries | Implemented and tested; measured CER 0.486 → 0.189 on an unseen degraded set (docs/EVALUATION.md) |
| Preprocessing steps (grayscale, invert, lighting correction, thresholds, denoise, contrast, deskew, orientation, sharpen, crop, upscale) | Implemented and tested |
| OCR review: suspicious-word flags with suggestions (applied only to your edits), per-line confidence, accuracy check against reference text, engine comparison | Implemented and tested |
| Local AI assistant (Ollama) | Intentionally disabled: nothing downloaded, no language model is called |
| PDF: embedded text, scanned/mixed pages, page ranges, page-by-page progress | Implemented and tested |
| Searchable PDF (OCRmyPDF + Ghostscript 10.08) | Implemented and tested (skip text, **redo OCR**, force OCR, mixed PDFs, deskew, auto-rotate) |
| Tables: PyMuPDF, Camelot lattice/stream, OCR-based grid/whitespace | Implemented and tested |
| Structured extraction (templates, validation, conflicts, corrections) | Implemented and tested |
| Document intelligence (no template needed): document type, e-mails, phones, dates, times, amounts, percentages, numbers, IDs, PAN/GSTIN/IFSC/Aadhaar/card/IBAN/account numbers with checksum checks, "Label: value" pairs, masking of personal IDs | Implemented and tested; on 12 synthetic typed documents after OCR: type 12/12, value recall 0.90, precision 0.94 (docs/EVALUATION.md) |
| Streamlit 1.65 UI (10 pages in grouped navigation, light and dark themes) | Implemented. Covered by headless UI tests; checked in Edge at 1920, 1440, 1366 and 420 px wide in both themes |
| REST API (FastAPI 0.124) | Implemented and tested |
| PaddleOCR engine (optional, PaddleOCR 2.10 / PaddlePaddle 3.0 CPU) | Implemented and tested end-to-end with real models |
| CER/WER evaluation and benchmark | Implemented and tested |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/EVALUATION.md](docs/EVALUATION.md),
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) and [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

---

## Install as a Windows app (easiest)

1. Download **DocIntel-Setup-<version>.exe** from the [Releases page](../../releases/latest).
2. Run it. Windows SmartScreen may say "Windows protected your PC" because the installer is not
   code-signed: click **More info → Run anyway**.
3. Start **DocIntel** from the Start menu (or the desktop icon). It opens in its own window.

The installer contains everything (a private Python, Tesseract OCR with English, Ghostscript), needs no
administrator rights and no internet connection, and installs to `%LOCALAPPDATA%\Programs\DocIntel`.
The app runs on this computer only (127.0.0.1) and stops about a minute after its last window is closed.
Logs and templates are kept in `%LOCALAPPDATA%\DocIntel`, saved files in `Documents\DocIntel`;
uninstalling (Settings → Apps) leaves both in place. PaddleOCR is not included in the installer.

To build the installer yourself, see [packaging/README.md](packaging/README.md).
The steps below are for running DocIntel from source (developers).

---

## 1. Install Python

Python **3.10, 3.11 or 3.12** (64-bit). The pinned dependencies were verified on **Python 3.10.11**.
Python 3.11 is a good choice for new installs.

1. Download from <https://www.python.org/downloads/windows/>, or run `winget install -e --id Python.Python.3.11`.
2. During setup, tick **"Add python.exe to PATH"**.
3. Check it in a new PowerShell window: `py -0p` (lists installed versions) or `python --version`.

## 2. Create a virtual environment

Open PowerShell in the project folder:

```powershell
python -m venv .venv
```

> **OneDrive / long paths:** if the project sits in a OneDrive folder, create the environment
> outside it, for example `python -m venv "$env:USERPROFILE\.venvs\docintel"`. This avoids syncing
> thousands of package files. Keep the path short: pip fails on Windows when paths exceed 260
> characters, unless long paths are enabled. Then use that path wherever `.venv` appears below.

## 3. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell says *"running scripts is disabled on this system"*, allow locally created scripts
for your user account only. This is a safe, commonly used setting that does not need admin rights:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Alternatively, skip activation and call `.\.venv\Scripts\python.exe` directly, or run the
provided scripts with `powershell -ExecutionPolicy Bypass -File scripts\<script>.ps1`. That form
only affects that one process.

## 4. Install the Python dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt            # core (required)
python -m pip install -r requirements-optional.txt   # API, Camelot, OCRmyPDF (optional)
python -m pip install -r requirements-dev.txt        # pytest (to run tests)
```

Or run everything with the installer script:
`powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1 -WithOptional -WithDev`
(add `-VenvPath "$env:USERPROFILE\.venvs\docintel"` to put the environment outside OneDrive).

## 5. Install Tesseract OCR (separate program)

The `pip` packages do **not** include Tesseract itself. Install the UB Mannheim Windows build
(Apache-2.0 licence):

```powershell
winget install -e --id UB-Mannheim.TesseractOCR
```

Or download the installer from <https://github.com/UB-Mannheim/tesseract/wiki>. The default
install location is `C:\Program Files\Tesseract-OCR`.

## 6. Configure the Tesseract path (only if needed)

DocIntel looks for `tesseract.exe` on `PATH`, then in `C:\Program Files\Tesseract-OCR` and
`%LOCALAPPDATA%\Programs\Tesseract-OCR`. If you installed it elsewhere:

```powershell
Copy-Item .env.example .env
notepad .env        # set DOCINTEL_TESSERACT_CMD=D:\Tools\Tesseract-OCR\tesseract.exe
```

## 7. Install OCR language data

The installer ships `eng` (English) and `osd` (orientation detection). For other languages:

* Re-run the Tesseract installer and tick the languages under *Additional language data*, **or**
* download `<lang>.traineddata` (for example `hin.traineddata`) from
  <https://github.com/tesseract-ocr/tessdata_fast> (smaller, faster) or `tessdata_best`
  (more accurate, slower), and copy it into `C:\Program Files\Tesseract-OCR\tessdata`.
  That folder needs admin rights. Alternatively, put the files in your own folder and set
  `DOCINTEL_TESSDATA_DIR` in `.env`. That folder must also contain `eng.traineddata` and
  `osd.traineddata`.

The app only lists languages that are actually installed. Check with
`& "C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs`, or use the **Settings > Re-check engines** button.

## 8. Verify the environment

```powershell
python scripts/check_environment.py
```

This runs the real `tesseract.exe` (rather than just checking that a Python package exists),
lists the installed languages, and reports optional components. It exits with code 1 if anything
required is missing.

## 9. Launch the app

```powershell
python app.py                       # starts the server and opens your browser
# or
python -m streamlit run app.py
# or
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1
```

`python app.py` works from any folder, and also from an IDE's **Run** button (PyCharm, VS Code).
Extra arguments are passed to Streamlit, for example `python app.py --server.port 8502`.

The app opens at <http://localhost:8501>. In every launch mode it listens on `127.0.0.1` only and
Streamlit usage statistics are off. For `python app.py`, these two settings are passed explicitly,
so they don't depend on the folder you start it from.

**Which Python?** `python` must be a Python that has the libraries installed:

* **With the virtual environment (recommended):** activate it first (`.\.venv\Scripts\Activate.ps1`)
  or call its interpreter directly: `& "$env:USERPROFILE\.venvs\docintel\Scripts\python.exe" app.py`.
  In PyCharm, set the project interpreter to that `python.exe` once; then **Run** on `app.py` just works.
* **Without a virtual environment:** you can install into your normal Python instead. If that
  Python already has other projects' packages, do **not** install the pinned requirements as they
  are: they would downgrade shared packages such as NumPy, Pillow, pandas, pydantic and FastAPI.
  Freeze the current versions as constraints and install only what is missing:

  ```powershell
  python -m pip freeze --exclude pip --exclude setuptools --exclude wheel > $env:TEMP\constraints.txt
  python -m pip install -c $env:TEMP\constraints.txt streamlit PyMuPDF==1.24.14 ocrmypdf==16.7.0 pikepdf==9.11.0
  ```

  pip then either finds versions that fit your existing packages or stops, rather than changing
  them. This is how DocIntel was installed into the development machine's main Python, and the
  full test suite passed there: Streamlit 1.65, numpy 2.2, Pillow 12.3, pydantic 2.14, FastAPI 0.124.
  Camelot was left out because it adds a second OpenCV build next to an existing `opencv-python`;
  table extraction then uses PyMuPDF only.

If the libraries are missing, `python app.py` prints the exact install command for the Python
you used instead of a traceback.

**Theme:** use the **Light / Dark** selector in the sidebar. It recolours the whole app,
including Streamlit's own inputs, tables, badges and charts. To start in dark mode, set
`DOCINTEL_UI_THEME=dark` in `.env`. The theme is a server-wide setting, so if several browser
tabs are open they all follow the most recent choice.

**Using the app.** The sidebar groups the pages:

| Section | Page | What it's for |
| --- | --- | --- |
| | **Dashboard** | Session figures, an upload panel with one-click processing, quick actions, system health, and recent activity with View and TXT buttons |
| Process | **Image OCR** | Upload and check files, set processing options (engine, languages, preprocessing, page range, normalization, tables, template), and process several files in a row |
| | **PDF tools** | Page analysis (embedded text vs. scanned), page-range processing, and searchable PDF (skip text / redo OCR / force OCR, deskew, auto-rotate) |
| | **Table extraction** | Source preview, detection, CSV/XLSX export, and clear found / uncertain / none / failed states |
| | **Structured extraction** | Templates, the extracted value next to your correction, validation statuses, and JSON/CSV export |
| Review | **Document intelligence** | Document type with its evidence, every e-mail, phone number, date, amount, ID and number found (filter by type or "needs review"), "Label: value" fields, highlighted text, personal-ID masking (on by default), CSV/JSON export, and a button that opens Structured extraction with the matching template |
| | **OCR results** | The document viewer (page navigation, zoom, rotate view, original vs. processed image) beside tabs for text, **Review** (suspicious words with suggestions), **Confidence** (highlighted words, per-line scores), **Quality** (analysis, chosen plan, attempts, coverage, verdict), tables, structured data and JSON. There's also an accuracy check against reference text. Edits are kept separately from the OCR output. |
| | **Image enhancement** | Image analysis with the smart plan and its reasons, a before/after preview (preset or individual controls), a Tesseract vs. PaddleOCR comparison, and preset-variant comparison with CER/WER against a reference |
| System | **Settings** | Engines, defaults, limits, resources and temporary-file cleanup |
| | **About** | Versions, licences, privacy and health details |

Every page that works on documents has its own upload panel and a **Try a sample** button.
Files added on any page are available on all pages for the rest of the session.

## 10. Run the automated tests

```powershell
python -m pytest tests/unit -q          # no Tesseract needed (fake OCR engine)
python -m pytest tests/integration -q   # real Tesseract; skipped automatically if missing
python -m pytest -q                     # everything
```

Results in the development environment (Windows 11, Python 3.10.11, Tesseract 5.4.0, Ghostscript 10.08.0):

* Project virtual environment: **369 passed, 7 skipped**. The skipped tests are the PaddleOCR tests, which only run where PaddleOCR is installed.
* PaddleOCR environment: **376 passed, 0 skipped**.
* Main (non-virtual-environment) Python with the constrained install above: **368 passed, 8 skipped** (the PaddleOCR tests and the Camelot test).
* Without Ghostscript, the searchable-PDF tests also skip automatically.

The UI tests use Streamlit's AppTest with a fake OCR engine. They cover every page, the dashboard
buttons, upload validation, processing actions, the workspace controls and edits, the light and
dark themes (including WCAG contrast of both palettes), empty states and missing-dependency
states, and they check that viewing results never re-runs OCR.

## 11. Optional components

| Component | Install | Notes |
| --- | --- | --- |
| REST API | `pip install -r requirements-optional.txt` | see section 12 |
| Camelot tables | same file | digital PDFs only; PyMuPDF tables work without it |
| Searchable PDF | same file **plus** Ghostscript (AGPL), see below | found automatically in `C:\Program Files\gs\gs<ver>\bin`; no PATH change needed |
| PaddleOCR engine | separate environment, see below | about 600 MB RAM while loaded; one language per run |

**Installing Ghostscript.** There is no official winget package. Download the 64-bit installer
`gs<version>w64.exe` from the Artifex releases page
<https://github.com/ArtifexSoftware/ghostpdl-downloads/releases> (or <https://ghostscript.com/releases/gsdnld.html>).
Check that its digital signature reads *Artifex Software, Inc.* (right-click > Properties >
Digital Signatures), then run it. A silent install from an elevated prompt also works:
`.\gs10080w64.exe /S`. Version 10.08.0 is the one tested.

Searchable PDF modes:

* **Skip text** (default): OCRs only pages without text, so it is safe for mixed PDFs.
* **Redo OCR**: replaces an existing (possibly wrong) OCR layer and keeps real digital text.
  DocIntel removes the old invisible text layer itself (PDF text render mode 3), then OCRs the
  pages that are left without text. It does not use OCRmyPDF's built-in `--redo-ocr`, which
  duplicated every word in testing (OCRmyPDF 16.7 and 16.13). Tested: no duplicates even after
  redoing twice, a wrong layer is replaced, and digital pages are untouched.
* **Force OCR**: rasterizes every page and replaces any text layer, including digital text.

*Deskew* needs scans of roughly 200 DPI or more.

**Installing PaddleOCR (optional deep-learning engine).** PaddleOCR replaces the headless OpenCV
with `opencv-python`/`opencv-contrib-python` 4.11, so install it in its own environment containing
the full app. The tested combination is PaddlePaddle 3.0.0 + PaddleOCR 2.10.0. On Windows,
PaddlePaddle 2.6 requires protobuf ≤ 3.20.2, which Streamlit 1.65 can't use.

```powershell
python -m venv "$env:USERPROFILE\.venvs\docintel-paddle"
$py = "$env:USERPROFILE\.venvs\docintel-paddle\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt -r requirements-optional.txt -r requirements-dev.txt
& $py -m pip install -r requirements-paddle.txt
$env:DOCINTEL_ENABLE_PADDLE = "true"      # or set DOCINTEL_ENABLE_PADDLE=true in .env
& $py -m streamlit run app.py             # PaddleOCR now appears in the "OCR engine" list
& $py -m pytest tests/integration/test_paddle_integration.py -v
```

The first recognition downloads the English models (about 16 MB) into `%USERPROFILE%\.paddleocr`,
so do it once while online. After that it works offline. Measured on the development machine:

* First call: 20.8 s, including the download and model load.
* After that: about 0.4 s per image, about 600 MB process memory.
* Accuracy: CER 0.017 on the clean sample.

PaddleOCR tends to drop sentence-final full stops and sometimes adds a spurious one at a line
end. Extracted *string* fields keep the verbatim value, and their `normalized_value` drops
trailing `. , ; :`.

EasyOCR is deliberately not included. It needs PyTorch (more than 1 GB) and adds no capability that
Tesseract plus the optional PaddleOCR engine don't already cover on CPU.

A local LLM is **not** used anywhere. Structured extraction is fully deterministic.

## 12. Start the REST API (optional)

```powershell
python -m uvicorn docintel.api.main:app --host 127.0.0.1 --port 8000
# or: powershell -ExecutionPolicy Bypass -File scripts\run_api.ps1
```

Interactive docs: <http://127.0.0.1:8000/docs>. Endpoints:

* `GET /health`: version and OCR engine availability
* `GET /templates`: stored extraction template names
* `POST /process`: multipart `file` plus form fields `engine`, `languages` (for example `eng+hin`),
  `psm`, `preprocess`, `pdf_mode`, `pages`, `extract_tables`, `table_method`, `template`.
  The response includes `analysis` (document type, entities, key-value pairs).
* `POST /extract`: JSON `{"text": "...", "template": "incident_report"}`
* `POST /analyze`: JSON `{"text": "..."}` returns the document type, entities and key-value pairs
  (no template needed). Identifiers are returned unmasked, as the API is for local use only.

```powershell
curl.exe -F "file=@data/samples/incident_report.png" -F "template=incident_report" http://127.0.0.1:8000/process
```

**API security.** There is no authentication. Uploads are size-limited (`DOCINTEL_API_MAX_REQUEST_MB`),
type-checked by content, and never stored. Keep the API bound to `127.0.0.1`. Before exposing it to
a network, put it behind a reverse proxy with TLS and add authentication, access control, rate
limiting and request logging. Run it under a low-privilege account.

## 13. Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Most common issues:

* *Tesseract not available*: install it (section 5) or set `DOCINTEL_TESSERACT_CMD`.
* *Language data not installed*: copy the `.traineddata` file (section 7), then use **Settings > Re-check engines**.
* *pip "No such file or directory" with long paths*: use a shorter venv path (section 2).

---

## Configuration reference

All settings are environment variables with the `DOCINTEL_` prefix, set in `.env` (see `.env.example`).
Session-only changes can be made on the **Settings** page.

## Performance and resources

These figures were measured on the development machine (8 logical CPUs, 16 GB RAM) with
Tesseract 5.4 `eng` (fast LSTM model). The input was a 5-page scanned PDF with about 40 lines
of text per page:

| Render DPI | Preprocessing | Time per page | Python process RSS |
| --- | --- | --- | --- |
| 200 | original | 1.6 s | about 100 MB |
| 200 | auto | 2.5 s | about 117 MB |
| 300 | original | 2.0 s | about 127 MB |
| 300 | auto | 3.0 s | about 136 MB |

Tesseract runs as a separate process and typically uses another 100–250 MB while it works.
Expect somewhat slower times on an 8 GB i5. Design choices that keep resource use bounded:

* PDFs are processed and rendered one page at a time.
* Engines are created lazily, and only one deep-learning model is loaded at a time.
* Files are processed sequentially by default (`DOCINTEL_MAX_WORKERS=1`). Concurrency is bounded at 4.
* Pixel limits apply to uploads, PDF rendering and upscaling.
* OCR and documents have time limits.

**Tesseract vs. PaddleOCR:** Tesseract needs about 100–250 MB per process, starts instantly, and is
strong on clean printed documents. PaddleOCR (deep learning) needs about 0.5–1 GB, takes several
seconds to load, is typically slower per page on CPU, and can do better on photos, scene text and
some scripts. It supports one language per run.

## Licences

DocIntel's own code has no third-party assets bundled. The main dependencies and their licences:

* Tesseract and tessdata: Apache-2.0
* Streamlit, PyMuPDF wheels: Apache-2.0 / AGPL-3.0 (PyMuPDF is dual-licensed: AGPL for open-source use, commercial licence from Artifex otherwise)
* OpenCV: Apache-2.0
* pandas, NumPy: BSD
* Pillow: MIT-CMU
* Pydantic, FastAPI: MIT
* Camelot: MIT
* OCRmyPDF: MPL-2.0
* Ghostscript: AGPL-3.0
* PaddleOCR: Apache-2.0

DocIntel itself is licensed under the **GNU AGPL-3.0** (see [LICENSE](LICENSE)), which matches the
AGPL components it uses and bundles (PyMuPDF, Ghostscript). You may use, copy and modify it; if you
distribute it or offer it as a network service, you must make your source code available under the same licence.
The Windows installer lists the bundled programs in `THIRD_PARTY_NOTICES.txt`.
