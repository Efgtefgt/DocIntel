# Troubleshooting

Start with `python scripts/check_environment.py`. It reports most problems precisely.

## Installation

| Symptom | Cause / fix |
| --- | --- |
| `running scripts is disabled on this system` | PowerShell execution policy. Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (affects only your account), or use `powershell -ExecutionPolicy Bypass -File scripts\...ps1`. Avoid `Unrestricted` and machine-wide changes. |
| `python` opens the Microsoft Store | Windows app-execution alias. Install Python from python.org, or turn off the alias under *Settings > Apps > Advanced app settings > App execution aliases*. Or use `py -3.11`. |
| pip `OSError: [Errno 2] No such file or directory ... HINT: ... Long Path support` | The venv path is too long, which is common in OneDrive or deep folders. Create the venv in a short path (`$env:USERPROFILE\.venvs\docintel`), or enable long paths (needs admin): `New-ItemProperty -Path HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`. |
| `ensurepip ... returned non-zero exit status` when creating the venv | Same long-path cause as above. |
| pip `Microsoft Visual C++ 14.0 is required` | A package fell back to building from source because no wheel matched your Python version. Use Python 3.10–3.12 64-bit. |
| `streamlit ... requires pillow<12` conflict | Something upgraded Pillow. Run `python -m pip install pillow==11.3.0`. |
| Corporate proxy blocks pip | `python -m pip install --proxy http://proxy:port -r requirements.txt`, or ask IT for an internal package index. |

## Tesseract

| Symptom | Cause / fix |
| --- | --- |
| *Tesseract executable not available* | Not installed (README step 5), or installed in a custom folder: set `DOCINTEL_TESSERACT_CMD` in `.env` to the full path of `tesseract.exe`. |
| *Language data not installed for Tesseract: hin* | Copy `hin.traineddata` into the tessdata folder (README step 7), then press **Settings > Re-check engines** or restart the app. |
| *Tesseract runs, but no language data was found* | `DOCINTEL_TESSDATA_DIR` points to a folder without `.traineddata` files. Clear it or fix it. |
| Orientation step says "orientation could not be determined" | `osd.traineddata` is missing, or the page has too little text for orientation detection. |
| Very slow OCR | Lower the PDF DPI (Settings), select page ranges, use `tessdata_fast` models, and keep `max_workers=1` on 8 GB machines. |
| *exceeded the 60s time limit* | Raise `DOCINTEL_OCR_TIMEOUT_SECONDS`, or reduce the image size or DPI. |

## OCR quality

* Try **Image enhancement** to compare `original`, `auto` and threshold variants. Paste reference
  text there to get CER/WER.
* Text in a single column or block: PSM 4 or 6. A single line: PSM 7. Scattered labels: PSM 11.
* Photos with shadows: try `adaptive_threshold`. Clean scans: `original` or `grayscale` often work best.
* Small text: make sure `upscale` is in the steps, or render PDFs at 300 DPI.

## PDFs and tables

| Symptom | Cause / fix |
| --- | --- |
| *password-protected* | Remove the password with your PDF tool first. Encrypted PDFs are rejected. |
| Page text looks garbled even though it is a digital PDF | Broken font encoding. Use PDF handling **"OCR every page"**. |
| Searchable PDF *Ghostscript not found* | Install Ghostscript from the Artifex releases page (README section 11; there is no winget package). If you installed it outside `C:\Program Files\gs`, add its `bin` folder to PATH. |
| Searchable PDF fails with *unexpected error* / `'Pdf' object has no attribute 'check'` | pikepdf 10 was installed, which is incompatible with OCRmyPDF 16.7. Run `python -m pip install pikepdf==9.11.0`. |
| Searchable PDF *already contains text* | Use "Redo OCR" to replace an old OCR layer (digital text is kept), or "Force OCR" to rasterize and OCR everything. |
| PaddleOCR shows *Disabled* | Set `DOCINTEL_ENABLE_PADDLE=true` in `.env` and start the app from the PaddleOCR environment (README section 11). |
| PaddleOCR *not installed* although `pip install` worked | The app is running from a different environment. Start it with that environment's `python.exe -m streamlit run app.py`. |
| PaddleOCR first run hangs or fails offline | The first call downloads models to `%USERPROFILE%\.paddleocr`. Run it once with internet access, or copy that folder from another machine. |
| `cv2` errors after installing PaddleOCR in the main environment | PaddleOCR replaced the headless OpenCV. Reinstall the main environment (`pip install --force-reinstall opencv-python-headless==4.10.0.84`), or use a separate PaddleOCR environment. |
| Searchable PDF deskew has no effect | OCRmyPDF measures a 0° angle on low-resolution pages. Scan at 200–300 DPI. |
| No tables found on a scan | Only ruled grids or clearly column-aligned text can be rebuilt. The OCR text is still shown. |
| Camelot finds nothing | Camelot only reads digital PDFs. Try `stream` for tables without lines, or the PyMuPDF method. |

## App and API

* **"Welcome to Streamlit!… Email:" prompt or "streamlit skills" message:** these are Streamlit's
  first-run messages in interactive mode. `python app.py` runs headless, so they don't appear. If you
  start with `python -m streamlit run app.py`, leave the e-mail blank (usage statistics are already
  off in `.streamlit/config.toml`), or add `--server.headless true`.
* **`ImportError` / versions conflict after installing into a shared Python:** use the constrained
  install from README section 9 so existing packages are never changed.

* Port already in use: `scripts\run_app.ps1 -Port 8502` / `scripts\run_api.ps1 -Port 8001`.
* API returns 413: the file exceeds `DOCINTEL_API_MAX_REQUEST_MB`.
* Logs (metadata only, no document text): `logs\docintel.log`.
