# Building the Windows installer

`build_installer.ps1` produces `dist\DocIntel-Setup-<version>.exe`: a per-user installer
(no administrator rights) that bundles a private Python 3.10, DocIntel's Python libraries,
Tesseract OCR and Ghostscript, so the target PC needs nothing else.

## Requirements on the build PC

* Python 3.10 64-bit (default: `%LOCALAPPDATA%\Programs\Python\Python310\python.exe`, or pass `-HostPython`),
  with PyMuPDF and Pillow installed (used to make the icon)
* Tesseract OCR and Ghostscript installed (their program folders are copied; pass
  `-TesseractDir` / `-GhostscriptDir` if they are not in the default location).
  Only the languages installed in Tesseract's `tessdata` folder are bundled.
* Inno Setup 6: `winget install JRSoftware.InnoSetup --scope user`
* Internet access (downloads the Python embeddable package, checksum-verified, and the wheels)
* Optional: the tested development environment (`%USERPROFILE%\.venvs\docintel`, or `-TestedPython`);
  its `pip freeze` pins every indirect dependency to versions the test suite passed with

## Build

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_installer.ps1
```

About 3–5 minutes. Steps: stage the files in `%LOCALAPPDATA%\DocIntelBuild` (outside OneDrive),
pre-compile Python files, run `smoke_test.py` with the staged Python using only the bundled
Tesseract and Ghostscript (OCR + searchable PDF), then compile `DocIntel.iss` with Inno Setup.
The installer is not built if the smoke test fails. `-SkipCompile` stops after the smoke test.

## Files

| File | Purpose |
| --- | --- |
| `build_installer.ps1` | Build script |
| `DocIntel.iss` | Inno Setup script (per-user install, shortcuts, uninstaller). Never change its `AppId`. |
| `DocIntel.pyw` | Launcher used by the shortcuts: starts the local server, opens an Edge/Chrome app window, stops the server about a minute after the last window closes, keeps user data outside the program folder |
| `smoke_test.py` | Pre-packaging check of the staged app |
| `make_icon.py` | Renders `docintel/ui/assets/icon.svg` to `DocIntel.ico` |
| `THIRD_PARTY_NOTICES.txt` | Licences of the bundled programs (copied into the installation) |

## Publishing a release

The installer (~190 MB) is larger than GitHub's 100 MB file limit, so it is not committed (`dist/` is
ignored). Attach it to a GitHub Release instead:

```powershell
gh release create v1.0.0 dist\DocIntel-Setup-1.0.0.exe --title "DocIntel 1.0.0" --notes "Windows installer"
```

The installer is not code-signed, so Windows SmartScreen shows "Unknown publisher" on first run.
