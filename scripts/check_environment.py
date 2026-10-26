"""Verify the local environment: Python, packages, Tesseract executable, language data, options.

Usage:  python scripts/check_environment.py
Exit code 0 = core requirements satisfied, 1 = something required is missing.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"
REQUIRED = {
    "streamlit": "streamlit", "numpy": "numpy", "pandas": "pandas", "PIL": "pillow", "cv2": "opencv-python-headless",
    "fitz": "PyMuPDF", "openpyxl": "openpyxl", "pydantic": "pydantic", "pydantic_settings": "pydantic-settings",
}
OPTIONAL = {"fastapi": "fastapi", "uvicorn": "uvicorn", "camelot": "camelot-py", "ocrmypdf": "ocrmypdf",
            "paddleocr": "paddleocr", "psutil": "psutil", "pytest": "pytest"}


def _version(dist: str) -> str:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "?"


def main() -> int:
    failures = 0
    print("DocIntel environment check\n")
    py_ok = (3, 10) <= sys.version_info[:2] <= (3, 12)
    print(f"{OK if py_ok else FAIL} Python {platform.python_version()} ({platform.architecture()[0]}) on {platform.platform()}")
    failures += not py_ok

    print("\nRequired Python packages")
    for module, dist in REQUIRED.items():
        try:
            importlib.import_module(module)
            print(f"{OK} {dist} {_version(dist)}")
        except ImportError as exc:
            print(f"{FAIL} {dist} not importable ({exc}); run: python -m pip install -r requirements.txt")
            failures += 1
    if failures:
        print("\nFix the required packages first.")
        return 1

    from docintel.config.settings import get_settings
    from docintel.documents.searchable_pdf import check_searchable_pdf_support, find_ghostscript
    from docintel.ocr.engine_factory import EngineFactory
    from docintel.ocr.tesseract_engine import resolve_tesseract_cmd

    settings = get_settings()
    print("\nTesseract OCR (native program)")
    cmd = resolve_tesseract_cmd(settings.tesseract_cmd)
    statuses = {s.name: s for s in EngineFactory(settings).statuses()}
    tess = statuses["tesseract"]
    if cmd is None:
        print(f"{FAIL} Tesseract executable not found. Install it (README step 5) or set DOCINTEL_TESSERACT_CMD in .env.")
        failures += 1
    elif not tess.available:
        print(f"{FAIL} {cmd}: {tess.reason}")
        failures += 1
    else:
        print(f"{OK} {cmd}")
        print(f"{OK} {tess.version}")
        print(f"{OK} Language data installed: {', '.join(tess.languages)}")
        missing = [lang for lang in settings.language_list if lang not in tess.languages]
        if missing:
            print(f"{FAIL} Configured default language(s) missing: {', '.join(missing)} (DOCINTEL_DEFAULT_LANGUAGES)")
            failures += 1
        from docintel.ocr.tesseract_engine import TesseractEngine

        if "osd" not in TesseractEngine(settings).available_languages():
            print(f"{WARN} osd.traineddata not installed: automatic 90/180 degree orientation correction is disabled.")

    print("\nOptional components")
    for module, dist in OPTIONAL.items():
        found = importlib.util.find_spec(module) is not None
        print(f"{OK if found else WARN} {dist} {_version(dist) if found else 'not installed'}")
    gs = find_ghostscript()
    print(f"{OK if gs else WARN} Ghostscript: {gs or 'not found (needed only for searchable PDF export)'}")
    pdf = check_searchable_pdf_support(settings)
    print(f"{OK if pdf.available else WARN} Searchable PDF export: {'available' if pdf.available else pdf.reason}")
    paddle = statuses["paddle"]
    print(f"{OK if paddle.available else WARN} PaddleOCR engine: {'available' if paddle.available else paddle.reason}")

    try:
        import psutil

        vm = psutil.virtual_memory()
        print(f"\nMemory: {vm.total / 2**30:.1f} GB total, {vm.available / 2**30:.1f} GB available; CPUs: {psutil.cpu_count()}")
        if vm.total < 7.5 * 2**30:
            print(f"{WARN} Less than 8 GB RAM: keep max_workers=1 and avoid PaddleOCR.")
    except ImportError:
        pass

    print("\nResult:", "all required components available." if not failures else f"{failures} required item(s) missing.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
