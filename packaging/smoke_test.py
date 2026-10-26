"""Check a staged installer folder before it is packed (run with the STAGED python.exe).

Uses only the bundled Tesseract and Ghostscript: the installed copies are removed from PATH,
so a missing file in the bundle fails here instead of on a user's PC.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

stage = Path(sys.argv[1]).resolve()
assert Path(sys.executable).resolve().parent == stage / "python", "run this with the staged python.exe"
tools = stage / "tools"
os.environ["PATH"] = os.pathsep.join(
    [str(tools / "tesseract"), str(tools / "ghostscript" / "bin")]
    + [p for p in os.environ.get("PATH", "").split(os.pathsep)
       if p and "tesseract" not in p.lower() and "\gs\\" not in p.lower()])
os.environ["DOCINTEL_TESSERACT_CMD"] = str(tools / "tesseract" / "tesseract.exe")
os.environ["DOCINTEL_LOG_TO_FILE"] = "false"
os.environ["DOCINTEL_EXPORT_DIR"] = tempfile.mkdtemp()

import camelot  # noqa: E402,F401 - optional features must import
import ocrmypdf  # noqa: E402,F401
import streamlit  # noqa: E402

from docintel.config.settings import get_settings  # noqa: E402
from docintel.core.pipeline import DocumentPipeline, PipelineOptions  # noqa: E402
from docintel.documents.searchable_pdf import check_searchable_pdf_support, create_searchable_pdf  # noqa: E402

settings = get_settings()
print("streamlit", streamlit.__version__, "| python", sys.version.split()[0])

started = time.perf_counter()
image = stage / "app" / "data" / "samples" / "incident_report.png"
result = DocumentPipeline(settings).process(image.read_bytes(), image.name, PipelineOptions())
text = result.document.combined_text() if result.document else ""
assert result.succeeded and "incident" in text.lower(), f"OCR failed: {result.errors}"
print(f"OCR ok ({len(text)} characters, {time.perf_counter() - started:.1f}s, "
      f"type: {result.analysis.classification.document_type if result.analysis else '-'})")

status = check_searchable_pdf_support(settings)
assert status.available, status.reason
assert Path(status.ghostscript).resolve().is_relative_to(tools), f"not the bundled Ghostscript: {status.ghostscript}"
import fitz  # noqa: E402

doc = fitz.open()
doc.new_page().insert_text((72, 100), "Searchable PDF smoke test 12345", fontsize=18)
pdf = doc.convert_to_pdf()
out = create_searchable_pdf(pdf, settings, timeout_seconds=300)
assert out.startswith(b"%PDF"), "searchable PDF output is not a PDF"
print("Searchable PDF ok (bundled Ghostscript)")
print("SMOKE TEST PASSED")
