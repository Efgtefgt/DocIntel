"""Exporters, safe naming and temporary-file cleanup."""

from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime

import openpyxl
import pandas as pd
import pytest

from docintel.core.pipeline import DocumentPipeline, PipelineOptions
from docintel.core.result_models import TableResult
from docintel.core.temp_files import cleanup_stale_temp_dirs, count_temp_dirs, managed_temp_dir
from docintel.evaluation import synthetic_data as synth
from docintel.export.csv_exporter import pages_to_csv_bytes, structured_to_csv_bytes, table_to_csv_bytes
from docintel.export.excel_exporter import tables_to_xlsx_bytes
from docintel.export.file_naming import export_filename, save_export
from docintel.export.json_exporter import document_to_json_bytes, structured_to_json_bytes
from docintel.export.txt_exporter import document_to_txt
from docintel.extraction.structured_extractor import StructuredExtractor
from docintel.extraction.templates import TemplateStore
from tests.conftest import TEMPLATES_DIR, make_factory
from tests.fakes import FakeEngine

TABLE = TableResult(table_index=1, page_number=2, method="pymupdf", columns=["A", "B"],
                    rows=[["=SUM(A1:A9)", "1"], [None, "-5"], ["Ünïcode", "@cmd"]])


@pytest.fixture
def document(settings, text_image):
    data = synth.make_mixed_pdf("Embedded page text with sufficient characters here.", text_image)
    return DocumentPipeline(settings, make_factory(settings, FakeEngine(text="Scanned Ünïcode"))).process(
        data, "mixed.pdf", PipelineOptions(engine="fake", preprocess_preset="original")).document


def test_txt_export_keeps_page_boundaries(document):
    text = document_to_txt(document)
    assert "===== Page 1 | source: embedded_text =====" in text
    assert "===== Page 2 | source: ocr =====" in text
    assert text.index("Page 1") < text.index("Page 2")
    assert "may contain recognition errors" in text
    assert "Scanned Ünïcode" in text.encode("utf-8").decode("utf-8")


def test_json_export_roundtrip(document):
    data = json.loads(document_to_json_bytes(document).decode("utf-8"))
    assert [p["page_number"] for p in data["pages"]] == [1, 2]
    assert "words" not in data["pages"][1]["ocr"]
    full = json.loads(document_to_json_bytes(document, include_words=True))
    assert full["pages"][1]["ocr"]["words"]


def test_structured_json_and_csv():
    result = StructuredExtractor(TemplateStore(TEMPLATES_DIR).load("incident_report")).extract("Caller Name: Asha")
    payload = json.loads(structured_to_json_bytes(result))
    assert payload["fields"]["caller_name"]["value"] == "Asha"
    assert payload["fields"]["phone_number"]["value"] is None and payload["fields"]["phone_number"]["status"] == "missing"
    df = pd.read_csv(io.BytesIO(structured_to_csv_bytes(result)), encoding="utf-8-sig")
    assert set(df["field"]) >= {"caller_name", "phone_number"}


def test_table_csv_is_utf8_and_formula_safe():
    raw = table_to_csv_bytes(TABLE)
    assert raw.startswith(b"\xef\xbb\xbf")
    df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", keep_default_na=False)
    assert df.iloc[0, 0] == "'=SUM(A1:A9)"
    assert str(df.iloc[1, 1]) == "-5"  # negative numbers are not escaped
    assert df.iloc[2, 0] == "Ünïcode" and df.iloc[2, 1] == "'@cmd"


def test_pages_csv(document):
    df = pd.read_csv(io.BytesIO(pages_to_csv_bytes(document)), encoding="utf-8-sig")
    assert list(df["page"]) == [1, 2]


def test_xlsx_export():
    book = openpyxl.load_workbook(io.BytesIO(tables_to_xlsx_bytes([TABLE])))
    assert book.sheetnames == ["p2_t1", "extraction_notes"]
    assert book["p2_t1"]["A2"].value == "'=SUM(A1:A9)"


def test_export_filename_is_safe_and_predictable():
    name = export_filename("../../My Report.pdf", "text", "txt", timestamp=datetime(2024, 1, 2, 3, 4, 5))
    assert name == "My Report_text_20240102-030405.txt"


def test_save_export_never_overwrites(tmp_path):
    first = save_export(b"one", "out.txt", tmp_path)
    second = save_export(b"two", "out.txt", tmp_path)
    assert first.read_bytes() == b"one" and second.read_bytes() == b"two"
    assert second.name == "out (1).txt"
    escaped = save_export(b"x", "../../evil.txt", tmp_path)
    assert escaped.parent == tmp_path.resolve()


def test_temp_dir_removed_even_on_error(settings):
    with pytest.raises(RuntimeError):
        with managed_temp_dir(settings) as tmp:
            (tmp / "file.bin").write_bytes(b"data")
            raise RuntimeError("boom")
    assert not tmp.exists()
    assert count_temp_dirs(settings) == 0


def test_cleanup_stale_temp_dirs(settings):
    root = settings.temp_dir
    root.mkdir(parents=True, exist_ok=True)
    stale, fresh = root / "docintel_old", root / "docintel_new"
    stale.mkdir()
    fresh.mkdir()
    old = time.time() - 7200
    os.utime(stale, (old, old))
    assert cleanup_stale_temp_dirs(settings, older_than_seconds=3600) == 1
    assert not stale.exists() and fresh.exists()


def test_pipeline_leaves_no_temp_files(settings, png_bytes):
    DocumentPipeline(settings, make_factory(settings)).process(png_bytes, "a.png", PipelineOptions(engine="fake", preprocess_preset="original"))
    assert count_temp_dirs(settings) == 0
