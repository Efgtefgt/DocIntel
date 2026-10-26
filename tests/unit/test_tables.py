"""Table extraction: digital PDFs and OCR word boxes."""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from docintel.core.result_models import BoundingBox, OCRResult, OCRWord
from docintel.evaluation import synthetic_data as synth
from docintel.extraction.table_extractor import (
    TableMethod,
    camelot_available,
    detect_grid_lines,
    extract_pdf_tables,
    extract_tables_from_ocr,
    remove_ruling_lines,
    table_with_header,
)

ROWS = [["Name", "Qty", "Price"], ["Apple", "3", "1.20"], ["Banana", "", "0.50"]]


def _word(text: str, x: int, y: int, w: int = 60, h: int = 20, conf: float = 95) -> OCRWord:
    return OCRWord(text=text, confidence=conf, bbox=BoundingBox(x=x, y=y, width=w, height=h))


def _ocr(words: list[OCRWord]) -> OCRResult:
    return OCRResult(engine="fake", languages=["eng"], text=" ".join(w.text for w in words), words=words)


def test_pymupdf_digital_table():
    tables = extract_pdf_tables(synth.make_table_pdf(ROWS), [1], TableMethod.PYMUPDF, settings=None)  # type: ignore[arg-type]
    assert len(tables) == 1
    table = tables[0]
    assert table.rows[0] == ["Name", "Qty", "Price"]
    assert table.rows[1] == ["Apple", "3", "1.20"]
    assert table.rows[2][1] is None  # missing cell is kept as empty, not dropped
    assert "empty or merged" in " ".join(table.notes)


@pytest.mark.skipif(not camelot_available(), reason="Camelot not installed")
def test_camelot_lattice_digital_table(settings):
    tables = extract_pdf_tables(synth.make_table_pdf(ROWS), [1], TableMethod.CAMELOT_LATTICE, settings)
    assert tables and tables[0].rows[1] == ["Apple", "3", "1.20"]
    assert tables[0].quality_score is not None


def test_table_with_header():
    tables = extract_pdf_tables(synth.make_table_pdf(ROWS), [1], TableMethod.PYMUPDF, settings=None)  # type: ignore[arg-type]
    headed = table_with_header(tables[0])
    assert headed.columns == ["Name", "Qty", "Price"] and len(headed.rows) == 2


def _grid_image() -> Image.Image:
    img = Image.new("RGB", (640, 260), "white")
    draw = ImageDraw.Draw(img)
    for y in (20, 100, 180, 240):
        draw.line([(20, y), (620, y)], fill="black", width=3)
    for x in (20, 220, 420, 620):
        draw.line([(x, 20), (x, 240)], fill="black", width=3)
    return img


def test_grid_line_detection():
    rows, cols = detect_grid_lines(_grid_image())
    assert len(rows) == 4 and len(cols) == 4


def test_remove_ruling_lines_keeps_geometry():
    img = _grid_image()
    ImageDraw.Draw(img).text((60, 50), "Name", fill="black")
    cleaned, removed = remove_ruling_lines(img)
    assert removed and cleaned.size == img.size
    gray = cleaned.convert("L")
    assert gray.getpixel((300, 100)) == 255  # a former line pixel is now white
    assert min(gray.crop((55, 45, 120, 70)).getdata()) < 128  # text survives
    assert detect_grid_lines(img) is not None  # original still has its grid
    blank = Image.new("RGB", (200, 100), "white")
    assert remove_ruling_lines(blank) == (blank, False)


def test_line_artifact_tokens_ignored_in_grid():
    words = [_word("|", 215, 40, w=8, h=150), _word("Name", 40, 50), _word("Qty", 240, 50), _word("Price", 440, 50)]
    table = extract_tables_from_ocr(_ocr(words), _grid_image(), page_number=1)[0]
    assert table.rows[0][:2] == ["Name", "Qty"]
    assert any("line characters" in n for n in table.notes)


def test_ocr_grid_table_keeps_missing_cells():
    words = [_word("Name", 40, 50), _word("Qty", 240, 50), _word("Price", 440, 50),
             _word("Apple", 40, 130), _word("3", 240, 130, w=15), _word("1.20", 440, 130),
             _word("Banana", 40, 200), _word("0.50", 440, 200)]
    tables = extract_tables_from_ocr(_ocr(words), _grid_image(), page_number=1)
    assert len(tables) == 1 and tables[0].method == "ocr_grid"
    assert tables[0].rows == [["Name", "Qty", "Price"], ["Apple", "3", "1.20"], ["Banana", None, "0.50"]]


def test_ocr_grid_flags_straddling_words():
    words = [_word("Name", 40, 50), _word("WideWordAcrossCells", 150, 130, w=200), _word("x", 440, 200),
             _word("y", 440, 50)]
    table = extract_tables_from_ocr(_ocr(words), _grid_image(), page_number=1)[0]
    assert table.uncertain and any("cross a cell boundary" in n for n in table.notes)


def test_whitespace_table_is_always_uncertain():
    words = []
    for r, row in enumerate(ROWS):
        for c, cell in enumerate(row):
            if cell:
                words.append(_word(cell, 50 + c * 250, 40 + r * 40))
    tables = extract_tables_from_ocr(_ocr(words), None, page_number=2)
    assert len(tables) == 1
    table = tables[0]
    assert table.uncertain and table.method == "ocr_whitespace" and table.page_number == 2
    assert table.rows[0] == ["Name", "Qty", "Price"]
    assert table.rows[2] == ["Banana", None, "0.50"]


def test_plain_paragraph_is_not_a_table():
    words = [_word(w, 50 + i * 70, 40 + r * 40) for r in range(4) for i, w in enumerate("this is ordinary text".split())]
    assert extract_tables_from_ocr(_ocr(words), None, page_number=1) == []
