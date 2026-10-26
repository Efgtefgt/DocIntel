"""Table extraction.

* Digital PDFs: PyMuPDF ``find_tables`` (always available) or Camelot (optional,
  lattice/stream flavours). These read the PDF's vector text and lines; they do not
  work on scanned pages.
* Scanned pages/images: tables are rebuilt from OCR word boxes. If ruling lines are
  visible, they define the cell grid; otherwise columns are inferred from whitespace,
  and the result is always flagged as uncertain.

Cells with no text are kept as ``None`` (never dropped or filled in). Words that cross a
column boundary are reported instead of being silently merged into a neighbour cell.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import statistics
from enum import Enum
from typing import Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from docintel.config.settings import Settings
from docintel.core.exceptions import OptionalDependencyMissingError, TableExtractionError
from docintel.core.result_models import OCRResult, OCRWord, TableResult
from docintel.core.temp_files import managed_temp_dir
from docintel.documents.pdf_reader import open_pdf

logger = logging.getLogger(__name__)


class TableMethod(str, Enum):
    AUTO = "auto"
    PYMUPDF = "pymupdf"
    CAMELOT_LATTICE = "camelot_lattice"
    CAMELOT_STREAM = "camelot_stream"


def camelot_available() -> bool:
    return importlib.util.find_spec("camelot") is not None


def available_pdf_methods() -> list[TableMethod]:
    methods = [TableMethod.AUTO, TableMethod.PYMUPDF]
    if camelot_available():
        methods += [TableMethod.CAMELOT_LATTICE, TableMethod.CAMELOT_STREAM]
    return methods


def _column_names(count: int) -> list[str]:
    return [f"Column {i + 1}" for i in range(count)]


def _clean_cell(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _rectangular(rows: Sequence[Sequence[object]]) -> tuple[list[list[Optional[str]]], bool]:
    """Pad rows to equal width with ``None``. Returns (rows, whether padding was needed)."""
    width = max((len(r) for r in rows), default=0)
    padded = any(len(r) != width for r in rows)
    return [[_clean_cell(c) for c in r] + [None] * (width - len(r)) for r in rows], padded


# --------------------------------------------------------------------------------------
# Digital PDFs
# --------------------------------------------------------------------------------------
def extract_pdf_tables(
    data: bytes, page_numbers: Sequence[int], method: TableMethod, settings: Settings
) -> list[TableResult]:
    """Extract tables from digital (text-based) PDF pages."""
    if method in (TableMethod.CAMELOT_LATTICE, TableMethod.CAMELOT_STREAM):
        return _extract_camelot(data, page_numbers, method, settings)
    if method is TableMethod.AUTO and camelot_available():
        tables = _extract_camelot(data, page_numbers, TableMethod.CAMELOT_LATTICE, settings)
        if tables:
            return tables
    return _extract_pymupdf(data, page_numbers)


def _extract_pymupdf(data: bytes, page_numbers: Sequence[int]) -> list[TableResult]:
    results: list[TableResult] = []
    with open_pdf(data) as doc:
        for number in page_numbers:
            page = doc[number - 1]
            try:
                finder = page.find_tables()
            except (RuntimeError, ValueError) as exc:
                raise TableExtractionError(f"Table detection failed on page {number}.") from exc
            for tab in finder.tables:
                rows, padded = _rectangular(tab.extract())
                if not rows:
                    continue
                table = TableResult(
                    table_index=len(results) + 1,
                    page_number=number,
                    method=TableMethod.PYMUPDF.value,
                    columns=_column_names(len(rows[0])),
                    rows=rows,
                )
                _flag_sparse(table, padded)
                results.append(table)
    return results


def _extract_camelot(
    data: bytes, page_numbers: Sequence[int], method: TableMethod, settings: Settings
) -> list[TableResult]:
    if not camelot_available():
        raise OptionalDependencyMissingError("Camelot is not installed (pip install -r requirements-optional.txt).")
    import camelot  # optional, heavy import

    flavor = "lattice" if method is TableMethod.CAMELOT_LATTICE else "stream"
    results: list[TableResult] = []
    with managed_temp_dir(settings) as tmp:
        path = tmp / "tables.pdf"  # Camelot only reads from a file path
        path.write_bytes(data)
        try:
            tables = camelot.read_pdf(str(path), pages=",".join(str(p) for p in page_numbers), flavor=flavor)
        except Exception as exc:  # Camelot surfaces many third-party exception types
            raise TableExtractionError(f"Camelot ({flavor}) could not process the PDF: {type(exc).__name__}.") from exc
        for table in tables:
            report = table.parsing_report or {}
            rows, padded = _rectangular(table.df.values.tolist())
            if not rows:
                continue
            accuracy = float(report.get("accuracy", 0.0))
            whitespace = float(report.get("whitespace", 0.0))
            result = TableResult(
                table_index=len(results) + 1,
                page_number=int(report.get("page", table.page)),
                method=method.value,
                columns=_column_names(len(rows[0])),
                rows=rows,
                quality_score=round(accuracy, 1),
                notes=[f"Camelot accuracy {accuracy:.0f}%, whitespace {whitespace:.0f}%"],
            )
            if accuracy < 90 or whitespace > 40:
                result.uncertain = True
                result.notes.append("Low Camelot accuracy or many empty cells; review the table.")
            _flag_sparse(result, padded)
            results.append(result)
    return results


def _flag_sparse(table: TableResult, padded: bool) -> None:
    cells = sum(len(r) for r in table.rows) or 1
    missing = table.missing_cell_count
    if padded:
        table.uncertain = True
        table.notes.append("Rows had different numbers of cells; short rows were padded with empty cells.")
    if missing:
        table.notes.append(f"{missing} empty or merged cell(s) left blank.")
        if missing / cells > 0.3:
            table.uncertain = True


# --------------------------------------------------------------------------------------
# Scanned pages (OCR word boxes)
# --------------------------------------------------------------------------------------
def extract_tables_from_ocr(ocr: OCRResult, image: Optional[Image.Image], page_number: int) -> list[TableResult]:
    """Rebuild at most one table per page from OCR word boxes. Returns [] if none is found."""
    words = [w for w in ocr.words if w.text.strip()]
    if len(words) < 4:
        return []
    if image is not None:
        grid = detect_grid_lines(image)
        if grid is not None:
            table = _table_from_grid(words, *grid, page_number)
            if table is not None:
                return [table]
    table = _table_from_whitespace(words, page_number)
    return [table] if table is not None else []


_LINE_ARTIFACT = re.compile(r"[|_\-—–=~\[\]]+")


def _cluster_positions(values: list[int], tolerance: int) -> list[int]:
    clusters: list[list[int]] = []
    for v in sorted(values):
        if clusters and v - clusters[-1][-1] <= tolerance:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [int(statistics.mean(c)) for c in clusters]


def _line_masks(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Binary masks of long horizontal and vertical strokes (candidate ruling lines)."""
    gray = np.asarray(image.convert("L"))
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    height, width = binary.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, width // 30), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, height // 30)))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel), cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)


def remove_ruling_lines(image: Image.Image) -> tuple[Image.Image, bool]:
    """Paint detected table grid lines white so OCR does not read them as '|' or '_'.

    Only applied when a grid is actually detected; returns (image, whether lines were removed).
    """
    if detect_grid_lines(image) is None:
        return image, False
    horizontal, vertical = _line_masks(image)
    mask = cv2.dilate(cv2.bitwise_or(horizontal, vertical), np.ones((3, 3), np.uint8), iterations=1)
    arr = np.asarray(image).copy()
    arr[mask > 0] = 255
    return Image.fromarray(arr), True


def detect_grid_lines(image: Image.Image) -> Optional[tuple[list[int], list[int]]]:
    """Detect ruled table lines. Returns (row boundaries y, column boundaries x) or None."""
    horizontal, vertical = _line_masks(image)
    height, width = horizontal.shape

    def line_positions(mask: np.ndarray, axis: int, min_length: int) -> list[int]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        positions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            length, pos = (w, y + h // 2) if axis == 0 else (h, x + w // 2)
            if length >= min_length:
                positions.append(pos)
        return _cluster_positions(positions, tolerance=6)

    rows = line_positions(horizontal, 0, width // 5)
    cols = line_positions(vertical, 1, height // 10)
    if len(rows) < 2 or len(cols) < 2:
        return None
    return rows, cols


def _table_from_grid(words: list[OCRWord], rows_y: list[int], cols_x: list[int], page_number: int) -> Optional[TableResult]:
    n_rows, n_cols = len(rows_y) - 1, len(cols_x) - 1
    cells: list[list[list[OCRWord]]] = [[[] for _ in range(n_cols)] for _ in range(n_rows)]
    straddling = 0
    placed = 0
    artifacts = [w for w in words if _LINE_ARTIFACT.fullmatch(w.text)]
    words = [w for w in words if not _LINE_ARTIFACT.fullmatch(w.text)]
    for word in words:
        cx, cy = word.bbox.center_x, word.bbox.center_y
        if not (rows_y[0] <= cy <= rows_y[-1] and cols_x[0] <= cx <= cols_x[-1]):
            continue
        r = max(i for i in range(n_rows) if rows_y[i] <= cy)
        c = max(i for i in range(n_cols) if cols_x[i] <= cx)
        if word.bbox.x < cols_x[c] - 3 or word.bbox.right > cols_x[c + 1] + 3:
            straddling += 1
        cells[min(r, n_rows - 1)][min(c, n_cols - 1)].append(word)
        placed += 1
    if placed == 0:
        return None
    rows = [[_join_words(cell) for cell in row] for row in cells]
    table = TableResult(
        table_index=1,
        page_number=page_number,
        method="ocr_grid",
        columns=_column_names(n_cols),
        rows=rows,
        notes=["Cells defined by detected ruling lines; text from OCR."],
    )
    if artifacts:
        table.notes.append(f"{len(artifacts)} token(s) made only of line characters (e.g. '|') were treated as ruling lines.")
    if straddling:
        table.uncertain = True
        table.notes.append(f"{straddling} word(s) cross a cell boundary; check for merged cells.")
    if any(w.low_confidence for row in cells for cell in row for w in cell):
        table.uncertain = True
        table.notes.append("Some cells contain low-confidence words.")
    _flag_sparse(table, padded=False)
    return table


def _join_words(cell_words: list[OCRWord]) -> Optional[str]:
    if not cell_words:
        return None
    ordered = sorted(cell_words, key=lambda w: (round(w.bbox.center_y / max(1, w.bbox.height)), w.bbox.x))
    return " ".join(w.text for w in ordered)


def _group_rows(words: list[OCRWord]) -> list[list[OCRWord]]:
    median_h = statistics.median(w.bbox.height for w in words) or 1
    rows: list[list[OCRWord]] = []
    for word in sorted(words, key=lambda w: w.bbox.center_y):
        if rows and abs(word.bbox.center_y - statistics.mean(w.bbox.center_y for w in rows[-1])) <= median_h * 0.6:
            rows[-1].append(word)
        else:
            rows.append([word])
    return [sorted(r, key=lambda w: w.bbox.x) for r in rows]


def _table_from_whitespace(words: list[OCRWord], page_number: int) -> Optional[TableResult]:
    median_h = statistics.median(w.bbox.height for w in words) or 1
    gap_min = median_h * 1.5
    rows = _group_rows(words)

    def has_column_gap(row: list[OCRWord]) -> bool:
        return any(b.bbox.x - a.bbox.right >= gap_min for a, b in zip(row, row[1:]))

    # Longest contiguous run of rows that contain a column-sized gap.
    best: list[list[OCRWord]] = []
    run: list[list[OCRWord]] = []
    for row in rows:
        if has_column_gap(row):
            run.append(row)
            if len(run) > len(best):
                best = list(run)
        else:
            run = []
    if len(best) < 2:
        return None

    # Column separators = horizontal gaps not covered by any word in the run.
    left = min(w.bbox.x for r in best for w in r)
    right = max(w.bbox.right for r in best for w in r)
    coverage = np.zeros(right - left + 1, dtype=np.int32)
    for row in best:
        for w in row:
            coverage[w.bbox.x - left : w.bbox.right - left + 1] += 1
    separators: list[int] = []
    start = None
    for i, value in enumerate(coverage):
        if value == 0 and start is None:
            start = i
        elif value != 0 and start is not None:
            if i - start >= gap_min:
                separators.append(left + (start + i) // 2)
            start = None
    if not separators:
        return None

    bounds = [left - 1, *separators, right + 1]
    n_cols = len(bounds) - 1
    table_rows: list[list[Optional[str]]] = []
    straddling = 0
    for row in best:
        cells: list[list[OCRWord]] = [[] for _ in range(n_cols)]
        for w in row:
            col = next(i for i in range(n_cols) if bounds[i] <= w.bbox.center_x < bounds[i + 1] or i == n_cols - 1)
            if w.bbox.x < bounds[col] or w.bbox.right > bounds[col + 1]:
                straddling += 1
            cells[col].append(w)
        table_rows.append([_join_words(c) for c in cells])

    table = TableResult(
        table_index=1,
        page_number=page_number,
        method="ocr_whitespace",
        columns=_column_names(n_cols),
        rows=table_rows,
        uncertain=True,
        notes=["No ruling lines found; columns were inferred from whitespace between OCR words. Review carefully."],
    )
    if straddling:
        table.notes.append(f"{straddling} word(s) cross an inferred column boundary.")
    _flag_sparse(table, padded=False)
    return table


def table_with_header(table: TableResult) -> TableResult:
    """Return a copy that uses the first row as column names (blank names get placeholders)."""
    if not table.rows:
        return table
    header = [cell or f"Column {i + 1}" for i, cell in enumerate(table.rows[0])]
    seen: dict[str, int] = {}
    unique = []
    for name in header:  # duplicate column names break CSV/XLSX consumers
        seen[name] = seen.get(name, 0) + 1
        unique.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return table.model_copy(update={"columns": unique, "rows": table.rows[1:]})
