"""Removal of invisible OCR text layers (used by Redo OCR)."""

from __future__ import annotations

import fitz
import pytest

pytest.importorskip("pikepdf")

from docintel.core.exceptions import CorruptFileError  # noqa: E402
from docintel.documents.text_layer import strip_invisible_text  # noqa: E402


def _text(data: bytes) -> list[str]:
    with fitz.open(stream=data, filetype="pdf") as doc:
        return [" ".join(page.get_text().split()) for page in doc]


def _pdf_with(visible: str | None, invisible: str | None) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    if visible:
        page.insert_text((72, 100), visible, fontsize=12)
    if invisible:
        page.insert_text((72, 200), invisible, fontsize=12, render_mode=3)
    return doc.tobytes()


def test_removes_invisible_keeps_visible():
    data = _pdf_with("Visible digital text", "OLD WRONG OCR LAYER")
    assert _text(data) == ["Visible digital text OLD WRONG OCR LAYER"]
    out, report = strip_invisible_text(data)
    assert _text(out) == ["Visible digital text"]
    assert report.text_objects_removed == 1 and report.pages_changed == 1


def test_no_invisible_text_returns_input_unchanged():
    data = _pdf_with("Only visible text", None)
    out, report = strip_invisible_text(data)
    assert out is data and report.text_objects_removed == 0


def test_invisible_text_inside_form_xobject():
    inner = fitz.open(stream=_pdf_with(None, "Hidden layer in a form"), filetype="pdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 80), "Page heading", fontsize=12)
    page.show_pdf_page(page.rect, inner, 0)  # embeds the other page as a form XObject
    data = doc.tobytes()
    assert "Hidden layer in a form" in _text(data)[0]
    out, report = strip_invisible_text(data)
    assert _text(out) == ["Page heading"]
    assert report.text_objects_removed == 1


def test_multi_page_only_affected_pages_change():
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), "Digital page", fontsize=12)
    doc.new_page().insert_text((72, 100), "ocr text", fontsize=12, render_mode=3)
    out, report = strip_invisible_text(doc.tobytes())
    assert _text(out) == ["Digital page", ""]
    assert report.pages_changed == 1


def test_text_state_preserved_after_removed_block():
    # The removed object sets the font; later visible text relies on it (no Tf of its own).
    import pikepdf

    pdf = pikepdf.new()
    font = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type1, BaseFont=pikepdf.Name.Helvetica))
    content = b"BT /F1 12 Tf 3 Tr 72 700 Td (hidden) Tj ET BT 0 Tr 72 650 Td (shown) Tj ET"
    page = pikepdf.Page(pikepdf.Dictionary(Type=pikepdf.Name.Page, MediaBox=[0, 0, 612, 792],
                                           Resources=pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font)),
                                           Contents=pdf.make_stream(content)))
    pdf.pages.append(page)
    import io

    buf = io.BytesIO()
    pdf.save(buf)
    out, report = strip_invisible_text(buf.getvalue())
    assert report.text_objects_removed == 1
    assert _text(out) == ["shown"]
    with pikepdf.open(io.BytesIO(out)) as result:  # keep the Pdf alive while reading its page
        ops = [str(i.operator) for i in pikepdf.parse_content_stream(result.pages[0])]
    assert ops[:4] == ["BT", "Tf", "Tr", "ET"]  # font/render state kept for the next object


def test_mixed_visible_and_invisible_object_is_kept():
    import io

    import pikepdf

    pdf = pikepdf.new()
    font = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type1, BaseFont=pikepdf.Name.Helvetica))
    content = b"BT /F1 12 Tf 72 700 Td (seen) Tj 3 Tr (unseen) Tj ET"
    pdf.pages.append(pikepdf.Page(pikepdf.Dictionary(Type=pikepdf.Name.Page, MediaBox=[0, 0, 612, 792],
                                                     Resources=pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font)),
                                                     Contents=pdf.make_stream(content))))
    buf = io.BytesIO()
    pdf.save(buf)
    out, report = strip_invisible_text(buf.getvalue())
    assert report.mixed_text_objects_kept == 1 and report.text_objects_removed == 0
    assert "seen" in _text(out)[0]


def test_corrupt_pdf():
    with pytest.raises(CorruptFileError):
        strip_invisible_text(b"%PDF-1.4 junk junk")
