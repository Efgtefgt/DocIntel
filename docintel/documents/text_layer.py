"""Remove invisible OCR text layers from a PDF without touching visible content.

OCR tools (OCRmyPDF, scanners, Acrobat) add searchable text drawn with text render mode 3
("neither fill nor stroke"), usually inside a form XObject placed over the page image. This
module deletes text objects (``BT ... ET``) whose every glyph is drawn in mode 3. Visible
text, images, vector graphics and annotations are left as they are, so a page's digital
text is never lost.

Removed text objects are replaced by an empty ``BT ... ET`` that repeats their text-state
operators (font, spacing, render mode...), because that state persists after ``ET`` and
later content may depend on it.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import pikepdf

from docintel.core.exceptions import CorruptFileError, EncryptedDocumentError

logger = logging.getLogger(__name__)

_TEXT_SHOW = {"Tj", "TJ", "'", '"'}
_TEXT_STATE = {"Tc", "Tw", "Tz", "TL", "Tf", "Tr", "Ts"}
INVISIBLE = 3


@dataclass
class StripReport:
    pages_changed: int = 0
    text_objects_removed: int = 0
    mixed_text_objects_kept: int = 0   # contain visible and invisible glyphs; kept to avoid losing text


def _strip_instructions(instructions: list, report: StripReport) -> tuple[list, bool]:
    out: list = []
    changed = False
    render_mode = 0
    saved_modes: list[int] = []
    block: list | None = None
    shown_modes: list[int] = []

    for inst in instructions:
        name = str(inst.operator)
        if block is not None:
            block.append(inst)
            if name == "Tr" and inst.operands:
                render_mode = int(inst.operands[0])
            elif name in _TEXT_SHOW:
                shown_modes.append(render_mode)
            elif name == "ET":
                if shown_modes and all(m == INVISIBLE for m in shown_modes):
                    out.append(pikepdf.ContentStreamInstruction([], pikepdf.Operator("BT")))
                    out.extend(i for i in block if str(i.operator) in _TEXT_STATE)
                    out.append(pikepdf.ContentStreamInstruction([], pikepdf.Operator("ET")))
                    report.text_objects_removed += 1
                    changed = True
                else:
                    if INVISIBLE in shown_modes:
                        report.mixed_text_objects_kept += 1
                    out.extend(block)
                block = None
            continue
        if name == "BT":
            block, shown_modes = [inst], []
            continue
        if name == "q":
            saved_modes.append(render_mode)
        elif name == "Q" and saved_modes:
            render_mode = saved_modes.pop()
        elif name == "Tr" and inst.operands:
            render_mode = int(inst.operands[0])
        out.append(inst)
    if block is not None:  # unterminated text object (malformed): keep it unchanged
        out.extend(block)
    return out, changed


def _process_forms(resources: pikepdf.Object, report: StripReport, seen: set) -> bool:
    changed = False
    xobjects = resources.get("/XObject") if resources is not None else None
    if xobjects is None:
        return False
    for _, xobj in xobjects.items():
        if not isinstance(xobj, pikepdf.Stream) or xobj.get("/Subtype") != "/Form":
            continue
        key = xobj.objgen
        if key in seen:
            continue
        seen.add(key)
        new_ops, form_changed = _strip_instructions(list(pikepdf.parse_content_stream(xobj)), report)
        if form_changed:
            xobj.write(pikepdf.unparse_content_stream(new_ops))
            changed = True
        changed |= _process_forms(xobj.get("/Resources"), report, seen)
    return changed


def strip_invisible_text(data: bytes) -> tuple[bytes, StripReport]:
    """Return a copy of the PDF without invisible (OCR) text, plus what was removed."""
    report = StripReport()
    try:
        pdf = pikepdf.open(io.BytesIO(data))
    except pikepdf.PasswordError as exc:
        raise EncryptedDocumentError("The PDF is password-protected.") from exc
    except pikepdf.PdfError as exc:
        raise CorruptFileError("The PDF could not be read; it may be corrupt.") from exc
    with pdf:
        seen: set = set()
        for page in pdf.pages:
            page_changed = False
            new_ops, content_changed = _strip_instructions(list(pikepdf.parse_content_stream(page)), report)
            if content_changed:
                page.obj.Contents = pdf.make_stream(pikepdf.unparse_content_stream(new_ops))
                page_changed = True
            page_changed |= _process_forms(page.obj.get("/Resources"), report, seen)
            report.pages_changed += page_changed
        if not report.text_objects_removed:
            return data, report
        buffer = io.BytesIO()
        pdf.save(buffer)
    logger.info("Invisible text removed", extra={"pages": report.pages_changed, "objects": report.text_objects_removed})
    return buffer.getvalue(), report
