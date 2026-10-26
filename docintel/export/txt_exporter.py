"""Plain-text export with explicit page boundaries (UTF-8)."""

from __future__ import annotations

from docintel.core.result_models import DocumentResult


def document_to_txt(document: DocumentResult, *, normalized: bool = True, include_header: bool = True) -> str:
    lines: list[str] = []
    if include_header:
        lines += [
            f"Source file: {document.file_info.filename}",
            f"Pages exported: {len(document.pages)}",
            f"Text version: {'normalized' if normalized else 'raw OCR / embedded text'}",
            "Note: OCR output may contain recognition errors; verify important content.",
            "",
        ]
    for page in document.pages:
        lines.append(f"===== Page {page.page_number} | source: {page.source.value} =====")
        if page.error:
            lines.append(f"[Page not processed: {page.error.message}]")
        else:
            lines.append(page.normalized_text if normalized else page.raw_text)
        lines.append("")
    return "\n".join(lines)


def document_to_txt_bytes(document: DocumentResult, *, normalized: bool = True) -> bytes:
    return document_to_txt(document, normalized=normalized).encode("utf-8")
