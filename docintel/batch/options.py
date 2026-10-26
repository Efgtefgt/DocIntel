"""Save and restore PipelineOptions as plain JSON, so a batch can be resumed after a restart.

Templates are stored by name and reloaded; a template deleted in the meantime is dropped
with a warning instead of failing the batch.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from docintel.core.exceptions import DocIntelError
from docintel.core.pipeline import PipelineOptions
from docintel.documents.document_processor import PDFMode
from docintel.extraction.table_extractor import TableMethod
from docintel.extraction.templates import TemplateStore
from docintel.extraction.text_normalizer import NormalizationOptions
from docintel.ocr.base_engine import OCROptions

logger = logging.getLogger(__name__)


def options_to_dict(options: PipelineOptions) -> dict[str, Any]:
    return {
        "engine": options.engine,
        "ocr": {**asdict(options.ocr), "languages": list(options.ocr.languages)},
        "preprocess_preset": options.preprocess_preset,
        "custom_steps": list(options.custom_steps) if options.custom_steps is not None else None,
        "pdf_mode": options.pdf_mode.value,
        "page_range": options.page_range,
        "normalization": asdict(options.normalization),
        "extract_tables": options.extract_tables,
        "table_method": options.table_method.value,
        "template": options.template.name if options.template else None,
        "analyze": options.analyze,
    }


def options_from_dict(data: dict[str, Any], templates_dir: Path) -> tuple[PipelineOptions, list[str]]:
    """Rebuild options; returns (options, warnings)."""
    warnings: list[str] = []
    template = None
    if data.get("template"):
        try:
            template = TemplateStore(templates_dir).load(data["template"])
        except DocIntelError:
            warnings.append(f"Template '{data['template']}' is no longer available; fields were not extracted.")
    ocr = dict(data.get("ocr") or {})
    ocr["languages"] = tuple(ocr.get("languages") or ("eng",))
    steps = data.get("custom_steps")
    options = PipelineOptions(
        engine=data.get("engine", "tesseract"),
        ocr=OCROptions(**ocr),
        preprocess_preset=data.get("preprocess_preset", "smart"),
        custom_steps=tuple(steps) if steps is not None else None,
        pdf_mode=PDFMode(data.get("pdf_mode", PDFMode.AUTO.value)),
        page_range=data.get("page_range", ""),
        normalization=NormalizationOptions(**(data.get("normalization") or {})),
        extract_tables=bool(data.get("extract_tables", False)),
        table_method=TableMethod(data.get("table_method", TableMethod.AUTO.value)),
        template=template,
        analyze=bool(data.get("analyze", True)),
    )
    return options, warnings


def stronger(options: PipelineOptions) -> PipelineOptions:
    """Options for a "retry with stronger settings": the smart preset (analysis + retries)."""
    if options.custom_steps is None and options.preprocess_preset in ("original", "grayscale"):
        return replace(options, preprocess_preset="smart")
    return options
