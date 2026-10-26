"""API routes. All processing happens in memory; uploads are never stored."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from docintel import __version__
from docintel.api.schemas import AnalyzeTextRequest, EngineInfo, ExtractTextRequest, HealthResponse, ProcessResponse
from docintel.core.exceptions import FileTooLargeError, InputValidationError
from docintel.core.pipeline import DocumentPipeline, PipelineOptions, PipelineResult
from docintel.core.result_models import DocumentAnalysis, StructuredResult
from docintel.documents.document_processor import PDFMode
from docintel.extraction.document_analysis import analyze_text
from docintel.extraction.structured_extractor import StructuredExtractor
from docintel.extraction.table_extractor import TableMethod
from docintel.extraction.templates import TemplateStore
from docintel.ocr.base_engine import OCROptions
from docintel.preprocessing.preprocessing_pipeline import PRESETS

router = APIRouter()
_CHUNK = 1024 * 1024


def get_pipeline(request: Request) -> DocumentPipeline:
    return request.app.state.pipeline


async def read_limited(upload: UploadFile, limit: int) -> bytes:
    """Read an upload in chunks, aborting as soon as it exceeds ``limit`` bytes."""
    chunks, size = [], 0
    while chunk := await upload.read(_CHUNK):
        size += len(chunk)
        if size > limit:
            raise FileTooLargeError(f"Upload exceeds the {limit // 1_048_576} MB limit.")
        chunks.append(chunk)
    return b"".join(chunks)


def _to_response(result: PipelineResult) -> ProcessResponse:
    return ProcessResponse(
        filename=result.filename,
        succeeded=result.succeeded,
        duration_seconds=result.duration_seconds,
        document=result.document,
        tables=result.tables,
        structured=result.structured,
        analysis=result.analysis,
        stages=result.stages,
        errors=result.errors,
    )


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(pipeline: DocumentPipeline = Depends(get_pipeline)) -> HealthResponse:
    engines = [EngineInfo(**{k: getattr(s, k) for k in EngineInfo.model_fields}) for s in pipeline.engines.statuses()]
    return HealthResponse(status="ok", version=__version__, engines=engines)


@router.get("/templates", response_model=list[str], tags=["extraction"])
def list_templates(pipeline: DocumentPipeline = Depends(get_pipeline)) -> list[str]:
    return TemplateStore(pipeline.settings.templates_dir).list_names()


@router.post("/process", response_model=ProcessResponse, tags=["processing"])
async def process_document(
    file: UploadFile = File(..., description="Image (PNG/JPEG/TIFF/BMP/WEBP) or PDF"),
    engine: str = Form("tesseract"),
    languages: str = Form("eng", pattern=r"^[A-Za-z0-9_]+(\+[A-Za-z0-9_]+)*$", max_length=200),
    psm: Optional[int] = Form(3, ge=0, le=13),
    preprocess: str = Form("smart"),
    pdf_mode: PDFMode = Form(PDFMode.AUTO),
    pages: str = Form("", max_length=200),
    extract_tables: bool = Form(False),
    table_method: TableMethod = Form(TableMethod.AUTO),
    template: Optional[str] = Form(None, max_length=64),
    pipeline: DocumentPipeline = Depends(get_pipeline),
) -> ProcessResponse:
    """Run the full pipeline on one uploaded file."""
    if preprocess not in PRESETS:
        raise InputValidationError(f"Unknown preprocessing preset '{preprocess}'. Options: {', '.join(PRESETS)}.")
    data = await read_limited(file, pipeline.settings.api_max_request_bytes)
    tmpl = TemplateStore(pipeline.settings.templates_dir).load(template) if template else None
    options = PipelineOptions(
        engine=engine,
        ocr=OCROptions(
            languages=tuple(languages.split("+")),
            psm=psm,
            timeout_seconds=pipeline.settings.ocr_timeout_seconds,
            low_confidence_threshold=pipeline.settings.low_confidence_threshold,
        ),
        preprocess_preset=preprocess,
        pdf_mode=pdf_mode,
        page_range=pages,
        extract_tables=extract_tables,
        table_method=table_method,
        template=tmpl,
    )
    # OCR is CPU-bound and blocking; run it off the event loop.
    result = await run_in_threadpool(pipeline.process, data, file.filename or "upload", options)
    return _to_response(result)


@router.post("/extract", response_model=StructuredResult, tags=["extraction"])
def extract_fields(body: ExtractTextRequest, pipeline: DocumentPipeline = Depends(get_pipeline)) -> StructuredResult:
    """Apply a stored template to text that was already recognised."""
    template = TemplateStore(pipeline.settings.templates_dir).load(body.template)
    return StructuredExtractor(template).extract(body.text)


@router.post("/analyze", response_model=DocumentAnalysis, tags=["extraction"])
def analyze(body: AnalyzeTextRequest, pipeline: DocumentPipeline = Depends(get_pipeline)) -> DocumentAnalysis:
    """Classify text and list the entities and key-value pairs found in it (no template needed)."""
    return analyze_text(body.text, template_types=TemplateStore(pipeline.settings.templates_dir).document_types())
