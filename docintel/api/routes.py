"""API routes. /process, /extract and /analyze work in memory and store nothing.

/batches queues files for background processing; like the app's Batch page, it keeps copies
of the uploaded files and the results under settings.batch_dir until the batch is deleted.
"""

from __future__ import annotations

import io
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from docintel import __version__
from docintel.api.schemas import AnalyzeTextRequest, EngineInfo, ExtractTextRequest, HealthResponse, ProcessResponse
from docintel.batch.models import Batch, OutputFormat
from docintel.batch.package import build_zip
from docintel.batch.runner import BatchRunner
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


# --- batches ---------------------------------------------------------------------------------------
# Local file locations are never returned (they can contain the Windows user name).
_PRIVATE = {"owner_pid": True, "items": {"__all__": {"source_path", "stored_input"}}}


def get_runner(request: Request) -> BatchRunner:
    return request.app.state.batch_runner


def _batch_or_404(runner: BatchRunner, batch_id: str) -> Batch:
    try:
        batch = runner.store.get(batch_id)
    except ValueError:
        batch = None
    if batch is None:
        raise HTTPException(status_code=404, detail="Batch not found.")
    return batch


def _public(batch: Batch) -> dict:
    return batch.model_dump(mode="json", exclude=_PRIVATE)


@router.post("/batches", status_code=202, tags=["batches"])
async def create_batch(
    files: list[UploadFile] = File(..., description="Images and/or PDFs (total size within the request limit)"),
    title: str = Form("API batch", max_length=120),
    engine: str = Form("tesseract"),
    languages: str = Form("eng", pattern=r"^[A-Za-z0-9_]+(\+[A-Za-z0-9_]+)*$", max_length=200),
    psm: Optional[int] = Form(3, ge=0, le=13),
    preprocess: str = Form("smart"),
    pdf_mode: PDFMode = Form(PDFMode.AUTO),
    pages: str = Form("", max_length=200),
    extract_tables: bool = Form(False),
    template: Optional[str] = Form(None, max_length=64),
    outputs: str = Form("text,json", description="Comma list: " + ", ".join(o.value for o in OutputFormat)),
    runner: BatchRunner = Depends(get_runner),
) -> dict:
    """Queue files for background processing; poll GET /batches/{id}, then download /batches/{id}/zip."""
    if preprocess not in PRESETS:
        raise InputValidationError(f"Unknown preprocessing preset '{preprocess}'. Options: {', '.join(PRESETS)}.")
    try:
        formats = [OutputFormat(o.strip()) for o in outputs.split(",") if o.strip()]
    except ValueError as exc:
        raise InputValidationError(f"Unknown output format in '{outputs}'.") from exc
    settings = runner.settings
    uploads, total = [], 0
    for upload in files:
        data = await read_limited(upload, settings.api_max_request_bytes)
        total += len(data)
        if total > settings.api_max_request_bytes:
            raise FileTooLargeError(f"Uploads exceed the {settings.api_max_request_mb:.0f} MB limit.")
        uploads.append((upload.filename or f"file{len(uploads) + 1}", data))
    options = PipelineOptions(
        engine=engine,
        ocr=OCROptions(languages=tuple(languages.split("+")), psm=psm, timeout_seconds=settings.ocr_timeout_seconds,
                       low_confidence_threshold=settings.low_confidence_threshold),
        preprocess_preset=preprocess, pdf_mode=pdf_mode, page_range=pages, extract_tables=extract_tables,
        template=TemplateStore(settings.templates_dir).load(template) if template else None,
    )
    try:
        batch = runner.create(title=title, options=options, outputs=formats, uploads=uploads)
    except ValueError as exc:
        raise InputValidationError(str(exc)) from exc
    return _public(batch)


@router.get("/batches", tags=["batches"])
def list_batches(runner: BatchRunner = Depends(get_runner)) -> list[dict]:
    return [_public(b) for b in runner.store.list()]


@router.get("/batches/{batch_id}", tags=["batches"])
def get_batch(batch_id: str, runner: BatchRunner = Depends(get_runner)) -> dict:
    return _public(_batch_or_404(runner, batch_id))


@router.post("/batches/{batch_id}/{action}", tags=["batches"])
def control_batch(batch_id: str, action: str, stronger: bool = Query(False),
                  runner: BatchRunner = Depends(get_runner)) -> dict:
    """action: pause (after the current file), resume, cancel (between pages) or retry (failed/cancelled files)."""
    _batch_or_404(runner, batch_id)
    if action == "pause":
        runner.pause(batch_id)
    elif action == "resume":
        runner.start(batch_id)
    elif action == "cancel":
        runner.cancel(batch_id)
    elif action == "retry":
        runner.retry_failed(batch_id, stronger_settings=stronger)
    else:
        raise HTTPException(status_code=404, detail="Unknown action. Use pause, resume, cancel or retry.")
    return _public(_batch_or_404(runner, batch_id))


@router.get("/batches/{batch_id}/zip", tags=["batches"], response_class=Response)
def batch_zip(batch_id: str, mask_ids: bool = Query(True), runner: BatchRunner = Depends(get_runner)) -> Response:
    batch = _batch_or_404(runner, batch_id)
    buffer = io.BytesIO()
    build_zip(runner.store, batch, buffer, mask_ids=mask_ids)
    return Response(buffer.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="DocIntel_batch_{batch_id}.zip"'})


@router.delete("/batches/{batch_id}", status_code=204, tags=["batches"])
def delete_batch(batch_id: str, runner: BatchRunner = Depends(get_runner)) -> Response:
    _batch_or_404(runner, batch_id)
    if not runner.delete(batch_id):
        raise HTTPException(status_code=409, detail="The batch is running; cancel it first.")
    return Response(status_code=204)
