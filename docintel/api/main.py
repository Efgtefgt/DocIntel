"""FastAPI application factory.

Run locally (binds to localhost only)::

    python -m uvicorn docintel.api.main:app --host 127.0.0.1 --port 8000

The API has no authentication. Do not expose it beyond localhost without adding
authentication, TLS and access controls (see README, "API security").
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from docintel import __version__
from docintel.api.routes import router
from docintel.config.logging_config import configure_logging
from docintel.config.settings import Settings, get_settings
from docintel.core.exceptions import (
    DocIntelError,
    FileTooLargeError,
    InputValidationError,
    LanguageDataMissingError,
    OCREngineUnavailableError,
    ProcessingTimeoutError,
    TemplateError,
)
from docintel.core.pipeline import DocumentPipeline
from docintel.ocr.engine_factory import EngineFactory

logger = logging.getLogger(__name__)


def _status_for(exc: DocIntelError) -> int:
    if isinstance(exc, FileTooLargeError):
        return 413
    if isinstance(exc, (InputValidationError, LanguageDataMissingError)):
        return 422
    if isinstance(exc, TemplateError):
        return 404 if "does not exist" in exc.user_message else 422
    if isinstance(exc, OCREngineUnavailableError):
        return 503
    if isinstance(exc, ProcessingTimeoutError):
        return 504
    return 500


def create_app(settings: Optional[Settings] = None, engines: Optional[EngineFactory] = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)
    app = FastAPI(
        title="DocIntel API",
        version=__version__,
        description="Local, CPU-only OCR and document extraction. OCR output may contain errors; "
        "confidence values are engine-specific indicators, not probabilities.",
    )
    app.state.pipeline = DocumentPipeline(settings, engines)
    from docintel.batch.runner import BatchRunner

    app.state.batch_runner = BatchRunner(settings, factory_builder=(lambda _s: engines) if engines else None)

    @app.middleware("http")
    async def limit_request_size(request: Request, call_next):
        # Reject oversized bodies early when the client declares a length. Streaming bodies
        # without Content-Length are still capped while reading (routes.read_limited).
        declared = request.headers.get("content-length")
        limit = settings.api_max_request_bytes + 64 * 1024  # allowance for multipart overhead
        if declared and declared.isdigit() and int(declared) > limit:
            return JSONResponse(status_code=413, content={"code": "file_too_large", "message": "Request body too large."})
        return await call_next(request)

    @app.exception_handler(DocIntelError)
    async def docintel_error_handler(_: Request, exc: DocIntelError) -> JSONResponse:
        logger.info("API request failed", extra={"code": exc.code})
        return JSONResponse(status_code=_status_for(exc), content={"code": exc.code, "message": exc.user_message})

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5])
        return JSONResponse(status_code=422, content={"code": "invalid_request", "message": details})

    app.include_router(router)
    return app


app = create_app()
