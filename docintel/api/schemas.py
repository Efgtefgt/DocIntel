"""API request/response schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from docintel.core.result_models import DocumentAnalysis, DocumentResult, StageError, StageRecord, StructuredResult, TableResult


class ErrorResponse(BaseModel):
    code: str
    message: str


class EngineInfo(BaseModel):
    name: str
    display_name: str
    available: bool
    reason: str = ""
    version: str = ""
    languages: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    version: str
    engines: list[EngineInfo]


class ProcessResponse(BaseModel):
    filename: str
    succeeded: bool
    duration_seconds: float
    document: Optional[DocumentResult] = None
    tables: list[TableResult] = Field(default_factory=list)
    structured: Optional[StructuredResult] = None
    analysis: Optional[DocumentAnalysis] = None
    stages: list[StageRecord] = Field(default_factory=list)
    errors: list[StageError] = Field(default_factory=list)


class ExtractTextRequest(BaseModel):
    text: str = Field(..., max_length=2_000_000)
    template: str = Field(..., description="Name of a stored template", max_length=64)


class AnalyzeTextRequest(BaseModel):
    text: str = Field(..., max_length=2_000_000)
