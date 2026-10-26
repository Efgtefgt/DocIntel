"""Pydantic result models shared by all processing stages.

Raw OCR text and normalized text are stored separately so that every transformation
is reviewable. Confidence values are engine-specific indicators, not probabilities.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class FileKind(str, Enum):
    IMAGE = "image"
    PDF = "pdf"


class StageStatus(str, Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


class PageSource(str, Enum):
    EMBEDDED_TEXT = "embedded_text"
    OCR = "ocr"
    NONE = "none"


class StageError(BaseModel):
    """A user-safe description of a failure in one stage."""

    stage: str
    code: str
    message: str
    page_number: Optional[int] = None


class StageRecord(BaseModel):
    stage: str
    status: StageStatus
    duration_seconds: float = 0.0
    detail: str = ""


class FileInfo(BaseModel):
    filename: str
    kind: FileKind
    format: str
    mime_type: str
    size_bytes: int
    width: Optional[int] = None
    height: Optional[int] = None
    page_count: int = 1
    warnings: list[str] = Field(default_factory=list)


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2


class OCRWord(BaseModel):
    text: str
    confidence: Optional[float] = None
    bbox: BoundingBox
    block: int = 0
    paragraph: int = 0
    line: int = 0
    low_confidence: bool = False


class OCRResult(BaseModel):
    """Output of a single OCR engine run on a single image."""

    engine: str
    engine_version: str = ""
    languages: list[str]
    text: str
    words: list[OCRWord] = Field(default_factory=list)
    mean_confidence: Optional[float] = None
    confidence_scale: str = ""
    duration_seconds: float = 0.0
    image_width: int = 0
    image_height: int = 0
    options: dict[str, Any] = Field(default_factory=dict)

    @property
    def low_confidence_words(self) -> list[OCRWord]:
        return [w for w in self.words if w.low_confidence]


class PreprocessStepRecord(BaseModel):
    step: str
    applied: bool
    detail: str = ""


class QualityFinding(BaseModel):
    code: str
    severity: str  # info | warning | problem
    message: str
    step: Optional[str] = None


class OCRAttempt(BaseModel):
    """One OCR run on a page with a particular preprocessing plan."""

    label: str
    steps: list[str]
    mean_confidence: Optional[float] = None
    coverage: Optional[float] = None
    score: float = 0.0
    seconds: float = 0.0
    chosen: bool = False


class QualityReport(BaseModel):
    """How the page looked before OCR and how trustworthy the output appears.

    ``coverage`` compares recognised characters with an independent estimate of the characters
    visible on the page. ``verdict`` is a review hint, never a guarantee of correctness.
    """

    metrics: dict[str, Any] = Field(default_factory=dict)
    findings: list[QualityFinding] = Field(default_factory=list)
    plan: list[str] = Field(default_factory=list)
    attempts: list[OCRAttempt] = Field(default_factory=list)
    estimated_characters: Optional[int] = None
    recognised_characters: int = 0
    coverage: Optional[float] = None
    verdict: str = "review"
    notes: list[str] = Field(default_factory=list)


class PageResult(BaseModel):
    page_number: int
    source: PageSource = PageSource.NONE
    raw_text: str = ""
    normalized_text: str = ""
    normalization_changes: list[str] = Field(default_factory=list)
    ocr: Optional[OCRResult] = None
    preprocessing: list[PreprocessStepRecord] = Field(default_factory=list)
    quality: Optional[QualityReport] = None
    embedded_text_chars: int = 0
    detection_reason: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    duration_seconds: float = 0.0
    error: Optional[StageError] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class DocumentResult(BaseModel):
    file_info: FileInfo
    pages: list[PageResult] = Field(default_factory=list)
    stages: list[StageRecord] = Field(default_factory=list)
    errors: list[StageError] = Field(default_factory=list)
    total_duration_seconds: float = 0.0

    @property
    def succeeded(self) -> bool:
        return not self.errors and any(p.succeeded for p in self.pages)

    @property
    def failed_pages(self) -> list[PageResult]:
        return [p for p in self.pages if not p.succeeded]

    def combined_text(self, *, normalized: bool = True) -> str:
        """Return all page texts with explicit page-boundary markers."""
        parts = []
        for page in self.pages:
            body = page.normalized_text if normalized else page.raw_text
            parts.append(f"===== Page {page.page_number} ({page.source.value}) =====\n{body}")
        return "\n\n".join(parts)


class TableResult(BaseModel):
    """A table extracted from one page. ``None`` cells mean no text was found there."""

    table_index: int
    page_number: int
    method: str
    columns: list[str]
    rows: list[list[Optional[str]]]
    uncertain: bool = False
    quality_score: Optional[float] = None
    notes: list[str] = Field(default_factory=list)

    @property
    def missing_cell_count(self) -> int:
        return sum(1 for row in self.rows for cell in row if cell is None or cell == "")

    def to_dataframe(self):  # -> pandas.DataFrame (imported lazily to keep this module light)
        import pandas as pd

        return pd.DataFrame(self.rows, columns=self.columns)


class FieldStatus(str, Enum):
    FOUND = "found"
    MISSING = "missing"          # no label or value located in the text
    EMPTY = "empty"              # label located, but no value after it
    CONFLICT = "conflict"        # several different candidate values
    INVALID = "invalid"          # value located but failed type validation
    UNCERTAIN = "uncertain"      # value located with weak evidence (e.g. no label, low OCR confidence)
    CORRECTED = "corrected"      # value edited by a user


class ExtractedField(BaseModel):
    name: str
    field_type: str
    value: Optional[str] = None
    normalized_value: Optional[str] = None
    status: FieldStatus
    source_text: Optional[str] = None
    source_start: Optional[int] = None
    source_end: Optional[int] = None
    candidates: list[str] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    original_value: Optional[str] = None  # value before a user correction

    @property
    def needs_review(self) -> bool:
        return self.status not in (FieldStatus.FOUND, FieldStatus.CORRECTED)


class StructuredResult(BaseModel):
    template_name: str
    document_type: str
    fields: dict[str, ExtractedField]
    source_text: str
    valid: bool
    issues: list[str] = Field(default_factory=list)

    def to_flat_records(self) -> list[dict[str, Any]]:
        return [
            {
                "field": f.name,
                "type": f.field_type,
                "value": f.value,
                "normalized_value": f.normalized_value,
                "status": f.status.value,
                "source_text": f.source_text,
                "messages": "; ".join(f.messages),
            }
            for f in self.fields.values()
        ]


class EntityStatus(str, Enum):
    VALID = "valid"          # format (and checksum, where one exists) or label evidence confirms the type
    UNCERTAIN = "uncertain"  # pattern match only, ambiguous, or contains low-confidence OCR words
    INVALID = "invalid"      # looks like the type but fails its checksum or calendar check


class Entity(BaseModel):
    """A typed value found in the text (e-mail, phone, amount, ID number...). Never generated."""

    type: str
    value: str                       # verbatim text from the document
    normalized: Optional[str] = None
    status: EntityStatus = EntityStatus.UNCERTAIN
    label: Optional[str] = None      # label written next to the value, e.g. "Invoice No"
    page_number: Optional[int] = None
    line: str = ""
    start: int = 0                   # offsets into the page text
    end: int = 0
    sensitive: bool = False          # personal or financial identifier
    notes: list[str] = Field(default_factory=list)


class KeyValuePair(BaseModel):
    """A "Label: value" pair found in the text, with any label (no template needed)."""

    key: str
    value: str
    value_type: str = "text"         # entity type when one entity makes up the value
    page_number: Optional[int] = None
    line: str = ""
    start: int = 0
    end: int = 0
    empty: bool = False
    notes: list[str] = Field(default_factory=list)


class TypeScore(BaseModel):
    document_type: str
    label: str
    score: float
    evidence: list[str] = Field(default_factory=list)


class Classification(BaseModel):
    """Rule-based document-type guess. ``unknown`` when the evidence is too weak."""

    document_type: str = "unknown"
    label: str = "Unknown"
    confidence: str = "none"         # none | low | medium | high (strength of keyword evidence)
    scores: list[TypeScore] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    suggested_templates: list[str] = Field(default_factory=list)


class DocumentAnalysis(BaseModel):
    classification: Classification = Field(default_factory=Classification)
    entities: list[Entity] = Field(default_factory=list)
    key_values: list[KeyValuePair] = Field(default_factory=list)
    duration_seconds: float = 0.0

    def entity_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entity in self.entities:
            counts[entity.type] = counts.get(entity.type, 0) + 1
        return counts
