"""Extraction templates and their JSON storage.

A template maps field names to specs. The short form ``"caller_name": "string"`` is
accepted and expanded to a spec whose label is derived from the field name
("caller name"). The long form allows label aliases, a custom regex and flags::

    "phone_number": {"type": "phone", "labels": ["Phone", "Contact No"], "required": true}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from docintel.core.exceptions import TemplateError
from docintel.core.security import ensure_within, safe_filename

FieldType = Literal["string", "text", "phone", "email", "date", "number", "amount"]
_FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
MAX_PATTERN_LENGTH = 300


class FieldSpec(BaseModel):
    type: FieldType = "string"
    labels: list[str] = Field(default_factory=list)
    pattern: Optional[str] = None
    required: bool = False
    multiline: bool = False
    search_unlabeled: Optional[bool] = None  # default: True for phone/email/date

    @field_validator("labels")
    @classmethod
    def _labels(cls, labels: list[str]) -> list[str]:
        cleaned = [label.strip() for label in labels if label and label.strip()]
        if any(len(label) > 60 for label in cleaned):
            raise ValueError("labels must be at most 60 characters")
        return cleaned

    @field_validator("pattern")
    @classmethod
    def _pattern(cls, pattern: Optional[str]) -> Optional[str]:
        if pattern is None or not pattern.strip():
            return None
        if len(pattern) > MAX_PATTERN_LENGTH:
            raise ValueError(f"pattern must be at most {MAX_PATTERN_LENGTH} characters")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid regular expression: {exc}") from exc
        return pattern

    @property
    def scans_unlabeled(self) -> bool:
        if self.search_unlabeled is not None:
            return self.search_unlabeled
        return self.type in ("phone", "email", "date")


class ExtractionTemplate(BaseModel):
    name: str
    document_type: str
    description: str = ""
    fields: dict[str, FieldSpec]

    @field_validator("name", "document_type")
    @classmethod
    def _identifier(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_\- ]{1,64}", value):
            raise ValueError("use 1-64 letters, digits, spaces, '-' or '_'")
        return value

    @field_validator("fields", mode="before")
    @classmethod
    def _expand_short_form(cls, fields: object) -> object:
        if not isinstance(fields, dict) or not fields:
            raise ValueError("fields must be a non-empty object")
        expanded = {}
        for name, spec in fields.items():
            if not isinstance(name, str) or not _FIELD_NAME.match(name):
                raise ValueError(f"invalid field name '{name}' (use lower_snake_case)")
            expanded[name] = {"type": spec} if isinstance(spec, str) else spec
        return expanded

    def labels_for(self, field_name: str) -> list[str]:
        spec = self.fields[field_name]
        return spec.labels or [field_name.replace("_", " ")]


def parse_template(raw: str | dict) -> ExtractionTemplate:
    """Parse and validate a template from JSON text or a dict."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return ExtractionTemplate.model_validate(data)
    except json.JSONDecodeError as exc:
        raise TemplateError(f"Template is not valid JSON: {exc.msg} (line {exc.lineno}).") from exc
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5])
        raise TemplateError(f"Template is invalid: {problems}") from exc


class TemplateStore:
    """Reads and writes ``*.json`` templates inside one directory only."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def _path(self, name: str) -> Path:
        filename = safe_filename(name.replace(" ", "_"), default="template")
        if not filename.endswith(".json"):
            filename += ".json"
        return ensure_within(self._dir, Path(filename))

    def list_names(self) -> list[str]:
        if not self._dir.is_dir():
            return []
        return sorted(p.stem for p in self._dir.glob("*.json"))

    def document_types(self) -> dict[str, str]:
        """Template name -> document type, skipping templates that fail to load."""
        types = {}
        for name in self.list_names():
            try:
                types[name] = self.load(name).document_type
            except TemplateError:
                continue
        return types

    def load(self, name: str) -> ExtractionTemplate:
        path = self._path(name)
        if not path.is_file():
            raise TemplateError(f"Template '{name}' does not exist.")
        return parse_template(path.read_text(encoding="utf-8"))

    def save(self, template: ExtractionTemplate, *, overwrite: bool = False) -> Path:
        path = self._path(template.name)
        if path.exists() and not overwrite:
            raise TemplateError(f"Template '{template.name}' already exists; enable overwrite to replace it.")
        self._dir.mkdir(parents=True, exist_ok=True)
        path.write_text(template.model_dump_json(indent=2, exclude_defaults=True), encoding="utf-8")
        return path
