"""Deterministic template-based field extraction (regular expressions and layout rules).

Rules:
* A value is only returned when it appears in the text; otherwise the field is ``missing``.
* A label with nothing after it is ``empty`` (distinct from ``missing``).
* Different values for one field are a ``conflict``; no candidate is picked automatically.
* Values found without a label, ambiguous dates and low-confidence OCR words are ``uncertain``.
* Values that fail type validation are kept verbatim and marked ``invalid``.
No LLM is used and no value is ever generated, completed or corrected automatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from pydantic import ValidationError

from docintel.core.result_models import ExtractedField, FieldStatus, StructuredResult
from docintel.extraction.field_validators import UNLABELED_PATTERNS, build_record_model, check_value
from docintel.extraction.templates import ExtractionTemplate, FieldSpec

_SEPARATOR = r"[ \t]*(?:[:=\-–—]|\.{2,}|[ \t]{2,})[ \t]*"
_BULLET = r"[ \t>*•\-]*"
_STRIP_TOKEN = re.compile(r"^\W+|\W+$")


@dataclass
class _Candidate:
    value: str
    start: int
    end: int
    line: str
    labeled: bool


def _label_regex(label: str) -> str:
    words = [re.escape(w) for w in label.split()]
    return r"[\s_]+".join(words)


def _line_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    return line_start, len(text) if line_end == -1 else line_end


def _looks_like_label_line(line: str, all_labels: list[str]) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r"^[A-Za-z][\w ()/#.]{0,40}:\s*", stripped):
        return True
    return any(re.match(rf"^{_BULLET}{_label_regex(lbl)}\b", stripped, re.IGNORECASE) for lbl in all_labels)


class StructuredExtractor:
    def __init__(self, template: ExtractionTemplate) -> None:
        self.template = template
        self._all_labels = [lbl for name in template.fields for lbl in template.labels_for(name)]
        self._model = build_record_model(template.name, {n: s.type for n, s in template.fields.items()})

    # --- candidate search ------------------------------------------------------------
    def _labeled_candidates(self, text: str, name: str, spec: FieldSpec) -> tuple[list[_Candidate], bool]:
        """Return candidates found next to a label, and whether any label occurrence was seen."""
        candidates: list[_Candidate] = []
        label_seen = False
        lines = text.split("\n")
        offsets = [0]
        for line in lines[:-1]:
            offsets.append(offsets[-1] + len(line) + 1)

        for label in self.template.labels_for(name):
            # "Label: value" on one line. The label may follow other content after 2+ spaces
            # or a pipe ("Name: A   Phone: B"); the value stops before the next "Key:" segment.
            pattern = re.compile(
                rf"(?:^{_BULLET}|[ \t]{{2,}}|\|[ \t]*)(?P<label>{_label_regex(label)}){_SEPARATOR}"
                rf"(?P<value>.*?)(?=[ \t]{{2,}}[A-Za-z][\w ]{{0,30}}:|[ \t]*\||$)",
                re.IGNORECASE | re.MULTILINE,
            )
            for match in pattern.finditer(text):
                label_seen = True
                value = match.group("value").strip()
                start, end = match.start("value"), match.start("value") + len(match.group("value").rstrip())
                line_start, line_end = _line_bounds(text, match.start("label"), match.end())
                line_no = text.count("\n", 0, match.start("label"))
                if spec.multiline or not value:
                    # Continue onto following lines until a blank line or another label.
                    extra: list[str] = []
                    for nxt in lines[line_no + 1 :]:
                        if not nxt.strip() or _looks_like_label_line(nxt, self._all_labels):
                            break
                        extra.append(nxt.strip())
                        end = offsets[line_no + 1 + len(extra) - 1] + len(nxt)
                        if not spec.multiline:
                            break  # single-line field: value is on the next line only
                    if extra:
                        value = " ".join([value, *extra]).strip() if value else " ".join(extra)
                        if not match.group("value").strip():
                            start = offsets[line_no + 1]
                        line_end = end
                candidates.append(_Candidate(value, start, end, text[line_start:line_end], labeled=True))
        return candidates, label_seen

    def _pattern_candidates(self, text: str, pattern: str) -> list[_Candidate]:
        found = []
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            group = "value" if "value" in match.re.groupindex else (1 if match.re.groups else 0)
            value = (match.group(group) or "").strip()
            start, end = match.span(group)
            ls, le = _line_bounds(text, start, end)
            found.append(_Candidate(value, start, end, text[ls:le], labeled=True))
        return found

    def _unlabeled_candidates(self, text: str, spec: FieldSpec) -> list[_Candidate]:
        regex = UNLABELED_PATTERNS.get(spec.type)
        if regex is None:
            return []
        found = []
        for match in regex.finditer(text):
            ls, le = _line_bounds(text, match.start(), match.end())
            found.append(_Candidate(match.group(0).strip(), match.start(), match.end(), text[ls:le], labeled=False))
        return found

    # --- main entry point ------------------------------------------------------------
    def extract(self, text: str, low_confidence_words: Optional[Iterable[str]] = None) -> StructuredResult:
        low_conf = {_token_key(w) for w in (low_confidence_words or []) if _token_key(w)}
        fields: dict[str, ExtractedField] = {}
        for name, spec in self.template.fields.items():
            fields[name] = self._extract_field(text, name, spec, low_conf)
        return self._finalize(fields, text)

    def _extract_field(self, text: str, name: str, spec: FieldSpec, low_conf: set[str]) -> ExtractedField:
        result = ExtractedField(name=name, field_type=spec.type, status=FieldStatus.MISSING)
        if spec.pattern:
            candidates, label_seen = self._pattern_candidates(text, spec.pattern), False
        else:
            candidates, label_seen = self._labeled_candidates(text, name, spec)
        non_empty = [c for c in candidates if c.value]

        if not non_empty and spec.scans_unlabeled and not label_seen:
            non_empty = self._unlabeled_candidates(text, spec)
        if not non_empty:
            if label_seen or candidates:
                result.status, result.value = FieldStatus.EMPTY, ""
                result.messages.append("Label found but no value follows it.")
                if candidates:
                    c = candidates[0]
                    result.source_text, result.source_start, result.source_end = c.line, c.start, c.end
            else:
                result.messages.append("Not found in the document.")
            return result

        distinct: dict[str, _Candidate] = {}
        for cand in non_empty:
            distinct.setdefault(re.sub(r"\s+", " ", cand.value).casefold(), cand)
        result.candidates = [c.value for c in distinct.values()]
        if len(distinct) > 1:
            result.status = FieldStatus.CONFLICT
            result.messages.append(f"{len(distinct)} different values found; choose the correct one.")
            result.source_text = "\n".join(c.line for c in distinct.values())
            return result

        chosen = next(iter(distinct.values()))
        result.value = chosen.value
        result.source_text, result.source_start, result.source_end = chosen.line, chosen.start, chosen.end
        check = check_value(spec.type, chosen.value)
        if not check.ok:
            result.status = FieldStatus.INVALID
            result.messages.append(f"Value {check.message}.")
            return result
        result.normalized_value = check.normalized
        result.status = FieldStatus.FOUND
        if check.uncertain:
            result.status = FieldStatus.UNCERTAIN
            result.messages.append(f"Value is uncertain: {check.message}.")
        if not chosen.labeled:
            result.status = FieldStatus.UNCERTAIN
            result.messages.append("Found by pattern without a matching label.")
        if low_conf and any(_token_key(tok) in low_conf for tok in chosen.value.split()):
            result.status = FieldStatus.UNCERTAIN
            result.messages.append("Contains low-confidence OCR words.")
        return result

    def _finalize(self, fields: dict[str, ExtractedField], text: str) -> StructuredResult:
        issues: list[str] = []
        values = {n: f.value for n, f in fields.items() if f.status not in (FieldStatus.CONFLICT, FieldStatus.EMPTY)}
        try:
            self._model.model_validate(values)
        except ValidationError as exc:
            for error in exc.errors():
                field_name = str(error["loc"][0]) if error["loc"] else ""
                if field_name in fields and fields[field_name].status is not FieldStatus.INVALID:
                    fields[field_name].status = FieldStatus.INVALID
                    fields[field_name].messages.append(error["msg"])
        for name, spec in self.template.fields.items():
            status = fields[name].status
            if spec.required and status in (FieldStatus.MISSING, FieldStatus.EMPTY):
                issues.append(f"Required field '{name}' is {status.value}.")
            if status in (FieldStatus.CONFLICT, FieldStatus.INVALID):
                issues.append(f"Field '{name}' is {status.value}.")
        return StructuredResult(
            template_name=self.template.name,
            document_type=self.template.document_type,
            fields=fields,
            source_text=text,
            valid=not issues,
            issues=issues,
        )

    def apply_corrections(self, result: StructuredResult, corrections: dict[str, Optional[str]]) -> StructuredResult:
        """Apply user-entered values. Originals are preserved and corrected values re-validated."""
        fields = {n: f.model_copy(deep=True) for n, f in result.fields.items()}
        for name, new_value in corrections.items():
            if name not in fields:
                continue
            field = fields[name]
            new_value = new_value.strip() if isinstance(new_value, str) else None
            if (new_value or None) == (field.value or None):
                continue
            field.original_value = field.original_value if field.status is FieldStatus.CORRECTED else field.value
            field.value = new_value or None
            field.messages = ["Value entered by user."]
            if field.value is None:
                field.status, field.normalized_value = FieldStatus.MISSING, None
                continue
            check = check_value(field.field_type, field.value)
            field.normalized_value = check.normalized if check.ok else None
            field.status = FieldStatus.CORRECTED if check.ok else FieldStatus.INVALID
            if not check.ok:
                field.messages.append(f"Value {check.message}.")
        return self._finalize(fields, result.source_text)


def _token_key(token: str) -> str:
    return _STRIP_TOKEN.sub("", token).casefold()
