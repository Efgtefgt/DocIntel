"""Document intelligence: type classification, entities and key-value pairs for a document.

Everything here is deterministic (regular expressions, checksums and keyword rules). It runs
in milliseconds and never sends text anywhere. No language model is used.
"""

from __future__ import annotations

import time
from typing import Iterable, Optional, Sequence

from docintel.core.result_models import DocumentAnalysis, DocumentResult
from docintel.extraction.classifier import classify
from docintel.extraction.entities import SENSITIVE_TYPES, find_entities, mask
from docintel.extraction.key_values import find_key_values


def analyze_pages(pages: Sequence[tuple[Optional[int], str]], *, low_confidence_words: Iterable[str] = (),
                  template_types: Optional[dict[str, str]] = None) -> DocumentAnalysis:
    """Analyse ``(page_number, text)`` pairs. Entity offsets refer to each page's own text."""
    started = time.perf_counter()
    low = list(low_confidence_words)
    entities, pairs = [], []
    for page_number, text in pages:
        page_entities = find_entities(text, page_number=page_number, low_confidence_words=low)
        entities += page_entities
        pairs += find_key_values(text, page_number=page_number, entities=page_entities)
    full_text = "\n\n".join(text for _, text in pages)
    classification = classify(full_text, entities, pairs, template_types)
    return DocumentAnalysis(classification=classification, entities=entities, key_values=pairs,
                            duration_seconds=round(time.perf_counter() - started, 4))


def analyze_text(text: str, **kwargs) -> DocumentAnalysis:
    return analyze_pages([(None, text)], **kwargs)


def analyze_document(document: DocumentResult, *, page_texts: Optional[dict[int, str]] = None,
                     template_types: Optional[dict[str, str]] = None) -> DocumentAnalysis:
    """Analyse a processed document. ``page_texts`` overrides page text (e.g. the user's edits)."""
    overrides = page_texts or {}
    pages = [(p.page_number, overrides.get(p.page_number, p.normalized_text)) for p in document.pages if p.succeeded]
    # Words the user retyped are no longer OCR output, so only unedited pages pass low-confidence words on.
    low = [w.text for p in document.pages if p.ocr and p.page_number not in overrides
           for w in p.ocr.low_confidence_words]
    return analyze_pages(pages, low_confidence_words=low, template_types=template_types)


def mask_sensitive(analysis: DocumentAnalysis) -> DocumentAnalysis:
    """Copy of ``analysis`` with identifiers (PAN, Aadhaar, card, IBAN, account number) masked."""
    masked = analysis.model_copy(deep=True)
    secrets = []
    for entity in masked.entities:
        if entity.sensitive:
            secrets.append((entity.value, mask(entity.value)))
            entity.value, entity.normalized = mask(entity.value), mask(entity.normalized) if entity.normalized else None
    for pair in masked.key_values:
        if pair.value_type in SENSITIVE_TYPES:
            secrets.append((pair.value, mask(pair.value)))
            pair.value = mask(pair.value)
    for secret, hidden in secrets:
        for item in [*masked.entities, *masked.key_values]:
            item.line = item.line.replace(secret, hidden)
        for pair in masked.key_values:
            pair.value = pair.value.replace(secret, hidden)
    return masked
