"""Measure how well document intelligence survives OCR errors.

For each ``<name>.png`` + ``<name>.gt.txt`` pair the analysis runs twice: on the ground-truth
text and on the OCR output. Values found in the ground truth are the reference, so

* value recall    = reference values also found (same type and normalized value) after OCR
* value precision = values found after OCR that are also in the reference

This isolates the effect of OCR errors. It does not measure whether the detector itself is
right; that is covered by hand-written unit tests. When ``<name>.type.txt`` exists, the
document type found from OCR text is compared with it.

Usage::

    python -m docintel.evaluation.intelligence_benchmark data\\samples\\documents [--preset smart] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from docintel.config.settings import get_settings
from docintel.core.pipeline import DocumentPipeline, PipelineOptions
from docintel.core.result_models import DocumentAnalysis
from docintel.evaluation.benchmark import find_samples
from docintel.extraction.document_analysis import analyze_text


@dataclass
class DocResult:
    sample: str
    expected_type: Optional[str]
    type_from_truth: str
    type_from_ocr: str
    confidence: str
    reference_values: int
    found_values: int
    matched_values: int
    missed: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    error: str = ""


def _keys(analysis: DocumentAnalysis) -> Counter:
    return Counter(f"{e.type}={e.normalized or e.value}" for e in analysis.entities)


def evaluate(dataset: Path, *, engine: str = "tesseract", preset: str = "smart",
             pipeline: Optional[DocumentPipeline] = None) -> list[DocResult]:
    settings = get_settings()
    pipeline = pipeline or DocumentPipeline(settings)
    options = PipelineOptions(engine=engine, preprocess_preset=preset)
    results = []
    for image, gt in find_samples(dataset):
        stem = gt.name[: -len(".gt.txt")]
        type_file = dataset / f"{stem}.type.txt"
        expected = type_file.read_text(encoding="utf-8").strip() if type_file.is_file() else None
        truth = analyze_text(gt.read_text(encoding="utf-8"))
        outcome = pipeline.process(image.read_bytes(), image.name, options)
        if outcome.analysis is None:
            message = "; ".join(e.message for e in outcome.errors) or "no text"
            results.append(DocResult(image.name, expected, truth.classification.document_type, "-", "none",
                                     sum(_keys(truth).values()), 0, 0, error=message))
            continue
        ref, got = _keys(truth), _keys(outcome.analysis)
        matched = ref & got
        results.append(DocResult(
            image.name, expected, truth.classification.document_type, outcome.analysis.classification.document_type,
            outcome.analysis.classification.confidence, sum(ref.values()), sum(got.values()), sum(matched.values()),
            missed=sorted((ref - got).elements()), extra=sorted((got - ref).elements()),
        ))
    return results


def summarize(results: list[DocResult]) -> dict:
    typed = [r for r in results if r.expected_type]
    ref = sum(r.reference_values for r in results)
    found = sum(r.found_values for r in results)
    matched = sum(r.matched_values for r in results)
    return {
        "documents": len(results),
        "type_accuracy_ocr": round(statistics.mean(r.type_from_ocr == r.expected_type for r in typed), 3) if typed else None,
        "type_accuracy_truth": round(statistics.mean(r.type_from_truth == r.expected_type for r in typed), 3) if typed else None,
        "value_recall": round(matched / ref, 3) if ref else None,
        "value_precision": round(matched / found, 3) if found else None,
        "reference_values": ref,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure document-type and value detection after OCR.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--engine", default="tesseract")
    parser.add_argument("--preset", default="smart")
    parser.add_argument("--out", type=Path, help="write report.json here")
    args = parser.parse_args(argv)
    results = evaluate(args.dataset, engine=args.engine, preset=args.preset)
    if not results:
        print(f"No <name>.png + <name>.gt.txt pairs found in {args.dataset}", file=sys.stderr)
        return 1
    for r in results:
        mark = "" if r.expected_type is None else ("ok " if r.type_from_ocr == r.expected_type else "XX ")
        print(f"{mark}{r.sample:28} type={r.type_from_ocr} ({r.confidence}) values {r.matched_values}/"
              f"{r.reference_values} found, {r.found_values - r.matched_values} extra {r.error}")
        for key in r.missed:
            print(f"      missed: {key}")
        for key in r.extra:
            print(f"      extra:  {key}")
    summary = summarize(results)
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.json").write_text(
            json.dumps({"summary": summary, "documents": [asdict(r) for r in results]}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
