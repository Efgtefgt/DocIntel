"""Benchmark runner: OCR every sample in a dataset and compare with ground truth.

Dataset layout: ``<name>.<png|jpg|tif|pdf...>`` next to ``<name>.gt.txt`` (UTF-8,
manually verified reference text). Usage::

    python -m docintel.evaluation.benchmark data/samples --preset auto --out outputs/benchmark
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from docintel.config.logging_config import configure_logging
from docintel.config.settings import Settings, get_settings
from docintel.core.file_detector import SUPPORTED_EXTENSIONS
from docintel.core.pipeline import DocumentPipeline, PipelineOptions
from docintel.evaluation.metrics import EvalNormalization, cer, wer
from docintel.ocr.base_engine import OCROptions


@dataclass
class SampleResult:
    sample: str
    success: bool
    cer: Optional[float]
    wer: Optional[float]
    duration_seconds: float
    mean_confidence: Optional[float]
    error: str = ""


@dataclass
class BenchmarkReport:
    engine: str
    preset: str
    normalization: dict
    samples: list[SampleResult]

    @property
    def summary(self) -> dict:
        ok = [s for s in self.samples if s.success and s.cer is not None]
        return {
            "samples": len(self.samples),
            "succeeded": len(ok),
            "failed": len(self.samples) - len(ok),
            "mean_cer": round(statistics.mean(s.cer for s in ok), 4) if ok else None,
            "mean_wer": round(statistics.mean(s.wer for s in ok), 4) if ok else None,  # type: ignore[misc]
            "total_duration_seconds": round(sum(s.duration_seconds for s in self.samples), 3),
        }

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "preset": self.preset,
            "normalization": self.normalization,
            "summary": self.summary,
            "samples": [asdict(s) for s in self.samples],
        }


def find_samples(dataset: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for gt in sorted(dataset.glob("*.gt.txt")):
        stem = gt.name[: -len(".gt.txt")]
        for ext in SUPPORTED_EXTENSIONS:
            candidate = dataset / f"{stem}.{ext}"
            if candidate.is_file():
                pairs.append((candidate, gt))
                break
    return pairs


def run_benchmark(
    dataset: Path,
    settings: Settings,
    *,
    engine: str = "tesseract",
    preset: str = "smart",
    languages: tuple[str, ...] = ("eng",),
    psm: int = 3,
    normalization: EvalNormalization | None = None,
    pipeline: DocumentPipeline | None = None,
) -> BenchmarkReport:
    norm = normalization or EvalNormalization()
    pipeline = pipeline or DocumentPipeline(settings)
    options = PipelineOptions(
        engine=engine,
        preprocess_preset=preset,
        ocr=OCROptions(languages=languages, psm=psm, timeout_seconds=settings.ocr_timeout_seconds),
    )
    results = []
    for sample, gt in find_samples(dataset):
        started = time.perf_counter()
        outcome = pipeline.process(sample.read_bytes(), sample.name, options)
        duration = time.perf_counter() - started
        reference = gt.read_text(encoding="utf-8")
        if outcome.document is None or not outcome.document.succeeded:
            message = "; ".join(e.message for e in outcome.errors) or "processing failed"
            results.append(SampleResult(sample.name, False, None, None, round(duration, 3), None, message))
            continue
        hypothesis = "\n".join(p.raw_text for p in outcome.document.pages)
        confidences = [p.ocr.mean_confidence for p in outcome.document.pages if p.ocr and p.ocr.mean_confidence is not None]
        results.append(
            SampleResult(
                sample=sample.name,
                success=True,
                cer=round(cer(reference, hypothesis, norm), 4),
                wer=round(wer(reference, hypothesis, norm), 4),
                duration_seconds=round(duration, 3),
                mean_confidence=round(statistics.mean(confidences), 2) if confidences else None,
            )
        )
    return BenchmarkReport(engine=engine, preset=preset, normalization=asdict(norm), samples=results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure OCR CER/WER against ground-truth text.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--engine", default="tesseract")
    parser.add_argument("--preset", default="smart")
    parser.add_argument("--lang", default="eng", help="e.g. eng or eng+hin")
    parser.add_argument("--psm", type=int, default=3)
    parser.add_argument("--lowercase", action="store_true", help="ignore case when scoring")
    parser.add_argument("--ignore-punctuation", action="store_true")
    parser.add_argument("--out", type=Path, help="directory for report.json and report.csv")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.model_copy(update={"log_level": "WARNING"}))
    if not args.dataset.is_dir():
        print(f"Dataset directory not found: {args.dataset}", file=sys.stderr)
        return 2
    report = run_benchmark(
        args.dataset,
        settings,
        engine=args.engine,
        preset=args.preset,
        languages=tuple(args.lang.split("+")),
        psm=args.psm,
        normalization=EvalNormalization(lowercase=args.lowercase, remove_punctuation=args.ignore_punctuation),
    )
    if not report.samples:
        print("No samples found (expected <name>.png next to <name>.gt.txt).", file=sys.stderr)
        return 2
    print(f"{'sample':<28}{'ok':<5}{'CER':>8}{'WER':>8}{'conf':>8}{'sec':>8}")
    for s in report.samples:
        fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else "-"  # noqa: E731
        print(f"{s.sample:<28}{'yes' if s.success else 'NO':<5}{fmt(s.cer):>8}{fmt(s.wer):>8}{fmt(s.mean_confidence):>8}{s.duration_seconds:>8.2f}")
        if s.error:
            print(f"    error: {s.error}")
    print(json.dumps(report.summary, indent=2))
    if args.out:
        import pandas as pd

        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        pd.DataFrame([asdict(s) for s in report.samples]).to_csv(args.out / "report.csv", index=False, encoding="utf-8")
        print(f"Reports written to {args.out}")
    return 0 if report.summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
