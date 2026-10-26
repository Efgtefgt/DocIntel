"""Unified document pipeline.

validate -> detect type -> (image frames | PDF pages) -> embedded text or
preprocess + OCR per page -> normalize -> optional tables -> document analysis (type,
entities, key-value pairs) -> optional structured extraction -> validation. Each stage is a separate service; this module only wires
them together and records per-stage status, timing and errors.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Sequence

from PIL import Image

from docintel.config.settings import Settings
from docintel.core.deadline import Deadline
from docintel.core.exceptions import DocIntelError, InputValidationError, OCREngineUnavailableError, ProcessingTimeoutError
from docintel.core.file_validator import ValidatedFile, validate_upload
from docintel.core.result_models import (
    DocumentAnalysis,
    DocumentResult,
    FileKind,
    OCRResult,
    PageResult,
    PageSource,
    OCRAttempt,
    PreprocessStepRecord,
    QualityFinding,
    QualityReport,
    StageError,
    StageRecord,
    StageStatus,
    StructuredResult,
    TableResult,
)
from docintel.documents.document_processor import PDFMode, parse_page_range, process_pdf
from docintel.extraction.document_analysis import analyze_document
from docintel.extraction.structured_extractor import StructuredExtractor
from docintel.extraction.table_extractor import (
    TableMethod,
    extract_pdf_tables,
    extract_tables_from_ocr,
    remove_ruling_lines,
)
from docintel.extraction.templates import ExtractionTemplate, TemplateStore
from docintel.extraction.text_normalizer import NormalizationOptions, normalize_text
from docintel.ocr.base_engine import OCROptions
from docintel.ocr import quality
from docintel.ocr.engine_factory import EngineFactory
from docintel.preprocessing.analysis import alternative_plans, analysis_metrics, analyze, smart_plan
from docintel.preprocessing.image_loader import iter_frames
from docintel.preprocessing.preprocessing_pipeline import PreprocessConfig, run_pipeline

logger = logging.getLogger(__name__)

# (fraction complete 0..1, short status message)
Progress = Callable[[float, str], None]


@dataclass(frozen=True)
class PipelineOptions:
    engine: str = "tesseract"
    ocr: OCROptions = field(default_factory=OCROptions)
    preprocess_preset: str = "smart"
    custom_steps: Optional[tuple[str, ...]] = None  # overrides the preset when set
    pdf_mode: PDFMode = PDFMode.AUTO
    page_range: str = ""
    normalization: NormalizationOptions = field(default_factory=NormalizationOptions)
    extract_tables: bool = False
    table_method: TableMethod = TableMethod.AUTO
    template: Optional[ExtractionTemplate] = None
    analyze: bool = True  # document type, entities and key-value pairs (rule-based, fast)


@dataclass
class PipelineResult:
    filename: str
    document: Optional[DocumentResult] = None
    tables: list[TableResult] = field(default_factory=list)
    structured: Optional[StructuredResult] = None
    analysis: Optional[DocumentAnalysis] = None
    stages: list[StageRecord] = field(default_factory=list)
    errors: list[StageError] = field(default_factory=list)
    duration_seconds: float = 0.0
    rss_mb: Optional[float] = None

    @property
    def succeeded(self) -> bool:
        return self.document is not None and self.document.succeeded


def _rss_mb() -> Optional[float]:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 1_048_576, 1)
    except (ImportError, OSError):
        return None


class DocumentPipeline:
    def __init__(self, settings: Settings, engines: Optional[EngineFactory] = None) -> None:
        self.settings = settings
        self.engines = engines or EngineFactory(settings)

    # --- stage helpers ---------------------------------------------------------------
    @staticmethod
    def _record(result: PipelineResult, stage: str, status: StageStatus, started: float, detail: str = "") -> None:
        result.stages.append(StageRecord(stage=stage, status=status, duration_seconds=round(time.perf_counter() - started, 3), detail=detail))

    @staticmethod
    def _fail(result: PipelineResult, stage: str, exc: DocIntelError, started: float) -> None:
        result.errors.append(StageError(stage=stage, code=exc.code, message=exc.user_message))
        DocumentPipeline._record(result, stage, StageStatus.FAILED, started, exc.code)

    def preprocess_config(self, options: PipelineOptions) -> PreprocessConfig:
        if options.custom_steps is not None:
            return PreprocessConfig(steps=tuple(options.custom_steps), max_pixels=self.settings.max_image_pixels)
        return PreprocessConfig.from_preset(options.preprocess_preset, max_pixels=self.settings.max_image_pixels)

    def _orientation_detector(self):
        """Tesseract OSD is used for orientation regardless of the recognition engine."""
        try:
            tesseract = self.engines.get("tesseract")
        except OCREngineUnavailableError:
            return None
        return tesseract.detect_orientation

    # --- main entry ------------------------------------------------------------------
    def process(self, data: bytes, filename: str, options: PipelineOptions, progress: Optional[Progress] = None) -> PipelineResult:
        started_all = time.perf_counter()
        result = PipelineResult(filename=filename)
        report = progress or (lambda fraction, message: None)
        deadline = Deadline(self.settings.document_timeout_seconds)

        # 1-2. Validation and file type detection
        started = time.perf_counter()
        report(0.0, "Validating file")
        try:
            validated = validate_upload(data, filename, self.settings)
        except InputValidationError as exc:
            self._fail(result, "validation", exc, started)
            result.duration_seconds = round(time.perf_counter() - started_all, 3)
            logger.info("File rejected", extra={"code": exc.code})
            return result
        result.filename = validated.info.filename
        self._record(result, "validation", StageStatus.SUCCESS, started)
        self._record(result, "file_type_detection", StageStatus.SUCCESS, started, validated.info.format)

        document = DocumentResult(file_info=validated.info)
        result.document = document

        # 3-7. Page processing (embedded text and/or preprocessing + OCR)
        started = time.perf_counter()
        ocr_tables: list[TableResult] = []
        quality_reports: dict[int, QualityReport] = {}
        try:
            page_numbers = parse_page_range(options.page_range, validated.info.page_count)
            ocr_page = self._make_page_ocr(options, deadline, ocr_tables, quality_reports)
            if validated.info.kind is FileKind.PDF:
                document.pages = process_pdf(
                    validated.data,
                    page_numbers=page_numbers,
                    mode=options.pdf_mode,
                    ocr_page=ocr_page,
                    render_dpi=self.settings.pdf_render_dpi,
                    max_pixels=self.settings.max_image_pixels,
                    deadline=deadline,
                    progress=lambda done, total, page, status: report(0.05 + 0.8 * done / max(total, 1), f"Page {page} of {validated.info.page_count}: {status}"),
                )
            else:
                document.pages = self._process_image(validated, page_numbers, ocr_page, deadline, report)
        except DocIntelError as exc:
            self._fail(result, "page_processing", exc, started)
            document.errors.append(result.errors[-1])
        else:
            failed = len(document.failed_pages)
            status = StageStatus.FAILED if failed == len(document.pages) else StageStatus.SUCCESS
            self._record(result, "page_processing", status, started, f"{len(document.pages)} page(s), {failed} failed")
            for page in document.failed_pages:
                if page.error:
                    result.errors.append(page.error)

        for page in document.pages:
            if page.source is PageSource.OCR:
                page.quality = quality_reports.get(page.page_number)

        # 8. Normalization (raw text is preserved on each page)
        started = time.perf_counter()
        report(0.88, "Normalizing text")
        for page in document.pages:
            normalized = normalize_text(page.raw_text, options.normalization)
            page.normalized_text, page.normalization_changes = normalized.text, normalized.changes
        self._record(result, "normalization", StageStatus.SUCCESS, started)

        # 9. Optional table extraction
        started = time.perf_counter()
        if options.extract_tables and document.pages:
            report(0.9, "Extracting tables")
            try:
                embedded_pages = [p.page_number for p in document.pages if p.source is PageSource.EMBEDDED_TEXT]
                tables = list(ocr_tables)
                if embedded_pages and validated.info.kind is FileKind.PDF:
                    tables += extract_pdf_tables(validated.data, embedded_pages, options.table_method, self.settings)
                tables.sort(key=lambda t: t.page_number)
                result.tables = [t.model_copy(update={"table_index": i + 1}) for i, t in enumerate(tables)]
                self._record(result, "tables", StageStatus.SUCCESS, started, f"{len(result.tables)} table(s)")
            except DocIntelError as exc:
                self._fail(result, "tables", exc, started)
        else:
            self._record(result, "tables", StageStatus.SKIPPED, started)

        # 10. Document intelligence: type, entities, key-value pairs
        started = time.perf_counter()
        if options.analyze and any(p.succeeded for p in document.pages):
            report(0.93, "Analysing document")
            try:
                result.analysis = analyze_document(
                    document, template_types=TemplateStore(self.settings.templates_dir).document_types())
                c = result.analysis.classification
                self._record(result, "analysis", StageStatus.SUCCESS, started,
                             f"{c.label} ({c.confidence}), {len(result.analysis.entities)} entities")
            except Exception:  # analysis is an extra; the recognised text is still returned
                logger.exception("Document analysis failed")
                result.errors.append(StageError(stage="analysis", code="analysis_failed",
                                                message="Document analysis failed; the recognised text is unaffected."))
                self._record(result, "analysis", StageStatus.FAILED, started, "analysis_failed")
        else:
            self._record(result, "analysis", StageStatus.SKIPPED, started)

        # 11. Optional structured extraction + validation
        started = time.perf_counter()
        if options.template is not None and document.pages:
            report(0.95, "Extracting fields")
            text = "\n\n".join(p.normalized_text for p in document.pages if p.succeeded)
            low_conf = [w.text for p in document.pages if p.ocr for w in p.ocr.low_confidence_words]
            result.structured = StructuredExtractor(options.template).extract(text, low_conf)
            self._record(result, "structured_extraction", StageStatus.SUCCESS, started, "valid" if result.structured.valid else "needs review")
        else:
            self._record(result, "structured_extraction", StageStatus.SKIPPED, started)

        document.stages = result.stages
        result.duration_seconds = document.total_duration_seconds = round(time.perf_counter() - started_all, 3)
        result.rss_mb = _rss_mb()
        report(1.0, "Done")
        logger.info(
            "Document processed",
            extra={
                "format": validated.info.format,
                "pages": len(document.pages),
                "failed_pages": len(document.failed_pages),
                "duration_s": result.duration_seconds,
                "rss_mb": result.rss_mb,
            },
        )
        return result

    def _make_page_ocr(self, options: PipelineOptions, deadline: Deadline, table_sink: list[TableResult],
                       quality_sink: dict[int, QualityReport]):
        smart = options.custom_steps is None and options.preprocess_preset == "smart"
        config = None if smart else self.preprocess_config(options)
        needs_detector = smart or "orientation" in config.steps  # type: ignore[union-attr]
        detector = self._orientation_detector() if needs_detector else None
        max_attempts = self.settings.max_ocr_attempts if smart else 1

        def run_once(engine, image: Image.Image, cfg: PreprocessConfig, det, page_number: int):
            prepared = run_pipeline(image, cfg, orientation_detector=det)
            ocr_input, steps = prepared.image, list(prepared.steps)
            if options.extract_tables:
                # Ruling lines confuse OCR (read as '|', '_'); OCR a copy without them. The
                # geometry is unchanged, so grid cells can still be taken from prepared.image.
                ocr_input, removed = remove_ruling_lines(prepared.image)
                steps.append(PreprocessStepRecord(step="remove_table_lines", applied=removed,
                                                  detail="grid lines hidden for OCR" if removed else "no ruled table found"))
            ocr_options = replace(options.ocr, timeout_seconds=deadline.cap(options.ocr.timeout_seconds))
            started = time.perf_counter()
            result = engine.recognize(ocr_input, ocr_options)
            return prepared, steps, result, time.perf_counter() - started

        def ocr_page(image: Image.Image, page_number: int) -> tuple[OCRResult, list[PreprocessStepRecord]]:
            deadline.check(f"OCR of page {page_number}")
            engine = self.engines.get(options.engine)
            expected = quality.estimate_glyph_count(image)
            report = QualityReport(estimated_characters=expected)
            if smart:
                analysis = analyze(image, detector)
                plan, reasons = smart_plan(analysis)
                report.metrics = analysis_metrics(analysis)
                report.findings = [QualityFinding(code=f.code, severity=f.severity, message=f.message, step=f.step)
                                   for f in analysis.findings]
                report.plan = reasons
                # Orientation was already measured during analysis; reuse it instead of a second OSD call.
                det = (lambda _img, r=analysis.rotation: r) if analysis.rotation else None
                candidates = [("smart plan", plan)] + alternative_plans(plan)
                cfg_kwargs = {"max_pixels": self.settings.max_image_pixels}
            else:
                det = detector
                candidates = [(options.preprocess_preset if options.custom_steps is None else "custom", config.steps)]
                cfg_kwargs = {}
            best = None
            for index, (label, steps) in enumerate(candidates[:max_attempts]):
                if index > 0:
                    if deadline.remaining() < 2:
                        report.notes.append("Further attempts skipped: document time limit nearly reached.")
                        break
                cfg = PreprocessConfig(steps=tuple(steps), **cfg_kwargs) if smart else config
                try:
                    prepared, rec_steps, result, seconds = run_once(engine, image, cfg, det, page_number)
                except ProcessingTimeoutError:
                    if best is None:
                        raise
                    report.notes.append(f"Attempt '{label}' timed out and was skipped.")
                    break
                attempt_score = quality.score(result, expected)
                report.attempts.append(OCRAttempt(label=label, steps=list(cfg.steps), mean_confidence=result.mean_confidence,
                                                  coverage=quality.coverage(result, expected), score=attempt_score,
                                                  seconds=round(seconds, 3)))
                if best is None or attempt_score > best[0]:
                    best = (attempt_score, index, prepared, rec_steps, result)
                if quality.verdict(result, expected)[0] == "good":
                    break  # good enough: don't spend CPU on alternatives
            _, best_index, prepared, rec_steps, result = best  # type: ignore[misc]
            report.attempts[best_index].chosen = True
            report.recognised_characters = quality.recognised_characters(result)
            report.coverage = quality.coverage(result, expected)
            report.verdict, notes = quality.verdict(result, expected)
            report.notes = notes + report.notes
            if len(report.attempts) > 1:
                report.notes.append(f"Tried {len(report.attempts)} preprocessing plans; kept '{report.attempts[best_index].label}'.")
            quality_sink[page_number] = report
            if options.extract_tables:
                table_sink.extend(extract_tables_from_ocr(result, prepared.image, page_number))
            return result, rec_steps

        return ocr_page

    def _process_image(self, validated: ValidatedFile, page_numbers: list[int], ocr_page, deadline: Deadline, report: Progress) -> list[PageResult]:
        pages: list[PageResult] = []
        total = validated.info.page_count
        frames = iter_frames(validated.data, page_numbers, self.settings.max_tiff_frames)
        for index, (number, frame) in enumerate(frames):
            report(0.05 + 0.8 * index / len(page_numbers), f"Image {number} of {total}: OCR")
            page = PageResult(page_number=number, width=frame.width, height=frame.height)
            started = time.perf_counter()
            try:
                ocr_result, steps = ocr_page(frame, number)
                page.source, page.raw_text, page.ocr, page.preprocessing = PageSource.OCR, ocr_result.text, ocr_result, steps
            except DocIntelError as exc:
                page.error = StageError(stage="ocr", code=exc.code, message=exc.user_message, page_number=number)
            page.duration_seconds = round(time.perf_counter() - started, 3)
            pages.append(page)
        return pages

    # --- batches ---------------------------------------------------------------------
    def process_batch(
        self,
        files: Sequence[tuple[bytes, str]],
        options: PipelineOptions,
        on_file_done: Optional[Callable[[int, PipelineResult], None]] = None,
    ) -> list[PipelineResult]:
        """Process several files: sequentially by default, or with bounded concurrency
        when ``settings.max_workers`` > 1 (each worker runs one Tesseract process)."""
        if self.settings.max_workers <= 1 or len(files) <= 1:
            results = []
            for index, (data, name) in enumerate(files):
                results.append(self.process(data, name, options))
                if on_file_done:
                    on_file_done(index, results[-1])
            return results
        with ThreadPoolExecutor(max_workers=self.settings.max_workers, thread_name_prefix="docintel") as pool:
            futures = [pool.submit(self.process, data, name, options) for data, name in files]
            results = []
            for index, future in enumerate(futures):
                results.append(future.result())
                if on_file_done:
                    on_file_done(index, results[-1])
            return results
