"""Exception hierarchy.

Every exception carries a stable machine-readable ``code`` and a ``user_message`` that is
safe to show in the UI or API. Messages must never include document contents.
"""

from __future__ import annotations


class DocIntelError(Exception):
    """Base class for all expected application errors."""

    code = "docintel_error"

    def __init__(self, user_message: str, *, code: str | None = None) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        if code:
            self.code = code


# --- Input validation ------------------------------------------------------------------
class InputValidationError(DocIntelError):
    code = "invalid_input"


class EmptyFileError(InputValidationError):
    code = "empty_file"


class FileTooLargeError(InputValidationError):
    code = "file_too_large"


class UnsupportedFormatError(InputValidationError):
    code = "unsupported_format"


class CorruptFileError(InputValidationError):
    code = "corrupt_file"


class ImageDimensionError(InputValidationError):
    code = "invalid_image_dimensions"


class PageLimitError(InputValidationError):
    code = "page_limit_exceeded"


class EncryptedDocumentError(InputValidationError):
    code = "encrypted_document"


class PageRangeError(InputValidationError):
    code = "invalid_page_range"


# --- OCR ---------------------------------------------------------------------------------
class OCREngineUnavailableError(DocIntelError):
    code = "ocr_engine_unavailable"


class LanguageDataMissingError(DocIntelError):
    code = "language_data_missing"


class OCRProcessingError(DocIntelError):
    code = "ocr_failed"


class ProcessingTimeoutError(DocIntelError):
    code = "processing_timeout"


# --- Documents, extraction and export --------------------------------------------------
class PDFProcessingError(DocIntelError):
    code = "pdf_processing_failed"


class TableExtractionError(DocIntelError):
    code = "table_extraction_failed"


class TemplateError(DocIntelError):
    code = "invalid_template"


class ExportError(DocIntelError):
    code = "export_failed"


class OptionalDependencyMissingError(DocIntelError):
    code = "optional_dependency_missing"


class UnsafePathError(DocIntelError):
    code = "unsafe_path"
