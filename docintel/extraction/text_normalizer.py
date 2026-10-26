"""Conservative, reviewable text normalization.

Only whitespace and Unicode representation are changed by default. Punctuation, digits
and special characters are never removed. Every rule that changed the text is reported
so the UI can show what happened; the raw OCR text is always kept separately.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizationOptions:
    unicode_form: str = "NFC"              # "NFC" is lossless; "NFKC" also folds ligatures (ﬁ -> fi)
    normalize_line_endings: bool = True
    strip_trailing_whitespace: bool = True
    collapse_spaces: bool = True           # runs of spaces/tabs inside a line -> one space
    max_blank_lines: int = 1               # longer runs of blank lines are reduced
    remove_control_chars: bool = True      # C0/C1 controls except tab and newline
    join_hyphenated_lines: bool = False    # "infor-\nmation" -> "information" (can merge real hyphens)
    straighten_quotes: bool = False        # curly quotes -> ASCII quotes


@dataclass(frozen=True)
class NormalizationResult:
    text: str
    changes: list[str]


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def normalize_text(text: str, options: NormalizationOptions | None = None) -> NormalizationResult:
    options = options or NormalizationOptions()
    changes: list[str] = []

    def apply(label: str, new: str, current: str) -> str:
        if new != current:
            changes.append(label)
        return new

    out = text
    if options.unicode_form:
        out = apply(f"Unicode {options.unicode_form} normalization", unicodedata.normalize(options.unicode_form, out), out)
    if options.normalize_line_endings:
        out = apply("line endings normalized", out.replace("\r\n", "\n").replace("\r", "\n"), out)
    if options.remove_control_chars:
        out = apply("control characters removed", _CONTROL.sub("", out), out)
    if options.collapse_spaces:
        out = apply("repeated spaces collapsed", re.sub(r"[ \t ]{2,}", " ", out), out)
    if options.strip_trailing_whitespace:
        out = apply("trailing whitespace removed", "\n".join(line.rstrip() for line in out.split("\n")), out)
    if options.join_hyphenated_lines:
        out = apply("hyphenated line breaks joined", re.sub(r"(\w)-\n(\w)", r"\1\2", out), out)
    if options.straighten_quotes:
        out = apply("curly quotes straightened", out.translate(_QUOTES), out)
    if options.max_blank_lines >= 0:
        pattern = r"\n{%d,}" % (options.max_blank_lines + 2)
        out = apply("blank lines reduced", re.sub(pattern, "\n" * (options.max_blank_lines + 1), out), out)
    out = apply("leading/trailing blank lines removed", out.strip("\n"), out)
    return NormalizationResult(out, changes)
