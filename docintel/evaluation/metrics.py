"""Character and word error rates based on Levenshtein edit distance.

CER = (S + D + I) / N over characters; WER is the same over whitespace-separated words,
where N is the length of the reference. Values can exceed 1.0 when the hypothesis has
many insertions. Scores depend on the normalization applied first; see
:class:`EvalNormalization` and docs/EVALUATION.md.
"""

from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass
from typing import Hashable, Sequence


@dataclass(frozen=True)
class EvalNormalization:
    """Normalization applied to both reference and hypothesis before scoring.

    Defaults: Unicode NFC, whitespace runs (including newlines) collapsed to one space,
    leading/trailing whitespace removed. Case and punctuation are preserved.
    """

    unicode_nfc: bool = True
    collapse_whitespace: bool = True
    lowercase: bool = False
    remove_punctuation: bool = False

    def apply(self, text: str) -> str:
        if self.unicode_nfc:
            text = unicodedata.normalize("NFC", text)
        if self.lowercase:
            text = text.casefold()
        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))
        if self.collapse_whitespace:
            text = re.sub(r"\s+", " ", text)
        return text.strip()


def edit_distance(reference: Sequence[Hashable], hypothesis: Sequence[Hashable]) -> int:
    """Levenshtein distance with unit costs, O(len(ref) * len(hyp)) time, O(len(hyp)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_item != hyp_item)))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str, normalization: EvalNormalization | None = None) -> float:
    norm = normalization or EvalNormalization()
    ref, hyp = norm.apply(reference), norm.apply(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str, normalization: EvalNormalization | None = None) -> float:
    norm = normalization or EvalNormalization()
    ref, hyp = norm.apply(reference).split(), norm.apply(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)
