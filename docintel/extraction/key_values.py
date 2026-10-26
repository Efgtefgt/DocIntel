"""Generic "Label: value" pairs, found without a template.

A pair is reported only when the text itself has a short label followed by ":", "=",
":-" or a dot leader ("Name ...... value"). Several pairs on one line are split at runs of
2+ spaces or "|", or at an inner label when OCR merged two columns ("Basic: 40,000 HRA: 16,000"). A label with nothing after it is reported as empty, unless the next line
holds a value that is not itself a label.
"""

from __future__ import annotations

import re
from typing import Optional

from docintel.core.result_models import Entity, KeyValuePair
from docintel.extraction.entities import trailing_label

_PAIR = re.compile(
    r"^[ \t>*•-]*(?P<key>[A-Za-z][A-Za-z0-9 .'/()&#%_-]{0,48}?)[ \t]*(?:\:-|:(?!//|\d)|=|\.{3,})[ \t]*(?P<value>.*)$"
)
_MAX_KEY_WORDS = 6


def _key_ok(key: str) -> bool:
    words = key.split()
    return 0 < len(words) <= _MAX_KEY_WORDS and not key.lower().startswith(("http", "www")) and \
        sum(c.isdigit() for c in key) <= 2


def _segments(line: str) -> list[tuple[int, str]]:
    """Split a line at 2+ spaces or "|" and return (offset, segment) pairs."""
    out, pos = [], 0
    for part in re.split(r"([ \t]{2,}|[ \t]*\|[ \t]*)", line):
        if part and not re.fullmatch(r"[ \t]{2,}|[ \t]*\|[ \t]*", part):
            out.append((pos, part))
        pos += len(part)
    return out


def find_key_values(text: str, *, page_number: Optional[int] = None,
                    entities: Optional[list[Entity]] = None) -> list[KeyValuePair]:
    pairs: list[KeyValuePair] = []
    lines = text.split("\n")
    offset = 0
    offsets = []
    for line in lines:
        offsets.append(offset)
        offset += len(line) + 1

    def is_label_line(line: str) -> bool:
        return any(_PAIR.match(seg) and _key_ok(_PAIR.match(seg).group("key")) for _, seg in _segments(line))

    for idx, line in enumerate(lines):
        segments = _segments(line)
        for seg_no, (seg_pos, seg) in enumerate(segments):
            match = _PAIR.match(seg)
            if not match or not _key_ok(match.group("key")):
                continue
            key = match.group("key").strip(" .-")
            start = offsets[idx] + seg_pos + match.start("value")
            end = start + len(match.group("value").rstrip())
            parts = _split_merged(text, key, start, end)
            for part_no, (part_key, part_start, part_end) in enumerate(parts):
                value = text[part_start:part_end]
                notes: list[str] = ["OCR joined this field with the previous one; split at its label."] if part_no else []
                src_line = line
                if not value:
                    nxt = lines[idx + 1] if idx + 1 < len(lines) else ""
                    if seg_no == len(segments) - 1 and part_no == len(parts) - 1 and nxt.strip() and \
                            not is_label_line(nxt):
                        value = nxt.strip()
                        part_start = offsets[idx + 1] + (len(nxt) - len(nxt.lstrip()))
                        part_end = part_start + len(value)
                        src_line = f"{line}\n{nxt}"
                        notes.append("Value taken from the next line.")
                    else:
                        notes.append("Label has no value after it.")
                pairs.append(_pair(part_key, value, page_number, src_line, part_start, part_end, notes, entities))
    return pairs


_INNER_SEP = re.compile(r"[ \t]*(?::-|:(?!//|\d))[ \t]*")


def _split_merged(text: str, key: str, start: int, end: int) -> list[tuple[str, int, int]]:
    """Split "40,000 HRA: 16,000" (value of "Basic") at the inner label: OCR often turns the gap
    between two columns into a single space. Returns (key, value start, value end) parts."""
    value = text[start:end]
    parts: list[tuple[str, int, int]] = []
    current_key, current = key, 0
    for sep in _INNER_SEP.finditer(value):
        before = value[current: sep.start()]
        label = trailing_label(before, separator=True)
        cut = before.rfind(label) if label else -1
        if cut <= 0 or not before[:cut].strip(" ,;\t"):
            continue
        kept = before[:cut].rstrip(" ,;\t")
        parts.append((current_key, start + current, start + current + len(kept)))
        current_key, current = label, sep.end()
    parts.append((current_key, start + current, end))
    return parts


def _pair(key: str, value: str, page_number: Optional[int], line: str, start: int, end: int, notes: list[str],
          entities: Optional[list[Entity]]) -> KeyValuePair:
    pair = KeyValuePair(key=key, value=value, page_number=page_number, line=line, start=start, end=end,
                        empty=not value, notes=notes)
    if value and entities:
        inside = [e for e in entities if e.start >= start and e.end <= end]
        covered = sum(e.end - e.start for e in inside)
        if len(inside) == 1 and covered >= 0.8 * len(value.rstrip(".,;")):
            pair.value_type = inside[0].type
    return pair
