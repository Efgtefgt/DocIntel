"""Find typed values in recognised text without a template.

Detected types: e-mail, URL, phone, date, time, amount, percentage, number, reference/ID
code, IP address, PIN code, Indian vehicle registration, and identifiers with a format or
checksum (PAN, GSTIN, IFSC, Aadhaar, payment card, IBAN, bank account number).

Rules:
* Every entity is a verbatim span of the text. Nothing is completed, corrected or guessed.
* ``valid`` needs the format plus a checksum or a matching label; a pattern match alone is
  ``uncertain``; a value that has the shape of a type but fails its checksum or calendar
  check is ``invalid`` (often an OCR error worth reviewing).
* Each character belongs to at most one entity. More specific types win (an e-mail is not
  also reported as a URL, a date is not also a number).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, Optional

from docintel.core.result_models import Entity, EntityStatus
from docintel.extraction.field_validators import DATE_RE, EMAIL_RE, check_date

ENTITY_TYPES = {
    "email": "E-mail",
    "url": "Web address",
    "phone": "Phone number",
    "date": "Date",
    "time": "Time",
    "amount": "Amount",
    "percentage": "Percentage",
    "number": "Number",
    "reference": "Reference / ID",
    "ip_address": "IP address",
    "pin_code": "PIN / ZIP code",
    "vehicle_number": "Vehicle registration",
    "pan": "PAN",
    "gstin": "GSTIN",
    "ifsc": "IFSC code",
    "aadhaar": "Aadhaar number",
    "card_number": "Payment card number",
    "iban": "IBAN",
    "account_number": "Bank account number",
}
SENSITIVE_TYPES = {"pan", "aadhaar", "card_number", "iban", "account_number"}
# Everyday groups of entity types (Key details shortcuts, batch summary columns), in display order.
ENTITY_GROUPS: dict[str, tuple[str, ...]] = {
    "Phone": ("phone",),
    "E-mail": ("email",),
    "PIN / ZIP code": ("pin_code",),
    "Website": ("url",),
    "Date": ("date", "time"),
    "Amount": ("amount", "percentage"),
    "ID / reference": ("reference", "pan", "gstin", "ifsc", "aadhaar", "vehicle_number", "ip_address"),
    "Bank / card": ("account_number", "card_number", "iban"),
}

# --- label handling ---------------------------------------------------------------------
_SEGMENT_SPLIT = re.compile(r"[ \t]{2,}|[ \t]*\|[ \t]*|(?<=[.;,])[ \t]+(?=[A-Za-z])")
_LABEL_TAIL = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z0-9 .'/()&%_-]*?)[ \t]*(?P<sep>:-|[:=#–—]|\.{2,}|-(?=[ \t]))?[ \t]*$"
)


def _kw(*words: str) -> re.Pattern[str]:
    return re.compile(r"(?<![a-z])(?:" + "|".join(words) + r")(?![a-z])", re.IGNORECASE)


PHONE_WORDS = _kw("phone", "mobile", "mob", "tel", "telephone", "contact", "ph", "cell", "fax", "whatsapp",
                  "helpline", "call", "landline")
AMOUNT_WORDS = _kw("total", "amount", "amt", "price", "sub ?total", "tax", "gst", "cgst", "sgst", "igst", "vat",
                   "balance", "due", "paid", "payable", "fee", "fees", "charges?", "cost", "rate", "net", "gross",
                   "salary", "discount", "pay", "earnings", "deductions?", "basic", "hra", "refund", "fine",
                   "deposit", "withdrawal", "debit", "credit", "provident fund", "pf", "esi", "allowance",
                   "bonus", "income", "wages", "tds", "loan", "emi", "premium", "rent")
ID_WORDS = _kw("no", "no\\.", "number", "num", "id", "ref", "reference", "code", "invoice", "order", "ticket", "case",
               "receipt", "bill", "txn", "transaction", "utr", "po", "fir", "complaint", "policy", "registration",
               "serial", "s/n", "folio", "challan", "docket", "#")
ACCOUNT_WORDS = _kw("account", "a/c", "acct", "acc")
PIN_WORDS = _kw("pin", "pincode", "pin code", "postal code", "postcode", "zip", "zip code")
CARD_WORDS = _kw("card", "credit", "debit", "visa", "mastercard", "rupay", "amex")
VEHICLE_WORDS = _kw("vehicle", "veh", "reg", "registration", "regn", "car", "bike", "truck")


_EXTRA_LABELS = (
    "employee id", "employee name", "phone number", "phone no", "mobile no", "contact no", "account number",
    "account no", "bank a/c no", "a/c no", "invoice no", "invoice date", "date of birth", "bill date", "order no",
    "ref no", "reference no", "vehicle no", "challan no", "consumer no", "pin code", "postal code", "total amount",
    "bill amount", "payment mode", "incident type", "caller name", "blood group", "father's name", "employee code",
)
_LABEL_TOKEN = re.compile(r"[A-Z][A-Za-z.'/()&%_-]*|[A-Z/]{2,}\.?")
_EVIDENCE_WORDS = (PHONE_WORDS, AMOUNT_WORDS, ID_WORDS, ACCOUNT_WORDS, PIN_WORDS, CARD_WORDS, VEHICLE_WORDS)


@lru_cache(maxsize=1)
def known_labels() -> frozenset[str]:
    """Common label phrases (lower case): extras plus the classifier's keywords."""
    from docintel.extraction.classifier import DOCUMENT_TYPES

    keywords = {k for t in DOCUMENT_TYPES.values() for k in t.keywords}
    return frozenset({*_EXTRA_LABELS, *keywords})


def trailing_label(text: str, *, separator: bool) -> Optional[str]:
    """Pick the label at the end of ``text`` when OCR merged it with the previous value.

    "Basic: 40,000 HRA" -> "HRA"; "Anita Rao Employee ID" -> "Employee ID". The run of
    capitalised words is cut to the longest known label phrase; otherwise the last word is
    used, and without a separator only when it is a typical label word (Qty, Rate, Phone...).
    """
    run: list[str] = []
    for token in reversed(text.split()):
        if len(run) == 4 or not _LABEL_TOKEN.fullmatch(token):
            break
        run.insert(0, token)
    if not run:
        return None
    known = known_labels()
    for n in range(len(run), 0, -1):
        phrase = " ".join(run[-n:])
        if phrase.lower().rstrip(".") in known:
            return phrase.rstrip(".")
    last = run[-1].rstrip(".")
    if separator or any(words.search(last) for words in _EVIDENCE_WORDS):
        return last
    return None


def label_before(line_prefix: str) -> Optional[str]:
    """The label written just before a value on the same line, e.g. "Invoice No" in "Invoice No: 12".

    A label must be short. Without a separator (":", "-", "#", "=") only 1-3 words count, so
    ordinary sentences ("Contact the help desk on ...") do not become labels.
    """
    stripped = line_prefix.rstrip(" \t")
    column_gap = len(line_prefix) - len(stripped) >= 2  # "Withdrawal   2,000.00": a gap works like ":"
    segment = _SEGMENT_SPLIT.split(stripped)[-1] if stripped.strip() else ""
    if not segment.strip():
        return None
    match = _LABEL_TAIL.fullmatch(segment.lstrip(" \t>*•-"))
    if match and len(segment) <= 48:
        label = match.group("label").strip(" .-")
        words = label.split()
        if label and not any(any(c.isdigit() for c in w) for w in words) and \
                len(words) <= (6 if match.group("sep") or column_gap else 3):
            return label
    # OCR often turns the gap between columns into one space ("Basic: 40,000 HRA: 16,000"),
    # so look for a label at the end of the segment instead.
    core = re.sub(r"[ \t]*(?::-|[:=#–—])[ \t]*$", "", segment)
    has_sep = core != segment
    return trailing_label(re.split(r"[:=#]", core)[-1], separator=has_sep)


# --- checksums --------------------------------------------------------------------------
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_BASE36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def verhoeff_ok(digits: str) -> bool:
    check = 0
    for i, ch in enumerate(reversed(digits)):
        check = _VERHOEFF_D[check][_VERHOEFF_P[i % 8][int(ch)]]
    return check == 0


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch) * (2 if i % 2 else 1)
        total += d - 9 if d > 9 else d
    return total % 10 == 0


def gstin_ok(gstin: str) -> bool:
    total = 0
    for i, ch in enumerate(gstin[:14]):
        product = _BASE36.index(ch) * (2 if i % 2 else 1)
        total += product // 36 + product % 36
    return _BASE36[(36 - total % 36) % 36] == gstin[14]


def iban_ok(iban: str) -> bool:
    rearranged = iban[4:] + iban[:4]
    return int("".join(str(_BASE36.index(c)) for c in rearranged)) % 97 == 1


# --- detectors --------------------------------------------------------------------------
@dataclass
class _Hit:
    priority: int
    entity: Entity


@dataclass
class _Ctx:
    text: str
    page_number: Optional[int]

    def make(self, etype: str, start: int, end: int, *, status: EntityStatus, normalized: Optional[str] = None,
             label: Optional[str] = None, notes: Optional[list[str]] = None) -> Entity:
        line_start = self.text.rfind("\n", 0, start) + 1
        line_end = self.text.find("\n", end)
        return Entity(
            type=etype, value=self.text[start:end], normalized=normalized, status=status, label=label,
            page_number=self.page_number, line=self.text[line_start: len(self.text) if line_end == -1 else line_end],
            start=start, end=end, sensitive=etype in SENSITIVE_TYPES, notes=notes or [],
        )

    def prefix(self, start: int) -> str:
        return self.text[self.text.rfind("\n", 0, start) + 1: start]

    def label(self, start: int) -> Optional[str]:
        return label_before(self.prefix(start))


def _has(words: re.Pattern[str], label: Optional[str]) -> bool:
    return bool(label and words.search(label))


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def _status(labeled: bool) -> EntityStatus:
    return EntityStatus.VALID if labeled else EntityStatus.UNCERTAIN


EMAIL = EMAIL_RE
URL_RE = re.compile(r"(?<![\w@/])(?:https?://|www\.)[^\s<>\"'()]+", re.IGNORECASE)
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b")
GSTIN_RE = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")
CARD_RE = re.compile(r"(?<![\w.,-])\d(?:[ -]?\d){12,18}(?![\w-]|[.,]\d)")
AADHAAR_RE = re.compile(r"(?<![\w.,+-])[2-9]\d{3}([ -]?)\d{4}\1\d{4}(?![\w-]|[.,]\d)")
PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
IFSC_RE = re.compile(r"\b[A-Z]{4}[0O][A-Z0-9]{6}\b")
VEHICLE_RE = re.compile(r"\b(?P<state>[A-Z]{2})[ -]?(?P<district>[\dO]{1,2})[ -]?(?:[A-Z]{1,3}[ -]?)?\d{4}\b")
IP_RE = re.compile(r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\w.]|\.\d)")
TIME_RE = re.compile(
    r"\b(?:(?P<h>[01]?\d|2[0-3]):(?P<m>[0-5]\d)(?::[0-5]\d)?(?:[ ]?(?P<ap>[AaPp])\.?[Mm]\b\.?)?"
    r"|(?P<h2>0?[1-9]|1[0-2])(?:[.:](?P<m2>[0-5]\d))?[ ]?(?P<ap2>[AaPp])\.?[Mm]\b\.?)"
)
_CURRENCY = r"(?:[$€£₹¥]|\bRs\.?|\bINR\b|\bUSD\b|\bEUR\b|\bGBP\b|\bAED\b|\bSGD\b)"
_AMOUNT_NUM = r"\d[\d,]*(?:\.\d{1,2})?"
AMOUNT_PREFIX_RE = re.compile(rf"(?P<cur>{_CURRENCY})[ ]?(?P<num>{_AMOUNT_NUM})(?![\d,]|\.\d)(?:[ ]?/-)?")
AMOUNT_SUFFIX_RE = re.compile(
    rf"(?<![\w.,])(?P<num>{_AMOUNT_NUM})[ ]?(?P<cur>/-|INR\b|USD\b|EUR\b|GBP\b|AED\b|rupees\b|Rs\.?)", re.IGNORECASE
)
PERCENT_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?[ ]?%")
PHONE_RE = re.compile(r"(?<![\w+/.,-])(?:\+\d{1,3}[ .-]?)?(?:\(\d{1,5}\)[ .-]?)?\d(?:[ .-]?\d){5,14}(?![\w/-]|[.,]\d)")
PIN_RE = re.compile(r"(?<![\w.,-])[1-9]\d{2} ?\d{3}(?![\w-]|[.,]\d)")
ZIP_RE = re.compile(r"(?<![\w.,-])\d{5}(?:-\d{4})?(?![\w-]|[.,]\d)")  # US ZIP / ZIP+4
REFERENCE_RE = re.compile(
    r"(?<![\w/#-])(?=[A-Z0-9/#-]*\d)(?=[A-Z0-9/#-]*[A-Z])[A-Z0-9](?:[A-Z0-9]|[-/#](?=[A-Z0-9])){3,39}(?![\w-])"
)
NUMBER_RE = re.compile(r"(?<![\w.,/:-])-?\d{1,3}(?:,\d{2,3})+(?:\.\d+)?(?![\w,]|\.\d)|(?<![\w.,/:-])-?\d+(?:\.\d+)?(?![\w,]|\.\d)")
_LIST_MARKER = re.compile(r"[ \t]*\d{1,2}[.)][ \t]")
_GROUPING_OK = re.compile(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d{1,2}(?:,\d{2})*,\d{3}(?:\.\d+)?|\d+(?:\.\d+)?")
_STATE_CODES = set(
    "AN AP AR AS BR CH CG DD DL GA GJ HR HP JK JH KA KL LA LD MP MH MN ML MZ NL OD OR PY PB RJ SK TN TS TR UP UK "
    "UA WB BH".split()
)
_CURRENCY_CODES = {"₹": "INR", "RS": "INR", "RS.": "INR", "INR": "INR", "RUPEES": "INR", "/-": None, "€": "EUR",
                   "£": "GBP", "USD": "USD", "EUR": "EUR", "GBP": "GBP", "AED": "AED", "SGD": "SGD"}


def _emails(ctx: _Ctx) -> Iterable[Entity]:
    for m in EMAIL.finditer(ctx.text):
        yield ctx.make("email", m.start(), m.end(), status=EntityStatus.VALID, normalized=m.group(0).lower(),
                       label=ctx.label(m.start()))


def _urls(ctx: _Ctx) -> Iterable[Entity]:
    for m in URL_RE.finditer(ctx.text):
        end = m.start() + len(m.group(0).rstrip(".,;:!?"))
        yield ctx.make("url", m.start(), end, status=EntityStatus.VALID, label=ctx.label(m.start()))


def _ibans(ctx: _Ctx) -> Iterable[Entity]:
    for m in IBAN_RE.finditer(ctx.text):
        compact = m.group(0).replace(" ", "")
        label = ctx.label(m.start())
        if not 15 <= len(compact) <= 34:
            continue
        if iban_ok(compact):
            yield ctx.make("iban", m.start(), m.end(), status=EntityStatus.VALID, normalized=compact, label=label)
        elif _has(_kw("iban"), label):
            yield ctx.make("iban", m.start(), m.end(), status=EntityStatus.INVALID, label=label,
                           notes=["IBAN check digits do not match; possibly an OCR error."])


def _gstins(ctx: _Ctx) -> Iterable[Entity]:
    for m in GSTIN_RE.finditer(ctx.text):
        ok = gstin_ok(m.group(0))
        yield ctx.make("gstin", m.start(), m.end(), status=EntityStatus.VALID if ok else EntityStatus.INVALID,
                       normalized=m.group(0) if ok else None, label=ctx.label(m.start()),
                       notes=[] if ok else ["GSTIN check character does not match; possibly an OCR error."])


def _cards(ctx: _Ctx) -> Iterable[Entity]:
    for m in CARD_RE.finditer(ctx.text):
        digits, label = _digits(m.group(0)), ctx.label(m.start())
        grouped = bool(re.fullmatch(r"\d{4}([ -])\d{4}\1\d{4}\1\d{1,7}", m.group(0)))
        if not 13 <= len(digits) <= 19 or _has(PHONE_WORDS, label) or _has(ACCOUNT_WORDS, label):
            continue
        if not (_has(CARD_WORDS, label) or grouped):
            continue
        if luhn_ok(digits):
            yield ctx.make("card_number", m.start(), m.end(), status=_status(_has(CARD_WORDS, label)),
                           normalized=digits, label=label)
        elif _has(CARD_WORDS, label):
            yield ctx.make("card_number", m.start(), m.end(), status=EntityStatus.INVALID, label=label,
                           notes=["Card number fails the Luhn check; possibly an OCR error."])


def _aadhaars(ctx: _Ctx) -> Iterable[Entity]:
    words = _kw("aadhaar", "aadhar", "uid", "uidai", "enrolment")
    for m in AADHAAR_RE.finditer(ctx.text):
        digits, label = _digits(m.group(0)), ctx.label(m.start())
        if _has(PHONE_WORDS, label) or _has(ACCOUNT_WORDS, label):
            continue
        labeled = _has(words, label) or bool(words.search(ctx.prefix(m.start())))
        if verhoeff_ok(digits):
            notes = [] if labeled else ["12 digits with a valid Aadhaar checksum, but no Aadhaar label nearby."]
            yield ctx.make("aadhaar", m.start(), m.end(), status=_status(labeled), normalized=digits, label=label,
                           notes=notes)
        elif labeled:
            yield ctx.make("aadhaar", m.start(), m.end(), status=EntityStatus.INVALID, label=label,
                           notes=["Aadhaar checksum does not match; possibly an OCR error."])


def _pans(ctx: _Ctx) -> Iterable[Entity]:
    words = _kw("pan", "permanent account number")
    for m in PAN_RE.finditer(ctx.text):
        label = ctx.label(m.start())
        labeled = _has(words, label)
        if m.group(0)[3] in "ABCFGHLJPT":
            yield ctx.make("pan", m.start(), m.end(), status=_status(labeled), normalized=m.group(0), label=label)
        elif labeled:
            yield ctx.make("pan", m.start(), m.end(), status=EntityStatus.INVALID, label=label,
                           notes=["4th character is not a valid PAN holder type; possibly an OCR error."])


def _ifscs(ctx: _Ctx) -> Iterable[Entity]:
    words = _kw("ifsc", "ifs code")
    for m in IFSC_RE.finditer(ctx.text):
        label = ctx.label(m.start())
        if m.group(0)[4] == "0":
            yield ctx.make("ifsc", m.start(), m.end(), status=_status(_has(words, label)), normalized=m.group(0),
                           label=label)
        elif _has(words, label):
            yield ctx.make("ifsc", m.start(), m.end(), status=EntityStatus.INVALID, label=label,
                           notes=["5th character must be the digit 0; OCR may have read it as the letter O."])


def _vehicles(ctx: _Ctx) -> Iterable[Entity]:
    for m in VEHICLE_RE.finditer(ctx.text):
        if m.group("state") not in _STATE_CODES:
            continue
        label = ctx.label(m.start())
        if "O" not in m.group("district"):
            yield ctx.make("vehicle_number", m.start(), m.end(), status=_status(_has(VEHICLE_WORDS, label)),
                           normalized=re.sub(r"[ -]", "", m.group(0)), label=label)
        elif _has(VEHICLE_WORDS, label):
            yield ctx.make("vehicle_number", m.start(), m.end(), status=EntityStatus.INVALID, label=label,
                           notes=["District code must be digits; OCR may have read 0 as the letter O."])


def _ips(ctx: _Ctx) -> Iterable[Entity]:
    for m in IP_RE.finditer(ctx.text):
        yield ctx.make("ip_address", m.start(), m.end(), status=EntityStatus.VALID, normalized=m.group(0),
                       label=ctx.label(m.start()))


def _dates(ctx: _Ctx) -> Iterable[Entity]:
    for m in DATE_RE.finditer(ctx.text):
        check = check_date(m.group(0))
        if not check.ok:
            status, notes = EntityStatus.INVALID, [f"Looks like a date but {check.message}."]
        elif check.uncertain:
            status, notes = EntityStatus.UNCERTAIN, [f"Date is uncertain: {check.message}."]
        else:
            status, notes = EntityStatus.VALID, []
        yield ctx.make("date", m.start(), m.end(), status=status, normalized=check.normalized,
                       label=ctx.label(m.start()), notes=notes)


def _times(ctx: _Ctx) -> Iterable[Entity]:
    for m in TIME_RE.finditer(ctx.text):
        hour = int(m.group("h") or m.group("h2"))
        minute = int(m.group("m") or m.group("m2") or 0)
        ampm = (m.group("ap") or m.group("ap2") or "").lower()
        if ampm and not 1 <= hour <= 12:
            continue
        if ampm:
            hour = hour % 12 + (12 if ampm == "p" else 0)
        yield ctx.make("time", m.start(), m.end(), status=EntityStatus.VALID, normalized=f"{hour:02d}:{minute:02d}",
                       label=ctx.label(m.start()))


def _amount_entity(ctx: _Ctx, start: int, end: int, number: str, currency: Optional[str],
                   label: Optional[str]) -> Entity:
    notes, status = [], EntityStatus.VALID
    if not _GROUPING_OK.fullmatch(number):
        status = EntityStatus.UNCERTAIN
        notes.append("Unusual digit grouping; check the value (possible OCR error).")
    plain = number.replace(",", "")
    if currency is None:
        notes.append("No currency shown.")
    return ctx.make("amount", start, end, status=status, normalized=f"{currency} {plain}" if currency else plain,
                    label=label, notes=notes)


def _amounts(ctx: _Ctx) -> Iterable[Entity]:
    for regex in (AMOUNT_PREFIX_RE, AMOUNT_SUFFIX_RE):
        for m in regex.finditer(ctx.text):
            raw_cur = m.group("cur").strip()
            currency = _CURRENCY_CODES.get(raw_cur.upper(), raw_cur)
            yield _amount_entity(ctx, m.start(), m.end(), m.group("num"), currency, ctx.label(m.start()))


def _percentages(ctx: _Ctx) -> Iterable[Entity]:
    for m in PERCENT_RE.finditer(ctx.text):
        yield ctx.make("percentage", m.start(), m.end(), status=EntityStatus.VALID,
                       normalized=re.sub(r"\s", "", m.group(0)), label=ctx.label(m.start()))


def _phones(ctx: _Ctx) -> Iterable[Entity]:
    for m in PHONE_RE.finditer(ctx.text):
        value, label = m.group(0), ctx.label(m.start())
        digits = _digits(value)
        if not 7 <= len(digits) <= 15 or re.fullmatch(r"\d+\.\d+", value):
            continue
        context = ctx.prefix(m.start())[-40:]
        normalized = ("+" if value.startswith("+") else "") + digits
        if _has(PHONE_WORDS, label) or PHONE_WORDS.search(context):
            yield ctx.make("phone", m.start(), m.end(), status=EntityStatus.VALID, normalized=normalized, label=label)
        elif value.startswith("+") and len(digits) >= 8:
            yield ctx.make("phone", m.start(), m.end(), status=EntityStatus.VALID, normalized=normalized, label=label,
                           notes=["International format (+country code)."])
        elif _has(AMOUNT_WORDS, label) or _has(ID_WORDS, label) or _has(ACCOUNT_WORDS, label) or _has(PIN_WORDS, label):
            continue
        elif re.fullmatch(r"(?:0|91)?[6-9]\d{9}", digits):
            yield ctx.make("phone", m.start(), m.end(), status=EntityStatus.UNCERTAIN, normalized=normalized,
                           label=label, notes=["Looks like an Indian mobile number; no phone label nearby."])
        elif re.match(r"\(?0\d", value) and 10 <= len(digits) <= 11 and re.search(r"[ ()-]", value):
            yield ctx.make("phone", m.start(), m.end(), status=EntityStatus.UNCERTAIN, normalized=normalized,
                           label=label, notes=["Looks like a landline number with an area code; no phone label nearby."])


def _labeled_numbers(ctx: _Ctx) -> Iterable[Entity]:
    """Account numbers and PIN codes need a label; without one they stay plain numbers."""
    for m in re.finditer(r"(?<![\w.,-])\d[\d -]{7,22}\d(?![\w-]|[.,]\d)", ctx.text):
        label = ctx.label(m.start())
        digits = _digits(m.group(0))
        if _has(ACCOUNT_WORDS, label) and 9 <= len(digits) <= 18 and not _has(PHONE_WORDS, label):
            yield ctx.make("account_number", m.start(), m.end(), status=EntityStatus.VALID, normalized=digits,
                           label=label)
    for m in PIN_RE.finditer(ctx.text):
        label = ctx.label(m.start())
        prefix = ctx.prefix(m.start())
        rest = ctx.text[m.end(): (ctx.text.find("\n", m.end()) + 1 or len(ctx.text) + 1) - 1]
        if _has(PIN_WORDS, label):
            yield ctx.make("pin_code", m.start(), m.end(), status=EntityStatus.VALID,
                           normalized=m.group(0).replace(" ", ""), label=label)
        elif re.search(r"[A-Za-z]{3,}[ \t]*[,–-]?[ \t]*$", prefix) and re.fullmatch(r"[ \t.,]*", rest) and "," in prefix:
            yield ctx.make("pin_code", m.start(), m.end(), status=EntityStatus.UNCERTAIN,
                           normalized=m.group(0).replace(" ", ""), label=label,
                           notes=["6 digits at the end of an address line; no PIN label."])
    for m in ZIP_RE.finditer(ctx.text):
        label = ctx.label(m.start())
        prefix = ctx.prefix(m.start())
        rest = ctx.text[m.end(): (ctx.text.find("\n", m.end()) + 1 or len(ctx.text) + 1) - 1]
        if _has(PIN_WORDS, label):
            yield ctx.make("pin_code", m.start(), m.end(), status=EntityStatus.VALID, normalized=m.group(0),
                           label=label)
        elif re.search(r",[ \t]*[A-Z]{2}[ \t]+$", prefix) and re.fullmatch(r"[ \t.,]*", rest):
            yield ctx.make("pin_code", m.start(), m.end(), status=EntityStatus.UNCERTAIN, normalized=m.group(0),
                           label=label, notes=["US-style ZIP after a state code at the end of an address line; "
                                               "no ZIP label."])


def _references(ctx: _Ctx) -> Iterable[Entity]:
    for m in REFERENCE_RE.finditer(ctx.text):
        value, label = m.group(0), ctx.label(m.start())
        if sum(c.isdigit() for c in value) < 2 and len(value) < 6 and not _has(ID_WORDS, label):
            continue  # "A4", "B12": too short to call an identifier without an ID label
        yield ctx.make("reference", m.start(), m.end(), status=_status(_has(ID_WORDS, label)),
                       normalized=value, label=label)


def _numbers(ctx: _Ctx) -> Iterable[Entity]:
    for m in NUMBER_RE.finditer(ctx.text):
        if _LIST_MARKER.fullmatch(ctx.prefix(m.start()) + ctx.text[m.start(): m.end() + 2]):
            continue  # "1. Item" list numbering
        label, value = ctx.label(m.start()), m.group(0)
        plain = value.replace(",", "")
        if _has(AMOUNT_WORDS, label) and not _has(ID_WORDS, label):
            yield _amount_entity(ctx, m.start(), m.end(), value.lstrip("-"), None, label)
        elif _has(ID_WORDS, label) and re.fullmatch(r"\d{2,}", value):
            yield ctx.make("reference", m.start(), m.end(), status=EntityStatus.VALID, normalized=value, label=label)
        else:
            notes = [] if _GROUPING_OK.fullmatch(value.lstrip("-")) else ["Unusual digit grouping; possible OCR error."]
            yield ctx.make("number", m.start(), m.end(), status=EntityStatus.UNCERTAIN if notes else EntityStatus.VALID,
                           normalized=plain, label=label, notes=notes)


# Higher priority first: a span claimed by an earlier detector is not reported again.
DETECTORS: tuple[Callable[[_Ctx], Iterable[Entity]], ...] = (
    _emails, _urls, _ibans, _gstins, _cards, _aadhaars, _pans, _ifscs, _vehicles, _ips, _dates, _times, _amounts,
    _percentages, _phones, _labeled_numbers, _references, _numbers,
)
_STRIP_TOKEN = re.compile(r"^\W+|\W+$")


def find_entities(text: str, *, page_number: Optional[int] = None,
                  low_confidence_words: Iterable[str] = ()) -> list[Entity]:
    """Return non-overlapping entities in reading order."""
    ctx = _Ctx(text, page_number)
    taken: list[tuple[int, int]] = []
    accepted: list[Entity] = []
    for priority, detector in enumerate(DETECTORS):
        hits = sorted(detector(ctx), key=lambda e: (e.start, -(e.end - e.start)))
        for entity in hits:
            if entity.end <= entity.start or any(entity.start < e and entity.end > s for s, e in taken):
                continue
            taken.append((entity.start, entity.end))
            accepted.append(entity)
    low = {_STRIP_TOKEN.sub("", w).casefold() for w in low_confidence_words}
    low.discard("")
    for entity in accepted:
        if low and any(_STRIP_TOKEN.sub("", tok).casefold() in low for tok in entity.value.split()):
            entity.status = EntityStatus.UNCERTAIN if entity.status is EntityStatus.VALID else entity.status
            entity.notes.append("Contains low-confidence OCR words.")
    return sorted(accepted, key=lambda e: e.start)


def mask(value: str, keep: int = 4) -> str:
    """Hide all but the last ``keep`` letters/digits of a sensitive value (for display only)."""
    total = sum(c.isalnum() for c in value)
    seen, out = 0, []
    for ch in value:
        if ch.isalnum():
            seen += 1
            out.append(ch if seen > total - keep else "•")
        else:
            out.append(ch)
    return "".join(out)
