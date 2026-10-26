"""Rule-based document-type classification.

Each type has title phrases (strong evidence, mostly in the first lines), keywords and
optional entity hints. The score is a weighted count of distinct matches; it is evidence
strength, not a probability. When the best score is weak or close to the runner-up the
result says so, and with too little evidence the type is ``unknown``. No model is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from docintel.core.result_models import Classification, Entity, KeyValuePair, TypeScore


@dataclass(frozen=True)
class DocType:
    label: str
    titles: tuple[str, ...]
    keywords: tuple[str, ...]
    entities: dict[str, float] = field(default_factory=dict)  # entity type -> bonus when present


DOCUMENT_TYPES: dict[str, DocType] = {
    "invoice": DocType(
        "Invoice", ("tax invoice", "invoice", "proforma invoice", "bill of supply"),
        ("invoice no", "invoice number", "invoice date", "bill to", "ship to", "due date", "hsn", "sac", "subtotal",
         "sub total", "grand total", "amount due", "qty", "quantity", "unit price", "rate", "cgst", "sgst", "igst",
         "taxable value", "payment terms", "place of supply"),
        {"gstin": 1.5, "amount": 1.0},
    ),
    "receipt": DocType(
        "Receipt", ("receipt", "payment receipt", "cash receipt", "money receipt"),
        ("received with thanks", "payment received", "amount paid", "paid", "cash", "change", "txn id",
         "transaction id", "upi", "payment mode", "thank you for", "cashier", "receipt no"),
        {"amount": 1.0},
    ),
    "purchase_order": DocType(
        "Purchase order", ("purchase order",),
        ("po number", "po no", "p.o.", "supplier", "vendor", "delivery date", "ship to", "order date",
         "terms and conditions", "qty", "unit price"),
        {"amount": 0.5},
    ),
    "bank_statement": DocType(
        "Bank statement", ("account statement", "statement of account", "bank statement"),
        ("opening balance", "closing balance", "withdrawal", "deposit", "debit", "credit", "transaction date",
         "value date", "cheque", "narration", "account number", "branch", "balance"),
        {"ifsc": 1.5, "account_number": 1.5, "amount": 0.5},
    ),
    "payslip": DocType(
        "Payslip", ("payslip", "pay slip", "salary slip", "salary statement"),
        ("earnings", "deductions", "net pay", "gross salary", "basic", "hra", "provident fund", "pf", "esi",
         "professional tax", "employee id", "employee code", "designation", "pay period", "lop", "days worked"),
        {"amount": 0.5},
    ),
    "identity_document": DocType(
        "Identity document", ("government of india", "income tax department", "permanent account number",
                              "driving licence", "passport", "election commission", "aadhaar"),
        ("date of birth", "dob", "gender", "male", "female", "father", "address", "valid till", "issue date",
         "nationality", "card", "enrolment"),
        {"pan": 2.0, "aadhaar": 2.0},
    ),
    "resume": DocType(
        "Resume / CV", ("curriculum vitae", "resume", "biodata"),
        ("education", "experience", "work experience", "skills", "projects", "objective", "certifications",
         "languages known", "career", "internship", "references", "achievements", "hobbies"),
        {"email": 0.5, "phone": 0.5},
    ),
    "letter": DocType(
        "Letter", ("dear sir", "dear madam", "to whom it may concern"),
        ("dear", "sincerely", "regards", "yours faithfully", "yours sincerely", "subject", "ref", "enclosure",
         "thanking you", "respected"),
        {},
    ),
    "incident_report": DocType(
        "Incident report", ("incident report", "accident report", "first information report", "complaint"),
        ("incident", "caller", "location", "incident type", "reported", "accident", "emergency", "police", "fir",
         "priority", "severity", "victim", "injured", "dispatch", "officer", "complainant", "status"),
        {"vehicle_number": 1.0, "time": 0.5},
    ),
    "medical": DocType(
        "Medical / prescription", ("prescription", "discharge summary", "medical report", "lab report"),
        ("patient", "diagnosis", "rx", "doctor", "hospital", "clinic", "dosage", "tablet", "capsule", "age",
         "symptoms", "blood", "test", "result", "reference range", "follow up"),
        {},
    ),
    "utility_bill": DocType(
        "Utility bill", ("electricity bill", "water bill", "gas bill", "telephone bill", "broadband bill"),
        ("consumer number", "consumer no", "units consumed", "meter reading", "previous reading", "current reading",
         "billing period", "bill date", "due date", "bill amount", "connection", "tariff", "late fee"),
        {"amount": 0.5},
    ),
    "contract": DocType(
        "Contract / agreement", ("agreement", "contract", "memorandum of understanding", "deed"),
        ("hereinafter", "party", "parties", "whereas", "witnesseth", "terms and conditions", "governing law",
         "termination", "indemnity", "jurisdiction", "signed", "witness", "effective date", "confidentiality"),
        {},
    ),
    "delivery_note": DocType(
        "Delivery note / challan", ("delivery challan", "delivery note", "dispatch note", "packing list"),
        ("consignee", "consignor", "e-way bill", "eway bill", "transporter", "lr no", "vehicle no", "dispatched",
         "delivered", "received by", "qty", "no of packages"),
        {"vehicle_number": 1.0, "gstin": 0.5},
    ),
    "certificate": DocType(
        "Certificate", ("certificate", "this is to certify"),
        ("certify", "certified", "awarded", "successfully completed", "in recognition", "issued", "authorised signatory"),
        {},
    ),
}

TITLE_WEIGHT = 3.0      # title phrase in the first lines
TITLE_BODY_WEIGHT = 1.5  # title phrase elsewhere
KEYWORD_WEIGHT = 1.0
UNKNOWN_BELOW = 3.0
HEAD_LINES = 5


def _phrase(phrase: str) -> re.Pattern[str]:
    words = [re.escape(w) for w in phrase.split()]
    return re.compile(r"(?<![a-z0-9])" + r"[\s_]+".join(words) + r"(?![a-z0-9])", re.IGNORECASE)


_COMPILED = {name: ([(_phrase(p), p) for p in t.titles], [(_phrase(p), p) for p in t.keywords])
             for name, t in DOCUMENT_TYPES.items()}


def _score(name: str, head: str, text: str, entity_types: set[str]) -> TypeScore:
    titles, keywords = _COMPILED[name]
    score, evidence = 0.0, []
    for regex, phrase in titles:
        if regex.search(head):
            score += TITLE_WEIGHT
            evidence.append(f'title "{phrase}"')
        elif regex.search(text):
            score += TITLE_BODY_WEIGHT
            evidence.append(f'"{phrase}"')
    title_words = {w for p in DOCUMENT_TYPES[name].titles for w in p.split()}
    for regex, phrase in keywords:
        if phrase in title_words:
            continue  # already counted as a title
        if regex.search(text):
            score += KEYWORD_WEIGHT
            evidence.append(f'"{phrase}"')
    for etype, bonus in DOCUMENT_TYPES[name].entities.items():
        if etype in entity_types:
            score += bonus
            evidence.append(f"{etype} found")
    return TypeScore(document_type=name, label=DOCUMENT_TYPES[name].label, score=round(score, 2), evidence=evidence)


def classify(text: str, entities: Iterable[Entity] = (), key_values: Iterable[KeyValuePair] = (),
             template_types: Optional[dict[str, str]] = None) -> Classification:
    """Rank document types for ``text``.

    ``template_types`` maps template name -> document_type, used to suggest templates.
    """
    head = "\n".join([ln for ln in text.splitlines() if ln.strip()][:HEAD_LINES])
    entity_types = {e.type for e in entities}
    scores = sorted((_score(name, head, text, entity_types) for name in DOCUMENT_TYPES),
                    key=lambda s: s.score, reverse=True)
    scores = [s for s in scores if s.score > 0]
    result = Classification(scores=scores)
    pairs = [kv for kv in key_values if not kv.empty]

    top = scores[0] if scores else None
    if top is None or top.score < UNKNOWN_BELOW:
        if len(pairs) >= 4:
            result.document_type, result.label, result.confidence = "form", "Form (labelled fields)", "low"
            result.notes.append(f"No document type matched clearly; {len(pairs)} labelled fields suggest a form.")
        else:
            result.notes.append("Not enough evidence to name a document type.")
    else:
        runner_up = scores[1].score if len(scores) > 1 else 0.0
        margin = top.score - runner_up
        result.document_type, result.label = top.document_type, top.label
        has_title = any(ev.startswith("title") for ev in top.evidence)
        if top.score >= 7 and margin >= 3 and has_title:
            result.confidence = "high"
        elif top.score >= 4.5 and margin >= 1.5:
            result.confidence = "medium"
        else:
            result.confidence = "low"
        if margin < 1.5 and len(scores) > 1:
            result.notes.append(f"Close to {scores[1].label} ({scores[1].score:g} vs {top.score:g}); check the evidence.")
    if template_types:
        result.suggested_templates = sorted(n for n, t in template_types.items() if t == result.document_type)
    return result
