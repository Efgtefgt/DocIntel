"""Document intelligence: entities, key-value pairs, classification, masking, pipeline and API."""

from __future__ import annotations

import pytest

from docintel.core.pipeline import DocumentPipeline, PipelineOptions
from docintel.core.result_models import EntityStatus
from docintel.export.csv_exporter import entities_to_csv_bytes, key_values_to_csv_bytes
from docintel.extraction.classifier import classify
from docintel.extraction.document_analysis import analyze_text, mask_sensitive
from docintel.extraction.entities import find_entities, gstin_ok, iban_ok, label_before, luhn_ok, mask, verhoeff_ok
from docintel.extraction.key_values import find_key_values
from tests.conftest import make_factory
from tests.fakes import FakeEngine

INVOICE = """TAX INVOICE
Acme Traders Pvt Ltd
Invoice No: INV-2024-0042        Invoice Date: 17/05/2024
Bill To: Ravi Kumar | Phone: +91 98450 12345
GSTIN: 29AAGCB7383J1Z4
Description        Qty   Rate     Amount
Steel bolts        250   12.50    3,125.00
CGST 9%: 281.25   SGST 9%: 281.25
Grand Total: Rs. 3,687.50
Email: accounts@acme.example.com  Web: https://acme.example.com/pay.
"""


def _one(text: str, etype: str):
    found = [e for e in find_entities(text) if e.type == etype]
    assert len(found) == 1, [(e.type, e.value) for e in find_entities(text)]
    return found[0]


def _types(text: str) -> list[tuple[str, str]]:
    return [(e.type, e.value) for e in find_entities(text)]


# --- invariants -----------------------------------------------------------------------------
def test_entities_are_verbatim_and_never_overlap():
    entities = find_entities(INVOICE)
    assert entities
    spans = []
    for e in entities:
        assert INVOICE[e.start:e.end] == e.value
        assert e.line in INVOICE
        spans.append((e.start, e.end))
    spans.sort()
    assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(spans, spans[1:]))


def test_plain_prose_has_no_entities():
    assert find_entities("The quick brown fox jumps over the lazy dog.") == []


# --- contact details --------------------------------------------------------------------------
def test_email_and_url_are_separate_and_trailing_punctuation_dropped():
    pairs = _types(INVOICE)
    assert ("email", "accounts@acme.example.com") in pairs
    assert ("url", "https://acme.example.com/pay") in pairs
    assert not any(t == "url" and "@" in v for t, v in pairs)


@pytest.mark.parametrize("text, normalized, status", [
    ("Phone: 080 2345 6789", "08023456789", EntityStatus.VALID),
    ("Reach us at +44 20 7946 0958 today", "+442079460958", EntityStatus.VALID),
    ("Ravi 9845012345", "9845012345", EntityStatus.UNCERTAIN),
    ("Call the desk on 080-2345-6789", "08023456789", EntityStatus.VALID),
])
def test_phone_numbers(text, normalized, status):
    phone = _one(text, "phone")
    assert phone.normalized == normalized and phone.status is status


@pytest.mark.parametrize("text", ["Population 12345678", "Total: 12345678", "Invoice No: 98450123"])
def test_numbers_without_phone_evidence_are_not_phones(text):
    assert not [e for e in find_entities(text) if e.type == "phone"]


# --- dates, times, amounts --------------------------------------------------------------------
def test_dates_valid_ambiguous_and_impossible():
    assert _one("Date: 17/05/2024", "date").normalized == "2024-05-17"
    ambiguous = _one("Date: 03/04/2024", "date")
    assert ambiguous.status is EntityStatus.UNCERTAIN and ambiguous.normalized is None
    impossible = _one("Date: 31/02/2024", "date")
    assert impossible.status is EntityStatus.INVALID


@pytest.mark.parametrize("text, normalized", [("at 14:35", "14:35"), ("at 2:30 pm", "14:30"), ("by 9 AM", "09:00"),
                                              ("12:05 am", "00:05")])
def test_times(text, normalized):
    assert _one(text, "time").normalized == normalized


@pytest.mark.parametrize("text, normalized", [
    ("Paid ₹ 3,125.00 today", "INR 3125.00"), ("Grand Total: Rs. 4,572.50", "INR 4572.50"),
    ("Fee 1,500/-", "1500"), ("Price $19.99", "$ 19.99"), ("Total: 1,23,456.00", "123456.00"),
    ("EUR 250", "EUR 250"),
])
def test_amounts(text, normalized):
    assert _one(text, "amount").normalized == normalized


def test_labelled_amount_without_currency_says_so():
    amount = _one("Sub Total: 3,875.00", "amount")
    assert amount.status is EntityStatus.VALID and "No currency shown." in amount.notes


def test_unusual_digit_grouping_is_uncertain():
    amount = _one("Total: ₹ 12,34.00", "amount")
    assert amount.status is EntityStatus.UNCERTAIN


def test_percentages():
    assert _one("GST 18%", "percentage").normalized == "18%"
    assert _one("Discount 12.5 %", "percentage").normalized == "12.5%"


# --- identifiers with checksums -----------------------------------------------------------------
def test_checksum_helpers():
    assert gstin_ok("29AAGCB7383J1Z4") and gstin_ok("27AAPFU0939F1ZV") and not gstin_ok("29AAGCB7383J1Z5")
    assert verhoeff_ok("499118665246") and not verhoeff_ok("499118665247")
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112")
    assert iban_ok("GB82WEST12345698765432") and not iban_ok("GB82WEST12345698765433")


def test_gstin_valid_and_ocr_error_is_invalid():
    assert _one("GSTIN: 29AAGCB7383J1Z4", "gstin").status is EntityStatus.VALID
    bad = _one("GSTIN: 29AAGCB7383J1Z5", "gstin")
    assert bad.status is EntityStatus.INVALID and bad.normalized is None


def test_pan_needs_label_to_be_valid():
    assert _one("PAN: AAGCB7383J", "pan").status is EntityStatus.VALID
    assert _one("ref AAGCB7383J", "pan").status is EntityStatus.UNCERTAIN
    assert _one("PAN: AAGCB7383J", "pan").sensitive


def test_aadhaar_checksum_and_label():
    assert _one("Aadhaar: 4991 1866 5246", "aadhaar").status is EntityStatus.VALID
    assert _one("Aadhaar No: 4991 1866 5247", "aadhaar").status is EntityStatus.INVALID
    assert not [e for e in find_entities("Count 4991 1866 5247") if e.type == "aadhaar"]


def test_card_number_needs_luhn_and_card_evidence():
    card = _one("Card: 4111 1111 1111 1111", "card_number")
    assert card.status is EntityStatus.VALID and card.normalized == "4111111111111111" and card.sensitive
    assert not [e for e in find_entities("4111 1111 1111 1112") if e.type == "card_number"]


def test_iban_ifsc_account_ip_vehicle_pin():
    assert _one("IBAN: GB82 WEST 1234 5698 7654 32", "iban").normalized == "GB82WEST12345698765432"
    assert _one("IFSC: HDFC0001234", "ifsc").status is EntityStatus.VALID
    account = _one("Bank A/c No: 50100234567890", "account_number")
    assert account.normalized == "50100234567890" and account.sensitive
    assert _one("Server 192.168.1.10 is down", "ip_address").value == "192.168.1.10"
    vehicle = _one("Vehicle No: KA 01 AB 1234", "vehicle_number")
    assert vehicle.normalized == "KA01AB1234" and vehicle.status is EntityStatus.VALID
    assert _one("Pincode: 560001", "pin_code").status is EntityStatus.VALID
    assert _one("12 MG Road, Bengaluru, 560001", "pin_code").status is EntityStatus.UNCERTAIN


# --- references and numbers ---------------------------------------------------------------------
def test_references():
    ref = _one("Invoice No: INV-2024-0042", "reference")
    assert ref.label == "Invoice No" and ref.status is EntityStatus.VALID
    assert _one("Ticket #45821", "reference").value == "45821"
    assert _one("see INC-1024 for details", "reference").status is EntityStatus.UNCERTAIN


def test_numbers_and_list_markers():
    pairs = _types("1. Steel bolts 250\n2) Washers 1,000\nCount 1,23,456")
    assert ("number", "250") in pairs and ("number", "1,000") in pairs and ("number", "1,23,456") in pairs
    assert ("number", "1") not in pairs and ("number", "2") not in pairs


def test_low_confidence_words_make_entities_uncertain():
    entity = find_entities("Phone: 080 2345 6789", low_confidence_words=["6789"])[0]
    assert entity.status is EntityStatus.UNCERTAIN and "low-confidence" in entity.notes[-1]


@pytest.mark.parametrize("prefix, label", [("Invoice No: ", "Invoice No"), ("Ticket #", "Ticket"),
                                           ("Name   Phone - ", "Phone"), ("Contact the help desk on ", None),
                                           ("done. Aadhaar: ", "Aadhaar")])
def test_label_before(prefix, label):
    assert label_before(prefix) == label


# --- key-value pairs ----------------------------------------------------------------------------
def test_key_values_split_line_segments_and_type_values():
    pairs = {kv.key: kv for kv in find_key_values(INVOICE, entities=find_entities(INVOICE))}
    assert pairs["Invoice No"].value == "INV-2024-0042" and pairs["Invoice No"].value_type == "reference"
    assert pairs["Invoice Date"].value_type == "date"
    assert pairs["Bill To"].value == "Ravi Kumar" and pairs["Phone"].value_type == "phone"
    assert pairs["CGST 9%"].value == "281.25"
    assert "https" not in pairs and "Web" in pairs


def test_key_values_next_line_and_empty():
    pairs = find_key_values("Hobby:\nReading\nSignature:\n\nRemarks:")
    found = {kv.key: kv for kv in pairs}
    assert found["Hobby"].value == "Reading" and "next line" in found["Hobby"].notes[0]
    assert found["Signature"].empty and found["Remarks"].empty


def test_key_values_split_columns_merged_by_ocr():
    text = "Basic: 40,000 HRA: 16,000\nEmployee Name: Anita Rao Employee ID: EMP0042\nTime: 4:15 PM"
    pairs = [(kv.key, kv.value) for kv in find_key_values(text)]
    assert pairs == [("Basic", "40,000"), ("HRA", "16,000"), ("Employee Name", "Anita Rao"),
                     ("Employee ID", "EMP0042"), ("Time", "4:15 PM")]


def test_labels_survive_merged_columns():
    amounts = [e for e in find_entities("Basic: 40,000 HRA: 16,000") if e.type == "amount"]
    assert [e.label for e in amounts] == ["Basic", "HRA"]
    assert _one("Payment Mode: UPI Txn ID: 412233987711", "reference").label == "Txn ID"
    assert _one("Steel bolts Qty 250 Rate 12.50", "amount").label == "Rate"


def test_letter_o_for_zero_is_flagged_not_corrected():
    ifsc = _one("IFSC: SBINO004567", "ifsc")
    assert ifsc.status is EntityStatus.INVALID and ifsc.normalized is None and "letter O" in ifsc.notes[0]
    vehicle = _one("Vehicle No: KA O1 AB 1234", "vehicle_number")
    assert vehicle.status is EntityStatus.INVALID and vehicle.normalized is None


def test_column_gap_acts_as_label_separator():
    amount = _one("05/06/2025   ATM Withdrawal   2,000.00", "amount")
    assert amount.label == "ATM Withdrawal"


def test_times_and_sentences_are_not_key_values():
    assert find_key_values("Reported on 17 March 2024 at 14:35") == []
    assert find_key_values("This is a long sentence that happens to end with a colon: yes") == []


# --- classification -----------------------------------------------------------------------------
def test_classifies_invoice_with_strong_evidence_and_suggests_template():
    c = classify(INVOICE, find_entities(INVOICE), template_types={"invoice": "invoice", "x": "incident_report"})
    assert c.document_type == "invoice" and c.confidence == "high"
    assert c.suggested_templates == ["invoice"]
    assert any(ev.startswith("title") for ev in c.scores[0].evidence)


@pytest.mark.parametrize("text, expected", [
    ("SALARY SLIP\nEarnings  Deductions\nBasic: 40,000\nNet Pay: 51,000\nProvident Fund: 4,800", "payslip"),
    ("Dear Sir,\nSubject: Leave\nPlease grant leave.\nYours faithfully,\nRam", "letter"),
    ("INCIDENT REPORT\nCaller Name: Asha\nLocation: MG Road\nIncident Type: Fire\nPriority: High", "incident_report"),
    ("STATEMENT OF ACCOUNT\nOpening Balance 1,000.00\nWithdrawal 200.00\nClosing Balance 800.00", "bank_statement"),
])
def test_classifies_common_types(text, expected):
    assert analyze_text(text).classification.document_type == expected


def test_unknown_when_evidence_is_weak():
    c = classify("The quick brown fox jumps over the lazy dog.")
    assert c.document_type == "unknown" and c.confidence == "none" and c.notes


def test_many_labelled_fields_fall_back_to_form():
    c = analyze_text("Name: A\nCity: Pune\nHobby: Chess\nFavourite colour: Blue").classification
    assert c.document_type == "form" and c.confidence == "low"


def test_close_call_is_reported():
    c = classify("Invoice\nReceipt\npaid  qty  cash")
    assert c.confidence == "low"


# --- masking and export -------------------------------------------------------------------------
def test_mask_keeps_last_four_and_layout():
    assert mask("4111 1111 1111 1111") == "•••• •••• •••• 1111"
    assert mask("AAGCB7383J") == "••••••383J"


def test_mask_sensitive_hides_ids_everywhere_without_touching_original():
    analysis = analyze_text("PAN: AAGCB7383J\nAadhaar: 4991 1866 5246\nEmail: a@b.co")
    masked = mask_sensitive(analysis)
    dumped = masked.model_dump_json()
    assert "AAGCB7383J" not in dumped and "4991 1866 5246" not in dumped and "499118665246" not in dumped
    assert "a@b.co" in dumped
    assert any(e.value == "AAGCB7383J" for e in analysis.entities)


def test_csv_exports_have_headers_even_when_empty():
    empty = analyze_text("nothing here")
    assert entities_to_csv_bytes(empty).decode("utf-8-sig").startswith("page,type,value")
    assert key_values_to_csv_bytes(empty).decode("utf-8-sig").startswith("page,key,value")
    full = analyze_text(INVOICE)
    assert b"accounts@acme.example.com" in entities_to_csv_bytes(full)


# --- pipeline and API ---------------------------------------------------------------------------
def test_pipeline_attaches_analysis(settings, png_bytes):
    engine = FakeEngine(text="TAX INVOICE\nInvoice No: INV-7\nEmail: a@b.co\nTotal: Rs. 500")
    pipeline = DocumentPipeline(settings, make_factory(settings, engine))
    result = pipeline.process(png_bytes, "scan.png", PipelineOptions(engine="fake", preprocess_preset="original"))
    assert result.analysis is not None
    assert result.analysis.classification.document_type == "invoice"
    assert result.analysis.classification.suggested_templates == ["invoice"]
    assert {e.type for e in result.analysis.entities} >= {"email", "amount", "reference"}
    assert all(e.page_number == 1 for e in result.analysis.entities)
    assert any(s.stage == "analysis" and s.status.value == "success" for s in result.stages)


def test_pipeline_analysis_can_be_switched_off(settings, png_bytes):
    pipeline = DocumentPipeline(settings, make_factory(settings, FakeEngine(text="a@b.co")))
    result = pipeline.process(png_bytes, "scan.png", PipelineOptions(engine="fake", preprocess_preset="original",
                                                                     analyze=False))
    assert result.analysis is None


def test_api_analyze_and_process(settings, png_bytes):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from docintel.api.main import create_app

    client = TestClient(create_app(settings, make_factory(settings, FakeEngine(text="Phone: 080 2345 6789"))))
    body = client.post("/analyze", json={"text": INVOICE}).json()
    assert body["classification"]["document_type"] == "invoice"
    assert any(e["type"] == "gstin" for e in body["entities"])
    processed = client.post("/process", files={"file": ("s.png", png_bytes, "image/png")},
                            data={"engine": "fake", "preprocess": "original"}).json()
    assert processed["analysis"]["entities"][0]["type"] == "phone"
