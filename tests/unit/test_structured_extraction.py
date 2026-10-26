"""Template parsing, field extraction, validation and user corrections."""

from __future__ import annotations

import pytest

from docintel.core.exceptions import TemplateError, UnsafePathError
from docintel.core.result_models import FieldStatus
from docintel.extraction.field_validators import build_record_model, check_date, check_phone
from docintel.extraction.structured_extractor import StructuredExtractor
from docintel.extraction.templates import TemplateStore, parse_template
from tests.conftest import TEMPLATES_DIR

FULL_REPORT = """INCIDENT REPORT
Caller Name: Priya Sharma
Phone Number: +91 98450 12345
Incident Location: 14 MG Road, Bengaluru
Incident Type: Road accident
Incident Date: 2024-05-17
Description: Two-wheeler collided with a parked car.
The rider was taken to hospital.

Officer: R. Kumar
"""


@pytest.fixture
def incident():
    return StructuredExtractor(TemplateStore(TEMPLATES_DIR).load("incident_report"))


def test_full_extraction_with_source_text(incident):
    result = incident.extract(FULL_REPORT)
    f = result.fields
    assert f["caller_name"].value == "Priya Sharma" and f["caller_name"].status is FieldStatus.FOUND
    assert f["phone_number"].normalized_value == "+919845012345"
    assert f["incident_date"].normalized_value == "2024-05-17"
    assert f["description"].value == "Two-wheeler collided with a parked car. The rider was taken to hospital."
    assert f["incident_location"].source_text == "Incident Location: 14 MG Road, Bengaluru"
    start, end = f["caller_name"].source_start, f["caller_name"].source_end
    assert FULL_REPORT[start:end] == "Priya Sharma"
    assert result.valid


def test_missing_fields_are_null_not_guessed(incident):
    result = incident.extract("Caller Name: Asha\nSomething unrelated here.")
    assert result.fields["phone_number"].value is None
    assert result.fields["phone_number"].status is FieldStatus.MISSING
    assert result.fields["incident_date"].status is FieldStatus.MISSING


def test_empty_differs_from_missing(incident):
    result = incident.extract("Caller Name:\n\nIncident Type: Fire")
    assert result.fields["caller_name"].status is FieldStatus.EMPTY and result.fields["caller_name"].value == ""
    assert result.fields["incident_location"].status is FieldStatus.MISSING


def test_label_with_value_on_next_line(incident):
    result = incident.extract("Caller Name:\nRavi Kumar\nIncident Type: Flood")
    assert result.fields["caller_name"].value == "Ravi Kumar"


def test_conflicting_values_flagged(incident):
    result = incident.extract("Phone: 080 1111 2222\nPhone Number: 080 9999 8888")
    field = result.fields["phone_number"]
    assert field.status is FieldStatus.CONFLICT and field.value is None
    assert set(field.candidates) == {"080 1111 2222", "080 9999 8888"}
    assert not result.valid


def test_same_value_twice_is_not_a_conflict(incident):
    result = incident.extract("Phone: 080 1111 2222\nContact Number: 080 1111 2222")
    assert result.fields["phone_number"].status is FieldStatus.FOUND


def test_invalid_value_kept_verbatim(incident):
    result = incident.extract("Phone Number: call me later")
    field = result.fields["phone_number"]
    assert field.status is FieldStatus.INVALID and field.value == "call me later"
    assert field.normalized_value is None


def test_unlabeled_phone_is_uncertain(incident):
    result = incident.extract("Please call 98450 12345 for details.")
    field = result.fields["phone_number"]
    assert field.status is FieldStatus.UNCERTAIN and field.value == "98450 12345"


def test_ambiguous_date_is_uncertain_not_resolved(incident):
    field = incident.extract("Incident Date: 03/04/2024").fields["incident_date"]
    assert field.status is FieldStatus.UNCERTAIN and field.normalized_value is None


def test_low_confidence_words_make_field_uncertain(incident):
    result = incident.extract("Caller Name: Prlya Sharma", low_confidence_words=["Prlya"])
    assert result.fields["caller_name"].status is FieldStatus.UNCERTAIN


def test_inline_multiple_labels(incident):
    result = incident.extract("Caller Name: Asha Rao    Phone: 080 1234 5678")
    assert result.fields["caller_name"].value == "Asha Rao"
    assert result.fields["phone_number"].value == "080 1234 5678"


def test_short_form_template():
    template = parse_template({"name": "demo", "document_type": "incident_report",
                               "fields": {"caller_name": "string", "phone_number": "phone"}})
    assert template.fields["caller_name"].type == "string"
    result = StructuredExtractor(template).extract("caller name: Meena\nphone number - 9876543210")
    assert result.fields["caller_name"].value == "Meena"
    assert result.fields["phone_number"].value == "9876543210"


def test_custom_regex_pattern():
    template = parse_template({"name": "ref", "document_type": "letter",
                               "fields": {"reference": {"type": "string", "pattern": r"Ref(?:erence)?\s*#\s*(?P<value>[A-Z]{3}-\d+)"}}})
    assert StructuredExtractor(template).extract("Our Ref # ABC-991 dated today").fields["reference"].value == "ABC-991"


def test_required_field_missing_makes_result_invalid():
    template = TemplateStore(TEMPLATES_DIR).load("invoice")
    result = StructuredExtractor(template).extract("Total: 1,200.00")
    assert not result.valid and any("invoice_number" in i for i in result.issues)
    assert result.fields["total_amount"].normalized_value == "1200.00"


@pytest.mark.parametrize(
    "raw",
    ['{"name": "x"}', "not json", '{"name": "x", "document_type": "y", "fields": {"Bad Name": "string"}}',
     '{"name": "x", "document_type": "y", "fields": {"a": "unknowntype"}}',
     '{"name": "x", "document_type": "y", "fields": {"a": {"pattern": "(unclosed"}}}'],
)
def test_invalid_templates_rejected(raw):
    with pytest.raises(TemplateError):
        parse_template(raw)


def test_template_store_roundtrip_and_no_overwrite(tmp_path):
    store = TemplateStore(tmp_path)
    template = parse_template({"name": "my form", "document_type": "form", "fields": {"a": "string"}})
    path = store.save(template)
    assert path.parent == tmp_path.resolve()
    assert store.load("my form").fields["a"].type == "string"
    with pytest.raises(TemplateError, match="already exists"):
        store.save(template)
    store.save(template, overwrite=True)


def test_template_store_path_traversal(tmp_path):
    store = TemplateStore(tmp_path / "templates")
    with pytest.raises(TemplateError):
        store.load("../../secrets")  # sanitized to a name inside the directory, which does not exist
    with pytest.raises(UnsafePathError):
        from pathlib import Path

        from docintel.core.security import ensure_within

        ensure_within(tmp_path / "templates", Path("../x.json"))


def test_user_corrections_preserve_original(incident):
    result = incident.extract("Caller Name: Prlya Sharma\nPhone: 12")
    corrected = incident.apply_corrections(result, {"caller_name": "Priya Sharma", "phone_number": "080 1234 5678"})
    name = corrected.fields["caller_name"]
    assert name.status is FieldStatus.CORRECTED and name.original_value == "Prlya Sharma"
    assert corrected.fields["phone_number"].status is FieldStatus.CORRECTED
    bad = incident.apply_corrections(result, {"phone_number": "abc"})
    assert bad.fields["phone_number"].status is FieldStatus.INVALID


def test_string_normalization_keeps_verbatim_value(incident):
    field = incident.extract("Caller Name: Priya Sharma.").fields["caller_name"]
    assert field.value == "Priya Sharma." and field.normalized_value == "Priya Sharma"
    desc = incident.extract("Description: Car hit a pole.").fields["description"]
    assert desc.normalized_value == "Car hit a pole."  # free text keeps its punctuation


def test_validators():
    assert check_phone("+44 (20) 7946-0958").normalized == "+442079460958"
    assert not check_phone("12").ok
    assert check_date("17 March 2024").normalized == "2024-03-17"
    assert check_date("31/12/2024").normalized == "2024-12-31"  # only one valid interpretation
    assert check_date("12/05/24").uncertain
    assert not check_date("2024-02-30").ok


def test_pydantic_record_model():
    model = build_record_model("demo", {"phone": "phone", "email": "email"})
    assert model.model_validate({"phone": "080 1234 5678", "email": None}).phone == "080 1234 5678"
    with pytest.raises(Exception):
        model.model_validate({"email": "not-an-email"})
