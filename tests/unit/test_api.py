"""FastAPI validation and error handling (fake OCR engine)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from docintel.api.main import create_app  # noqa: E402
from tests.conftest import make_factory  # noqa: E402
from tests.fakes import FakeEngine  # noqa: E402


@pytest.fixture
def client(settings):
    engine = FakeEngine(text="Caller Name: Asha Rao\nPhone: 080 1234 5678")
    return TestClient(create_app(settings, make_factory(settings, engine)))


def test_health_lists_engines(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert {"tesseract", "paddle", "fake"} <= {e["name"] for e in body["engines"]}


def test_process_image(client, png_bytes):
    response = client.post("/process", files={"file": ("scan.png", png_bytes, "image/png")},
                           data={"engine": "fake", "preprocess": "original", "template": "incident_report"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["succeeded"]
    assert body["document"]["pages"][0]["normalized_text"].startswith("Caller Name")
    assert body["structured"]["fields"]["caller_name"]["value"] == "Asha Rao"


def test_invalid_file_reports_validation_error(client):
    response = client.post("/process", files={"file": ("x.png", b"plain text", "image/png")}, data={"engine": "fake"})
    assert response.status_code == 200
    body = response.json()
    assert not body["succeeded"] and body["errors"][0]["code"] == "unsupported_format"


def test_unknown_preset_rejected(client, png_bytes):
    response = client.post("/process", files={"file": ("a.png", png_bytes)}, data={"preprocess": "bogus"})
    assert response.status_code == 422 and response.json()["code"] == "invalid_input"


@pytest.mark.parametrize("field, value", [("languages", "eng;del"), ("psm", "99"), ("pdf_mode", "nope")])
def test_form_validation(client, png_bytes, field, value):
    response = client.post("/process", files={"file": ("a.png", png_bytes)}, data={field: value})
    assert response.status_code == 422 and response.json()["code"] == "invalid_request"


def test_missing_file_rejected(client):
    assert client.post("/process", data={"engine": "fake"}).status_code == 422


def test_request_size_limit(settings, png_bytes):
    small = settings.model_copy(update={"api_max_request_mb": 0.0005})
    client = TestClient(create_app(small, make_factory(small)))
    response = client.post("/process", files={"file": ("a.png", png_bytes + b"\0" * 200_000)}, data={"engine": "fake"})
    assert response.status_code == 413


def test_unknown_template_is_404(client):
    response = client.post("/extract", json={"text": "x", "template": "nope"})
    assert response.status_code == 404


def test_extract_endpoint(client):
    response = client.post("/extract", json={"text": "Phone Number: 98450 12345", "template": "incident_report"})
    assert response.status_code == 200
    assert response.json()["fields"]["phone_number"]["normalized_value"] == "9845012345"


def test_templates_endpoint(client):
    assert "incident_report" in client.get("/templates").json()
