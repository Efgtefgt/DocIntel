"""Headless UI tests (Streamlit AppTest) with a fake OCR engine.

Covers every navigation page, dashboard buttons, upload panels, processing actions, the OCR
workspace (viewer controls, edits, exports), empty states, dependency error states, and that
OCR is not re-run by ordinary UI reruns.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from docintel.config.settings import PROJECT_ROOT  # noqa: E402
from docintel.core.file_validator import validate_upload  # noqa: E402
from docintel.core.pipeline import DocumentPipeline, PipelineOptions  # noqa: E402
from docintel.evaluation import synthetic_data as synth  # noqa: E402
from docintel.ui import state as ui_state  # noqa: E402
from docintel.ui.state import PAGE_FILES  # noqa: E402
from tests.conftest import make_factory  # noqa: E402
from tests.fakes import FakeEngine  # noqa: E402

TITLES = {"dashboard": "Dashboard", "image_ocr": "Image OCR", "pdf_tools": "PDF tools", "tables": "Table extraction",
          "structured": "Structured extraction", "intelligence": "Document intelligence", "results": "OCR results", "enhancement": "Image enhancement",
          "settings": "Settings", "about": "About"}
OCR_TEXT = "Caller Name: Asha Rao\nPhone: 080 1234 5678"


@pytest.fixture
def ui_settings(settings):
    return settings.model_copy(update={"default_engine": "fake"})


def _files() -> dict[str, bytes]:
    image = synth.render_text_image(["Caller Name: Asha Rao", "Phone: 080 1234 5678"])
    return {"scan.png": synth.image_bytes(image),
            "mixed.pdf": synth.make_mixed_pdf("Digital page text that is long enough to be used.", image)}


def _app(settings, *, uploaded: bool = False, processed: bool = False, engine: FakeEngine | None = None) -> AppTest:
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60)
    engine = engine or FakeEngine(text=OCR_TEXT)
    factory = make_factory(settings, engine)
    at.session_state["settings"] = settings
    at.session_state["engine_factory"] = factory
    at.session_state["fake_engine"] = engine
    if uploaded or processed:
        uploads, results, stats = {}, {}, ui_state.SessionStats()
        pipeline = DocumentPipeline(settings, factory)
        for name, data in _files().items():
            fid = ui_state.file_id_for(name, data)
            uploads[fid] = ui_state.UploadedDoc(fid, name, data, info=validate_upload(data, name, settings).info)
            if processed:
                results[fid] = pipeline.process(data, name, PipelineOptions(engine="fake", preprocess_preset="grayscale"))
                ui_state.add_to_stats(stats, results[fid], fid)
        at.session_state["uploads"] = uploads
        at.session_state["results"] = results
        at.session_state["stats"] = stats
    return at.run()


def _open(at: AppTest, page: str) -> AppTest:
    at.switch_page(PAGE_FILES[page]).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _title(at: AppTest) -> str:
    return at.title[0].value


def _button(at: AppTest, key: str):
    return at.button(key=key)


# --- every page -------------------------------------------------------------------------------
@pytest.mark.parametrize("state_kind", ["empty", "uploaded", "processed"])
@pytest.mark.parametrize("page", list(PAGE_FILES))
def test_every_page_renders(ui_settings, page, state_kind):
    at = _app(ui_settings, uploaded=state_kind == "uploaded", processed=state_kind == "processed")
    _open(at, page)
    assert _title(at) == TITLES[page]


def test_navigation_has_theme_selector_and_session_summary(ui_settings):
    at = _app(ui_settings, processed=True)
    assert at.sidebar.segmented_control[0].label == "Theme"
    assert any("file(s) in this session" in c.value for c in at.sidebar.caption)


# --- dashboard ----------------------------------------------------------------------------------
def test_dashboard_empty_state_shows_no_fake_numbers(ui_settings):
    at = _app(ui_settings)
    values = {m.label: m.value for m in at.metric}
    assert values["Files processed"] == "0"
    assert values["OCR confidence"] == "—" and values["Time per page"] == "—"
    assert any("No documents processed yet" in m.value for m in at.markdown)


def test_dashboard_metrics_use_session_data(ui_settings):
    at = _app(ui_settings, processed=True)
    values = {m.label: m.value for m in at.metric}
    assert values["Files processed"] == "2" and values["Pages processed"] == "3"
    assert values["OK / failed"] == "2 / 0" and values["OCR confidence"] == "95"


@pytest.mark.parametrize("key, page", [("dash_settings", "settings"), ("qa_Image to text", "image_ocr"),
                                       ("qa_PDF to text", "pdf_tools"), ("qa_Table extraction", "tables"),
                                       ("qa_Structured extraction", "structured"),
                                       ("qa_Find values", "intelligence"), ("qa_Compare OCR results", "enhancement"),
                                       ("dash_health_details", "about")])
def test_dashboard_buttons_navigate(ui_settings, key, page):
    at = _app(ui_settings)
    _button(at, key).click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert _title(at) == TITLES[page]


def test_dashboard_sample_then_process_opens_workspace(ui_settings):
    at = _app(ui_settings)
    _button(at, "sample_dashboard").click().run()
    assert len(at.session_state["uploads"]) == 1
    _button(at, "dash_process").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert _title(at) == "OCR results"
    assert at.session_state["stats"].files_processed == 1


def test_dashboard_activity_view_button(ui_settings):
    at = _app(ui_settings, processed=True)
    _button(at, "act_view_0").click().run()
    assert _title(at) == "OCR results"
    assert len(at.get("download_button")) >= 1 or at.code


def test_dependency_error_states(ui_settings, tmp_path):
    broken = ui_settings.model_copy(update={"tesseract_cmd": tmp_path / "missing" / "tesseract.exe"})
    at = _app(broken, uploaded=True, engine=FakeEngine(available=False))
    badges = " ".join(m.value for m in at.markdown)
    assert "Unavailable" in badges  # Tesseract health shows a real failure
    assert at.button(key="dash_process").disabled
    _open(at, "image_ocr")
    assert any("No OCR engine is available" in e.value for e in at.error)
    assert at.button(key="ocr_process").disabled


# --- processing pages ---------------------------------------------------------------------------
def test_image_ocr_processes_and_removes(ui_settings):
    at = _open(_app(ui_settings, uploaded=True), "image_ocr")
    _button(at, "ocr_process").click().run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    assert len(at.session_state["results"]) == 2
    _button(at, "ocr_remove_0").click().run()
    assert len(at.session_state["uploads"]) == 1


def test_upload_panel_rejects_invalid_file(ui_settings):
    at = _open(_app(ui_settings), "image_ocr")
    at.file_uploader(key="uploader_image_ocr").upload("fake.png", b"this is not an image", "image/png").run()
    assert not at.exception
    assert any("not a supported file type" in e.value for e in at.error)


def test_pdf_tools_process_pages(ui_settings):
    at = _open(_app(ui_settings, uploaded=True), "pdf_tools")
    assert {m.label for m in at.metric} >= {"Pages", "Embedded-text pages", "Pages needing OCR"}
    _button(at, "pdf_process").click().run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    assert any(e.label.startswith("Page 2") for e in at.expander)


def test_pdf_tools_searchable_pdf(ui_settings):
    from docintel.documents.searchable_pdf import check_searchable_pdf_support

    if not check_searchable_pdf_support(ui_settings).available:
        pytest.skip("searchable PDF dependencies not installed")
    at = _open(_app(ui_settings, uploaded=True), "pdf_tools")
    at.session_state["processing_options"]["languages"] = ["eng"]
    _button(at, "pdf_sp_create").click().run(timeout=180)
    assert not at.exception and not at.error, [e.value for e in at.error]
    assert any(f"searchable_{fid}" in at.session_state for fid in at.session_state["uploads"])


def test_tables_detect(ui_settings):
    at = _open(_app(ui_settings, uploaded=True), "tables")
    _button(at, "tbl_detect").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(f"tables_{fid}" in at.session_state for fid in at.session_state["uploads"])


def test_structured_extract_and_process_now(ui_settings):
    at = _open(_app(ui_settings, uploaded=True), "structured")
    _button(at, "fields_process").click().run()  # uploaded but unprocessed -> process here
    assert not at.exception
    _button(at, "fields_extract").click().run()
    result = at.session_state["structured_result"]
    assert result.fields["caller_name"].value == "Asha Rao"
    assert any(m.label == "Found" for m in at.metric)


def test_intelligence_on_processed_document(ui_settings):
    at = _open(_app(ui_settings, processed=True), "intelligence")
    values = {m.label: m.value for m in at.metric}
    assert int(values["Values found"]) >= 1 and "Labelled fields" in values
    frames = [df.value for df in at.dataframe]
    assert any("080 1234 5678" in frame.to_string() for frame in frames)
    assert len(at.get("download_button")) == 3


def test_intelligence_paste_text_masks_ids_and_opens_template(ui_settings):
    at = _open(_app(ui_settings), "intelligence")
    at.segmented_control(key="intel_source").set_value("Paste text").run()
    at.text_area(key="intel_paste").input(
        "INCIDENT REPORT\nCaller Name: Asha\nLocation: MG Road\nIncident Type: Fire\nAadhaar: 4991 1866 5246").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.value == "### Incident report" for m in at.markdown)
    table = " ".join(df.value.to_string() for df in at.dataframe)
    assert "4991 1866 5246" not in table and "5246" in table  # masked by default
    at.toggle(key="intel_mask").set_value(False).run()
    assert "4991 1866 5246" in " ".join(df.value.to_string() for df in at.dataframe)
    _button(at, "intel_tmpl_incident_report").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert _title(at) == "Structured extraction"
    assert at.selectbox(key="tmpl_select").value == "incident_report"


def test_enhancement_compare(ui_settings):
    at = _open(_app(ui_settings, uploaded=True), "enhancement")
    at.segmented_control(key="enh_mode").set_value("Individual controls").run()
    at.checkbox(key="enh_ctl_contrast").check().run()
    assert not at.exception
    _button(at, "enh_run").click().run()
    assert not at.exception and not at.error
    saved = at.session_state["cmp_results"]  # per-page copy is stored under cmp_<file>:<page>:<dpi>
    assert {row["Variant"] for row in saved["rows"]} == {"original", "smart"}


def test_settings_apply_and_clear(ui_settings):
    at = _open(_app(ui_settings, processed=True), "settings")
    at.slider[0].set_value(70)  # low-confidence threshold (first slider in the form)
    submit = next(b for b in at.button if b.label == "Apply settings")
    submit.click().run()
    assert at.session_state["settings"].low_confidence_threshold == 70
    _button(at, "set_clear").click().run()
    assert at.session_state["uploads"] == {}


# --- OCR workspace ------------------------------------------------------------------------------
def test_workspace_viewer_controls_do_not_rerun_ocr(ui_settings):
    engine = FakeEngine(text=OCR_TEXT)
    at = _app(ui_settings, processed=True, engine=engine)
    calls_after_processing = len(engine.calls)
    _open(at, "results")
    at.selectbox(key="ws_doc").select(next(k for k in at.session_state["results"] if "mixed" in at.session_state["results"][k].filename)).run()
    fid = at.session_state["last_processed"]
    _button(at, f"ws_next_{fid}").click().run()
    _button(at, f"ws_rr_{fid}").click().run()
    at.segmented_control(key=f"ws_zoom_{fid}").set_value("150%").run()
    at.segmented_control(key=f"ws_view_{fid}").set_value("Processed").run()
    _button(at, f"ws_prev_{fid}").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state[f"ws_rot_{fid}"] == 270
    assert len(engine.calls) == calls_after_processing  # viewing never re-runs OCR


def test_workspace_edits_are_kept_separately(ui_settings):
    at = _open(_app(ui_settings, processed=True), "results")
    fid = at.session_state["last_processed"]
    result = at.session_state["results"][fid]
    page = result.document.pages[0].page_number
    at.toggle(key=f"ws_edit_toggle_{fid}_{page}").set_value(True).run()
    at.text_area(key=f"ws_edit_{fid}_{page}").input("Corrected text").run()
    _button(at, f"ws_save_{fid}_{page}").click().run()
    assert at.session_state["corrections"][fid][page] == "Corrected text"
    assert at.session_state["results"][fid].document.pages[0].normalized_text != "Corrected text"  # OCR untouched
    labels = [d.label for d in at.get("download_button")]
    assert "TXT (with your edits)" in labels and "TXT" in labels


def test_workspace_tabs_reflect_document(ui_settings):
    at = _open(_app(ui_settings, processed=True), "results")
    labels = [t.label for t in at.tabs]
    assert labels[0] == "Text" and "JSON" in labels
    assert "Structured data" not in labels  # no template was used


def test_results_empty_state_offers_upload(ui_settings):
    at = _open(_app(ui_settings), "results")
    assert any("No processed documents yet" in i.value for i in at.info)
    assert at.file_uploader(key="uploader_results") is not None


# --- smart OCR pipeline in the UI -----------------------------------------------------------------
def _smart_app(ui_settings, text="Inv0ice number 48Z13 was paid"):
    engine = FakeEngine(text=text)
    at = _app(ui_settings, uploaded=True, engine=engine)
    _open(at, "image_ocr")
    _button(at, "ocr_process").click().run()  # default preset is now "smart"
    assert not at.exception, [e.value for e in at.exception]
    # Review the scanned image (the PDF's first page is embedded text and has no OCR report).
    at.session_state["last_processed"] = next(fid for fid, r in at.session_state["results"].items()
                                              if r.filename == "scan.png")
    return _open(at, "results")


def test_smart_is_default_and_quality_tab_shows_report(ui_settings):
    at = _smart_app(ui_settings)
    fid = at.session_state["last_processed"]
    page = at.session_state["results"][fid].document.pages[0]
    assert page.quality is not None and page.quality.plan  # smart plan reasons recorded
    assert page.quality.attempts and sum(a.chosen for a in page.quality.attempts) == 1
    labels = [t.label for t in at.tabs]
    assert {"Text", "Review", "Confidence", "Quality", "JSON"} <= set(labels)
    assert any(m.label == "Text coverage" for m in at.metric)


def test_review_suggestion_goes_into_edits_only(ui_settings):
    at = _smart_app(ui_settings)
    fid = at.session_state["last_processed"]
    result = at.session_state["results"][fid]
    page = result.document.pages[0]
    original = page.normalized_text
    assert "Inv0ice" in original
    button = next(b for b in at.button if b.label.startswith("Use") and "Invoice" in b.label)
    button.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Invoice" in at.session_state["corrections"][fid][page.page_number]
    assert at.session_state["results"][fid].document.pages[0].normalized_text == original  # OCR untouched


def test_accuracy_check_against_reference(ui_settings):
    at = _smart_app(ui_settings, text="Hello World")
    fid = at.session_state["last_processed"]
    page = at.session_state["results"][fid].document.pages[0].page_number
    at.text_area(key=f"ws_ref_{fid}_{page}").input("Hello World").run()
    values = {m.label: m.value for m in at.metric}
    assert values["CER (OCR output)"] == "0.000" and values["WER (OCR output)"] == "0.000"


def test_enhancement_shows_analysis_and_engine_comparison(ui_settings):
    at = _open(_app(ui_settings, uploaded=True), "enhancement")
    assert any("Smart preprocessing plan" in m.value for m in at.markdown)
    _button(at, "enh_cmp_run").click().run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    assert any("not comparable between engines" in c.value for c in at.caption)
