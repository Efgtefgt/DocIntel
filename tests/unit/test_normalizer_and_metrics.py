"""Text normalization and CER/WER evaluation."""

from __future__ import annotations

import pytest

from docintel.core.pipeline import DocumentPipeline
from docintel.evaluation.benchmark import find_samples, run_benchmark
from docintel.evaluation.metrics import EvalNormalization, cer, edit_distance, wer
from docintel.evaluation.synthetic_data import write_benchmark_dataset
from docintel.extraction.text_normalizer import NormalizationOptions, normalize_text
from tests.conftest import make_factory
from tests.fakes import FakeEngine


def test_normalization_preserves_punctuation_and_numbers():
    text = "Total:  $1,250.00 (incl. 18% GST) — ref #A-12/B!\r\n\r\n\r\n\r\nNext   line   "
    result = normalize_text(text)
    assert result.text == "Total: $1,250.00 (incl. 18% GST) — ref #A-12/B!\n\nNext line"
    assert "repeated spaces collapsed" in result.changes and "blank lines reduced" in result.changes


def test_normalization_reports_no_changes_for_clean_text():
    assert normalize_text("Clean text.\nSecond line.").changes == []


def test_optional_rules_are_off_by_default():
    assert normalize_text("infor-\nmation “quoted”").text == "infor-\nmation “quoted”"
    on = normalize_text("infor-\nmation “quoted”", NormalizationOptions(join_hyphenated_lines=True, straighten_quotes=True))
    assert on.text == 'information "quoted"'


def test_nfkc_folds_ligatures_only_when_requested():
    assert normalize_text("ﬁle").text == "ﬁle"
    assert normalize_text("ﬁle", NormalizationOptions(unicode_form="NFKC")).text == "file"


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance(["a", "b"], ["a", "b"]) == 0


def test_cer_and_wer():
    assert cer("hello world", "hello world") == 0
    assert cer("abcd", "abed") == pytest.approx(0.25)
    assert wer("the cat sat", "the cat sat down") == pytest.approx(1 / 3)
    assert wer("", "") == 0 and cer("", "x") == 1.0


def test_evaluation_normalization_rules():
    assert cer("Hello,\nWorld", "hello world") > 0
    loose = EvalNormalization(lowercase=True, remove_punctuation=True)
    assert cer("Hello,\nWorld", "hello world", loose) == 0


def test_benchmark_with_fake_engine(tmp_path, settings):
    write_benchmark_dataset(tmp_path)
    assert len(find_samples(tmp_path)) == 5
    engine = FakeEngine(text="The quick brown fox jumps over the lazy dog.")
    pipeline = DocumentPipeline(settings, make_factory(settings, engine))
    report = run_benchmark(tmp_path, settings, engine="fake", preset="original", pipeline=pipeline)
    summary = report.summary
    assert summary["samples"] == 5 and summary["succeeded"] == 5
    clean = next(s for s in report.samples if s.sample == "clean_paragraph.png")
    assert 0 < clean.cer < 1
