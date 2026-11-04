import pytest

jiwer = pytest.importorskip("jiwer")

from carelingo.asr import word_error_rate  # noqa: E402


def test_wer_ignores_case_and_punctuation():
    assert word_error_rate(["Pulse 82, sorry."], ["pulse 82 sorry"]) == 0


def test_wer_counts_substitutions():
    assert word_error_rate(["blood pressure 145 over 90"], ["blood pressure 154 over 90"]) == 0.2
