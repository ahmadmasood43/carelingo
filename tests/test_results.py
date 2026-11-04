"""The numbers quoted in docs/report.md, recomputed from the committed run files."""

import sys
from pathlib import Path

import pytest

from carelingo.cli import _speech_metrics, main

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results"
LABELS = ROOT / "data" / "annotations"


def _run(monkeypatch, capsys, *argv) -> str:
    monkeypatch.setattr(sys, "argv", ["carelingo", *map(str, argv)])
    main()
    return capsys.readouterr().out


def test_held_out_report(monkeypatch, capsys):
    out = _run(
        monkeypatch,
        capsys,
        "report",
        "--pipeline",
        RESULTS / "narrative-test-pipeline.jsonl",
        "--narrative",
        RESULTS / "narrative-test-pipeline.jsonl",
        "--reference",
        RESULTS / "narrative-test-14b.jsonl",
        "--reference-narrative",
        RESULTS / "narrative-test-14b.jsonl",
        "--labels",
        LABELS / "test-pipeline-large-low.yaml",
        LABELS / "test-ministral-14b.yaml",
    )
    assert "| notes with a clinical error | 0/48 (0%" in out
    assert "| clean notes flagged (no narrative) | 14/48 (29%" in out
    assert "| clean notes flagged (+ narrative) | 15/48 (31%" in out
    assert "| blocked | 3/48 (6%" in out
    assert "| errors caught (no narrative) | 18/23 (78%" in out
    assert "| errors caught (+ narrative) | 22/23 (96%" in out
    assert "cost/note $0.00201, latency p50 10.8 s, p95 39.3 s" in out


def test_injection(monkeypatch, capsys):
    out = _run(monkeypatch, capsys, "score-injection", RESULTS / "verify-injected-test.jsonl")
    assert "| narrative_invented | 48 | 37/48 (77%) | 10/48 |" in out
    assert "| none (controls: false alarms) | 96 | 21/96 (22%) | 2/96 |" in out


def test_speech(monkeypatch, capsys):
    out = _run(
        monkeypatch,
        capsys,
        "score-speech",
        "--pipeline",
        RESULTS / "pipeline-tts-extract-large.jsonl",
        "--labels",
        LABELS / "test-tts-pipeline.yaml",
    )
    assert "| notes with a clinical error | 20/48 (42%" in out
    assert "| errors marked ready | 6/20 (30%" in out
    assert "| clean notes flagged | 12/25 (48%" in out
    assert "| resident name differs from the dictated name | 25/48" in out


def test_typed_control(monkeypatch, capsys):
    out = _run(
        monkeypatch,
        capsys,
        "score-speech",
        "--pipeline",
        RESULTS / "pipeline-typed-extract-large.jsonl",
        "--labels",
        LABELS / "test-typed-large-extractor.yaml",
    )
    assert "| clean notes flagged | 13/48 (27%" in out
    assert "| blocked | 2/48 (4%" in out


@pytest.mark.parametrize(
    ("file", "expected"),
    [
        ("asr-fleurs-en.jsonl", "WER 6.4%"),
        ("asr-fleurs-pl.jsonl", "WER 6.2%"),
        ("asr-tts-en.jsonl", "WER 26.7% | language detected 23/24"),
        ("asr-tts-pinned-en.jsonl", "WER 22.7%"),
        ("asr-tts-pinned-pl.jsonl", "WER 15.4%"),
        ("asr-primock57-doctor.jsonl", "WER 11.8%"),
    ],
)
def test_word_error_rates(monkeypatch, capsys, file, expected):
    pytest.importorskip("jiwer")
    assert expected in _run(monkeypatch, capsys, "score-asr", RESULTS / file)


def test_speech_metrics_keep_misspelt_drugs_out_of_errors_and_clean_notes():
    rows = [
        ("c1", "en", "ready", {"drug_changed"}, False),
        ("c2", "en", "check", {"drug_misspelled"}, False),
        ("c3", "en", "check", {"language_leak"}, True),
        ("c4", "en", "ready", set(), False),
    ]
    metrics = _speech_metrics(rows)
    assert metrics["notes with a clinical error"] == (1, 4)
    assert metrics["errors marked ready"] == (1, 1)
    assert metrics["notes with only a misspelt drug"] == (1, 4)
    assert metrics["clean notes flagged"] == (1, 2)
    assert metrics["resident name differs from the dictated name"] == (1, 4)
