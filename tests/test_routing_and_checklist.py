from pathlib import Path

import pytest

from carelingo.checklist import render_checklist, render_fact
from carelingo.dataset import load_cases
from carelingo.facts import CareMeasure, Finding, Polarity
from carelingo.routing import decide
from carelingo.verification import Discrepancy, compare_facts

CASES = load_cases(Path(__file__).parents[1] / "data" / "cases")
CRITICAL = Discrepancy("changed", "critical", "medication", "dose 5 → 50")
MAJOR = Discrepancy("added", "major", "finding", "finding Sturz (present)")
MINOR = Discrepancy("added", "minor", "medication", "route p.o. not in source")


@pytest.mark.parametrize(
    ("discrepancies", "narrative", "escalated", "tier"),
    [
        ([], (), False, "ready"),
        ([MINOR], (), False, "ready"),
        ([MAJOR], (), False, "check"),
        ([], ({"statement": "nach Rücksprache", "why": ""},), False, "check"),
        ([CRITICAL, MAJOR], (), False, "escalate"),
        ([CRITICAL], (), True, "blocked"),
        ([MAJOR], (), True, "check"),
    ],
)
def test_tiers(discrepancies, narrative, escalated, tier):
    assert decide(discrepancies, narrative, escalated).tier == tier


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("language", ["en", "pl"])
def test_every_gold_fact_renders_without_german_fallback(case, language):
    for fact in case.facts:
        line = render_fact(fact, language)
        assert "German term" not in line and "termin niemiecki" not in line, line


def test_polish_checklist_reports_status_change_in_polish():
    case = next(c for c in CASES if c.id == "c011")
    output = [
        f.model_copy(update={"status": "done"}) if isinstance(f, CareMeasure) else f
        for f in case.facts
    ]
    checklist = render_checklist(list(case.facts), compare_facts(list(case.facts), output), "pl")
    assert "Zapisano inaczej" in checklist
    assert "zaplanowano → Poinformowanie lekarza rodzinnego: wykonano" in checklist


def test_inverse_terms_render_with_their_meaning():
    line = render_fact(Finding(concept="fieberfrei", polarity=Polarity.PRESENT), "en")
    assert line == "Fever: no"
