from pathlib import Path

import pytest

from carelingo.dataset import load_cases
from carelingo.demo import SCENARIOS, render, replay

ROOT = Path(__file__).parents[1]
CASES = {case.id: case for case in load_cases(ROOT / "data" / "cases")}


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_scenarios_replay_with_the_outcome_they_describe(scenario):
    view = replay(scenario, ROOT / "results", CASES[scenario.case])
    assert view.tier == scenario.tier
    assert view.checklist


def test_narrative_scenario_is_flagged_only_by_the_narrative_check():
    scenario = next(s for s in SCENARIOS if s.name == "narrative")
    view = replay(scenario, ROOT / "results", CASES[scenario.case])
    assert "Keine weiteren Auffälligkeiten" in view.checklist
    assert view.checklist.count("  ! ") == 1


def test_speech_scenario_shows_what_was_said_and_what_was_heard():
    scenario = next(s for s in SCENARIOS if s.name == "missed")
    view = replay(scenario, ROOT / "results", CASES[scenario.case])
    assert "3 cm" in view.said or "3 centimet" in view.said
    assert "30 cm" in view.heard
    assert "Wound size: 30 cm" in view.checklist
    assert "Heard by speech recognition:" in render(view)
