from pathlib import Path

import pytest

from carelingo.dataset import load_cases
from carelingo.evaluation import score_output

CASES = load_cases(Path(__file__).parents[1] / "data" / "cases")

# Sources that imply a number without a digit, which the surface metric reports as an
# unsupported number in a correct German note.
IMPLICIT_NUMBERS = {("c026", "pl"): '"po godzinie" (after an hour) vs. "Nach 1 Stunde"'}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("language", ["en", "pl"])
def test_gold_reference_scores_perfectly(case, language):
    if (case.id, language) in IMPLICIT_NUMBERS:
        pytest.xfail(IMPLICIT_NUMBERS[case.id, language])
    score = score_output(case, language, case.german_reference)
    assert score.numeric_facts_preserved == score.numeric_facts
    assert score.drugs_preserved == score.drugs
    assert score.resident_preserved
    assert score.unsupported_numbers == ()


def test_changed_dose_is_detected():
    case = next(case for case in CASES if case.id == "c001")
    output = case.german_reference.replace("Metamizol 500 mg", "Metamizol 50 mg")
    score = score_output(case, "en", output)
    assert score.numeric_facts_preserved == score.numeric_facts - 1
    assert score.unsupported_numbers == (50,)


def test_uncorrected_self_correction_is_detected():
    # The dictation says "Pulse 92, sorry, 82"; keeping 92 loses the pulse fact.
    case = next(case for case in CASES if case.id == "c004")
    output = case.german_reference.replace("Puls 82/min", "Puls 92/min")
    score = score_output(case, "en", output)
    assert score.numeric_facts_preserved == score.numeric_facts - 1
