from pathlib import Path

import pytest

from carelingo.dataset import load_cases
from carelingo.facts import numeric_values
from carelingo.injection import INJECTORS, benign_rewrite, variants
from carelingo.numbers import contains_number

CASES = load_cases(Path(__file__).parents[1] / "data" / "cases")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_every_error_variant_changes_the_note(case):
    for variant in variants(case):
        if variant.error != "none":
            assert variant.text != case.german_reference, variant


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_benign_rewrite_keeps_every_number(case):
    rewritten = benign_rewrite(case)
    for fact in case.facts:
        for value in numeric_values(fact):
            assert contains_number(rewritten, "de", value)


def test_variants_are_deterministic():
    assert [v.text for v in variants(CASES[0])] == [v.text for v in variants(CASES[0])]


def test_every_injector_applies_somewhere():
    produced = {v.error for case in CASES for v in variants(case)}
    assert produced >= set(INJECTORS)
