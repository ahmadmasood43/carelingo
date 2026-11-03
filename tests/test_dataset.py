"""Integrity checks for the synthetic evaluation set.

These do not prove the sources are good translations; they catch authoring slips
where a gold fact's number, drug, or resident is missing from one of the texts.
"""

from pathlib import Path

import pytest

from carelingo.dataset import SourceLanguage, fingerprint, load_cases
from carelingo.facts import Medication, numeric_values
from carelingo.numbers import contains_number

CASES_DIR = Path(__file__).parents[1] / "data" / "cases"

# Updating this hash is a deliberate act: the test split must not drift silently
# after evaluation results have been reported against it.
FROZEN_TEST_SHA256 = "15e036dd21e4fe5f71f1f456ecbf9d7277c23f38645ab06de65f1227dbbd68f3"

# Drugs whose names are translated rather than kept as international names.
TRANSLATED_DRUG_NAMES = {"Sauerstoff": {"en": "oxygen", "pl": "tlen"}}

CASES = load_cases(CASES_DIR)


def _texts(case):
    yield "de", case.german_reference
    for language, text in case.sources.items():
        yield language.value, text


def test_both_splits_present():
    splits = {case.split for case in CASES}
    assert splits == {"dev", "test"}


def test_test_split_is_frozen():
    test_cases = [case for case in CASES if case.split == "test"]
    assert fingerprint(test_cases) == FROZEN_TEST_SHA256


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_every_source_language_present(case):
    assert set(case.sources) == set(SourceLanguage)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_numbers_appear_in_every_text(case):
    for fact in case.facts:
        for value in numeric_values(fact):
            for language, text in _texts(case):
                assert contains_number(text, language, value), (
                    f"{case.id}: {value} from {fact.kind} missing in {language}"
                )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_drugs_and_resident_appear_in_every_text(case):
    for language, text in _texts(case):
        assert case.resident.name in text, f"{case.id}: resident missing in {language}"
        for fact in case.facts:
            if not isinstance(fact, Medication):
                continue
            name = TRANSLATED_DRUG_NAMES.get(fact.drug, {}).get(language, fact.drug)
            assert name.lower() in text.lower(), f"{case.id}: {fact.drug} missing in {language}"
