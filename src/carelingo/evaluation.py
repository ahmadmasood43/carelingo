"""Surface-level checks of hard facts in a German output against the gold facts.

This is a lenient first metric: a number counts as preserved if it appears anywhere
in the output, not necessarily attached to the right fact. It cannot see negation,
laterality, or medication status errors. Fact-aligned verification replaces it later.
"""

import statistics
from dataclasses import dataclass

from carelingo.dataset import Case
from carelingo.facts import Medication, numeric_values
from carelingo.numbers import contains_number, unsupported_numbers


@dataclass(frozen=True)
class HardFactScore:
    numeric_facts: int
    numeric_facts_preserved: int
    drugs: int
    drugs_preserved: int
    resident_preserved: bool
    unsupported_numbers: tuple[float, ...]


def score_output(case: Case, language: str, output: str) -> HardFactScore:
    numeric_facts = [fact for fact in case.facts if numeric_values(fact)]
    preserved = sum(
        all(contains_number(output, "de", value) for value in numeric_values(fact))
        for fact in numeric_facts
    )
    drugs = [fact.drug for fact in case.facts if isinstance(fact, Medication)]
    drugs_preserved = sum(drug.lower() in output.lower() for drug in drugs)

    gold_numbers = {value for fact in case.facts for value in numeric_values(fact)}
    unsupported = tuple(
        value
        for value in unsupported_numbers(case.sources[language], language, output)
        if value not in gold_numbers
    )

    return HardFactScore(
        numeric_facts=len(numeric_facts),
        numeric_facts_preserved=preserved,
        drugs=len(drugs),
        drugs_preserved=drugs_preserved,
        resident_preserved=case.resident.name in output,
        unsupported_numbers=unsupported,
    )


def percentile(values: list[float], q: int) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1]
