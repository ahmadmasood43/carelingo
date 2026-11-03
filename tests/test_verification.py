from pathlib import Path

import pytest

from carelingo.dataset import load_cases
from carelingo.facts import (
    CareMeasure,
    Finding,
    Measurement,
    Medication,
    Polarity,
    ResidentRef,
    Score,
)
from carelingo.verification import compare_facts, needs_review, verify
from carelingo.vocabulary import is_known

CASES = load_cases(Path(__file__).parents[1] / "data" / "cases")
RESIDENT = ResidentRef(name="Krüger", room="9")


def _changed(source, output):
    return compare_facts([RESIDENT, *source], [RESIDENT, *output])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_gold_facts_use_known_vocabulary(case):
    for fact in case.facts:
        for category, attribute in [
            ("finding", "concept"),
            ("action", "action"),
            ("site", "site"),
            ("scale", "scale"),
            ("substance", "substance"),
        ]:
            value = getattr(fact, attribute, None)
            if value is not None:
                assert is_known(category, value), f"{case.id}: {category} {value!r}"


def test_gold_facts_have_no_discrepancies_with_themselves():
    for case in CASES:
        assert compare_facts(list(case.facts), list(case.facts)) == []


def test_lost_hedge_is_critical():
    [d] = _changed(
        [Finding(concept="Verwirrtheit", polarity=Polarity.POSSIBLE)],
        [Finding(concept="desorientiert", polarity=Polarity.PRESENT)],
    )
    assert (d.type, d.severity) == ("changed", "critical")


def test_planned_action_reported_as_done_is_critical():
    [d] = _changed(
        [CareMeasure(action="Hausarzt informieren", status="planned")],
        [CareMeasure(action="Hausarzt informiert", status="done")],
    )
    assert d.severity == "critical" and "status" in d.detail


def test_wrong_site_is_major():
    [d] = _changed(
        [Finding(concept="Dekubitus", site="Sakralbereich")],
        [Finding(concept="Dekubitus", site="Sitzbein")],
    )
    assert d.severity == "major"


def test_site_synonyms_and_parents_are_compatible():
    assert (
        _changed(
            [
                Finding(concept="Dekubitus", site="Sakralbereich"),
                Finding(concept="Hämatom", site="Arm"),
            ],
            [
                Finding(concept="Druckgeschwür", site="Kreuzbein"),
                Finding(concept="Prellung", site="Oberarm"),
            ],
        )
        == []
    )


def test_inverse_terms_keep_meaning():
    # "Obstipation" (present) states the same as "Stuhlgang" (absent).
    assert (
        _changed(
            [Finding(concept="Stuhlgang", polarity=Polarity.ABSENT)],
            [Finding(concept="Obstipation", polarity=Polarity.PRESENT)],
        )
        == []
    )


def test_invented_route_is_minor_but_invented_finding_is_major():
    discrepancies = _changed(
        [Medication(drug="Metamizol", dose=500, unit="mg")],
        [Medication(drug="Metamizol", dose=500, unit="mg", route="i.v."), Finding(concept="Sturz")],
    )
    assert sorted(d.severity for d in discrepancies) == ["major", "minor"]


def test_changed_dose_and_weight_direction_are_critical():
    discrepancies = _changed(
        [
            Medication(drug="Lorazepam", dose=0.5, unit="mg"),
            Measurement(quantity="weight_gain", values=(2,), unit="kg"),
        ],
        [
            Medication(drug="Lorazepam", dose=1, unit="mg"),
            Measurement(quantity="weight_loss", values=(2,), unit="kg"),
        ],
    )
    assert {(d.type, d.severity) for d in discrepancies} == {
        ("changed", "critical"),
        ("omitted", "critical"),
        ("added", "major"),
    }


def test_resident_name_ignores_diacritics_but_not_room():
    [d] = compare_facts([RESIDENT], [ResidentRef(name="Kruger", room="19")])
    assert d.severity == "critical" and "room" in d.detail


def test_invented_timestamp_flags_review():
    discrepancies = verify(
        [RESIDENT],
        [RESIDENT],
        "Mr Krüger, room 9.",
        "en",
        "Pflegebericht 22.10.2023: Hr. Krüger, Zi. 9",
    )
    assert needs_review(discrepancies)
    assert {d.kind for d in discrepancies} == {"number", "date"}


def test_refused_versus_withheld_is_major_but_given_versus_refused_is_critical():
    [minor_change] = _changed(
        [Medication(drug="Ramipril", dose=5, unit="mg", status="refused")],
        [Medication(drug="Ramipril", dose=5, unit="mg", status="withheld")],
    )
    [serious_change] = _changed(
        [Medication(drug="Ramipril", dose=5, unit="mg", status="refused")],
        [Medication(drug="Ramipril", dose=5, unit="mg", status="given")],
    )
    assert (minor_change.severity, serious_change.severity) == ("major", "critical")


def test_planned_and_not_done_are_equivalent():
    assert (
        _changed(
            [CareMeasure(action="Hausarzt informieren", status="planned")],
            [CareMeasure(action="Hausarzt informieren", status="not_done")],
        )
        == []
    )


def test_route_spellings_across_languages_match():
    assert (
        _changed(
            [Medication(drug="NovoRapid", dose=6, unit="IE", route="podskórnie")],
            [Medication(drug="NovoRapid", dose=6, unit="E", route="s.c.")],
        )
        == []
    )


def test_pain_score_implies_pain_finding():
    assert (
        _changed(
            [Score(scale="NRS", value=4, maximum=10)],
            [Score(scale="Schmerzskala", value=4, maximum=10), Finding(concept="Schmerzen")],
        )
        == []
    )


def test_equivalent_recordings_do_not_differ():
    source = [
        Medication(drug="NovoRapid", dose=6, unit="IE"),
        Finding(concept="Schmerzen", site="Knie", laterality="right"),
        Finding(concept="Sturz", site="Badezimmer"),
        Finding(concept="Stuhlgang", polarity=Polarity.ABSENT),
    ]
    output = [
        Medication(drug="Insulin NovoRapid", dose=6, unit="IE"),
        Finding(concept="Schmerz", site="rechtes Knie"),
        Finding(concept="Sturz", site="Bad"),
        Finding(concept="Sturz", site="Bad"),
        CareMeasure(action="Stuhlgang", status="not_done"),
    ]
    assert _changed(source, output) == []


def test_only_repeat_counts_are_compared():
    assert _changed(
        [Finding(concept="Erbrechen", count=3)], [Finding(concept="Erbrechen", count=2)]
    )
    assert not _changed(
        [Finding(concept="Dekubitus", count=2)], [Finding(concept="Dekubitus", count=1)]
    )


def test_numeric_disagreement_backed_by_note_text_is_minor():
    [d] = compare_facts(
        [RESIDENT, Measurement(quantity="wound_size", values=(2, 3), unit="cm")],
        [RESIDENT, Measurement(quantity="wound_size", values=(3, 3), unit="cm")],
        output_text="Dekubitus am Kreuzbein, Kategorie 2, 2 x 3 cm.",
    )
    assert d.severity == "minor"


def test_value_change_absent_from_note_text_stays_critical():
    [d] = compare_facts(
        [RESIDENT, Measurement(quantity="pulse", values=(82,), unit="/min")],
        [RESIDENT, Measurement(quantity="pulse", values=(92,), unit="/min")],
        output_text="Puls 92/min",
    )
    assert d.severity == "critical"
