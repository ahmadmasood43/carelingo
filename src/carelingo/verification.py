"""Compare facts from a source dictation with facts from its German note.

Facts of the same kind are aligned by entity (the same drug, the same finding
concept, the same measured quantity) and then compared attribute by attribute.
Anything in the source without a counterpart is an omission; anything in the note
without one is an unsupported addition.

Severity policy:
- critical: a fact a reader could act on wrongly — changed values, doses, status,
  polarity, laterality or resident; an omitted medication, measurement or resident;
  an invented medication.
- major: lost or invented detail that changes the clinical picture — omitted or
  invented findings and care actions, wrong site, invented numbers.
- minor: harmless elaboration, e.g. a route the dictation did not state.
"""

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from carelingo.facts import (
    FACT_LIST,
    CareMeasure,
    Fact,
    Finding,
    Intake,
    Measurement,
    Medication,
    ResidentRef,
    Score,
    numeric_values,
)
from carelingo.numbers import contains_number, unsupported_dates, unsupported_numbers
from carelingo.vocabulary import (
    canonical,
    canonical_finding,
    canonical_site,
    drugs_match,
    finding_mentioned,
    normalize_drug,
    site_laterality,
    sites_compatible,
)

Severity = Literal["critical", "major", "minor"]


@dataclass(frozen=True)
class Discrepancy:
    type: Literal["omitted", "added", "changed"]
    severity: Severity
    kind: str
    detail: str
    # The facts involved, so a checklist can describe the problem in the nurse's language.
    source: Fact | None = None
    output: Fact | None = None

    def as_record(self) -> dict:
        return {
            "type": self.type,
            "severity": self.severity,
            "kind": self.kind,
            "detail": self.detail,
            "source": self.source.model_dump(mode="json") if self.source else None,
            "output": self.output.model_dump(mode="json") if self.output else None,
        }


def _close(first: float, second: float) -> bool:
    return abs(first - second) < 1e-6


def _same_values(first: tuple[float, ...], second: tuple[float, ...]) -> bool:
    return len(first) == len(second) and all(map(_close, first, second))


_TITLES = re.compile(r"^(frau|herr|fr|hr|mrs|mr|ms|pani|pan|bewohnerin|bewohner)\.?\s+")


def _fold(name: str) -> str:
    # "Krüger" and "Kruger" are the same resident; ASR drops diacritics routinely.
    decomposed = unicodedata.normalize("NFKD", name.casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return _TITLES.sub("", plain.strip())


@dataclass(frozen=True)
class _Check:
    """One attribute comparison: returns a description when the pair disagrees."""

    severity: Severity
    compare: Callable[[Fact, Fact], str | None]
    # Disagreements about numbers can be checked against the raw texts.
    numeric: bool = False


def _differs(attribute: str, normalize=lambda value: value):
    """Both sides state the attribute and disagree."""

    def compare(source, output):
        a, b = getattr(source, attribute), getattr(output, attribute)
        if a is not None and b is not None and normalize(a) != normalize(b):
            return f"{attribute} {a} → {b}"
        return None

    return compare


def _dropped(attribute: str):
    def compare(source, output):
        if getattr(source, attribute) is not None and getattr(output, attribute) is None:
            return f"{attribute} {getattr(source, attribute)} dropped"
        return None

    return compare


def _invented(attribute: str):
    def compare(source, output):
        if getattr(source, attribute) is None and getattr(output, attribute) is not None:
            return f"{attribute} {getattr(output, attribute)} not in source"
        return None

    return compare


def _number_differs(attribute: str):
    def compare(source, output):
        a, b = getattr(source, attribute), getattr(output, attribute)
        if a is not None and b is not None and not _close(a, b):
            return f"{attribute} {a:g} → {b:g}"
        return None

    return compare


def _values_differ(source: Measurement, output: Measurement) -> str | None:
    if _same_values(source.values, output.values):
        return None
    return f"values {_slashed(source.values)} → {_slashed(output.values)}"


def _slashed(values: tuple[float, ...]) -> str:
    return "/".join(f"{value:g}" for value in values)


def _count_differs(source: Finding, output: Finding) -> str | None:
    # Extractors put stray counts (0, 1, or a pain score) on single observations;
    # only two repeated-event counts ("3-mal" vs "2-mal") are worth comparing.
    a, b = source.count, output.count
    if a is not None and b is not None and min(a, b) > 1 and a != b:
        return f"count {a} → {b}"
    return None


def _finding_polarity(source: Finding, output: Finding) -> str | None:
    _, a = canonical_finding(source.concept, source.polarity)
    _, b = canonical_finding(output.concept, output.polarity)
    return f"polarity {a} → {b}" if a != b else None


# Whether the thing happened is what a reader acts on; which non-event it was
# ("refused" by the resident vs "withheld" by staff) matters less.
_HAPPENED = {"given": True, "done": True}
# "Still needs to be done" is extracted as either; both say it has not happened.
_EQUIVALENT_STATUS = {frozenset({"planned", "not_done"})}


def _status_happened_changed(source, output) -> str | None:
    if _HAPPENED.get(source.status, False) != _HAPPENED.get(output.status, False):
        return f"status {source.status} → {output.status}"
    return None


def _status_detail_changed(source, output) -> str | None:
    a, b = source.status, output.status
    same_class = _HAPPENED.get(a, False) == _HAPPENED.get(b, False)
    if a != b and same_class and frozenset({a, b}) not in _EQUIVALENT_STATUS:
        return f"status {a} → {b}"
    return None


def _site(source, output) -> str | None:
    if not sites_compatible(source.site, output.site):
        return f"site {source.site} → {output.site}"
    return None


def _unit(source, output) -> str | None:
    a, b = canonical("unit", source.unit), canonical("unit", output.unit)
    if a is not None and b is not None and a != b:
        return f"unit {source.unit} → {output.unit}"
    return None


@dataclass(frozen=True)
class _KindRules:
    same_entity: Callable[[Fact, Fact], bool]
    checks: tuple[_Check, ...]
    omitted: Severity
    added: Severity
    label: Callable[[Fact], str]


_RULES: dict[type, _KindRules] = {
    ResidentRef: _KindRules(
        same_entity=lambda a, b: True,
        checks=(
            _Check("critical", _differs("name", _fold)),
            _Check("critical", _differs("room")),
            _Check("major", _dropped("room")),
        ),
        omitted="critical",
        added="critical",
        label=lambda f: f"resident {f.name}",
    ),
    Measurement: _KindRules(
        same_entity=lambda a, b: a.quantity == b.quantity,
        checks=(
            _Check("critical", _values_differ, numeric=True),
            _Check("major", _unit),
        ),
        omitted="critical",
        added="major",
        label=lambda f: f"{f.quantity} {_slashed(f.values)}",
    ),
    Medication: _KindRules(
        same_entity=lambda a, b: drugs_match(a.drug, b.drug),
        checks=(
            _Check("critical", _number_differs("dose"), numeric=True),
            _Check("critical", _dropped("dose")),
            _Check("critical", _status_happened_changed),
            _Check("major", _status_detail_changed),
            _Check("critical", _differs("laterality")),
            _Check("major", _differs("as_needed")),
            _Check("major", _unit),
            _Check("major", _differs("route", lambda route: canonical("route", route))),
            _Check("major", _site),
            _Check("minor", _invented("route")),
        ),
        omitted="critical",
        added="critical",
        label=lambda f: f"medication {f.drug} {f.dose or ''}{f.unit or ''}".strip(),
    ),
    Finding: _KindRules(
        same_entity=lambda a, b: (
            canonical_finding(a.concept, a.polarity)[0]
            == canonical_finding(b.concept, b.polarity)[0]
        ),
        checks=(
            _Check("critical", _finding_polarity),
            _Check("critical", _differs("laterality")),
            _Check("critical", _count_differs),
            _Check("major", _dropped("laterality")),
            _Check("major", _invented("laterality")),
            _Check("major", _site),
        ),
        omitted="major",
        added="major",
        label=lambda f: f"finding {f.concept} ({f.polarity})",
    ),
    CareMeasure: _KindRules(
        same_entity=lambda a, b: canonical("action", a.action) == canonical("action", b.action),
        checks=(
            _Check("critical", _status_happened_changed),
            _Check("major", _status_detail_changed),
        ),
        omitted="major",
        added="major",
        label=lambda f: f"measure {f.action} ({f.status})",
    ),
    Intake: _KindRules(
        same_entity=lambda a, b: (
            canonical("substance", a.substance) == canonical("substance", b.substance)
            or _close(a.amount, b.amount)
        ),
        checks=(
            _Check("critical", _number_differs("amount"), numeric=True),
            _Check("major", _unit),
        ),
        omitted="major",
        added="major",
        label=lambda f: f"intake {f.substance} {f.amount:g}{f.unit}",
    ),
    Score: _KindRules(
        same_entity=lambda a, b: canonical("scale", a.scale) == canonical("scale", b.scale),
        checks=(
            _Check("critical", _number_differs("value"), numeric=True),
            _Check("major", _number_differs("maximum")),
        ),
        omitted="major",
        added="major",
        label=lambda f: f"score {f.scale} {f.value:g}",
    ),
}

_SEVERITY_RANK = {"critical": 0, "major": 1, "minor": 2}


def _align(source: list[Fact], output: list[Fact], rules: _KindRules):
    """Greedy one-to-one alignment, preferring pairs that disagree least."""
    candidates = []
    for i, s in enumerate(source):
        for j, o in enumerate(output):
            if rules.same_entity(s, o):
                problems = sum(check.compare(s, o) is not None for check in rules.checks)
                candidates.append((problems, i, j))
    pairs, used_source, used_output = [], set(), set()
    for _, i, j in sorted(candidates):
        if i not in used_source and j not in used_output:
            pairs.append((source[i], output[j]))
            used_source.add(i)
            used_output.add(j)
    omitted = [s for i, s in enumerate(source) if i not in used_source]
    added = [o for j, o in enumerate(output) if j not in used_output]
    return pairs, omitted, added


def _normalized(facts: list[Fact]) -> list[Fact]:
    """Make equivalent ways of recording the same facts compare equal, on both sides.

    - A pain score above zero implies a pain finding; notes record either or both.
    - A side written into the site ("rechtes Knie") becomes the laterality.
    - A count of 0 or 1 carries no information and is dropped.
    - Facts asserting the same thing twice, which extractors sometimes emit, count once,
      as are care actions that only restate a medication ("Movicol geben") or a
      finding on the same side ("kein Stuhlgang" as both).
    """
    facts = [
        f.model_copy(update={"laterality": site_laterality(f.site)})
        if isinstance(f, Finding | Medication) and f.laterality is None and site_laterality(f.site)
        else f
        for f in facts
    ]
    facts = [
        f.model_copy(update={"count": None})
        if isinstance(f, Finding) and f.count is not None and f.count <= 1
        else f
        for f in facts
    ]
    has_pain_finding = any(
        isinstance(f, Finding) and canonical_finding(f.concept, f.polarity)[0] == "Schmerzen"
        for f in facts
    )
    pain_scored = any(
        isinstance(f, Score) and canonical("scale", f.scale) == "NRS" and f.value > 0 for f in facts
    )
    if pain_scored and not has_pain_finding:
        facts.append(Finding(concept="Schmerzen"))
    drugs = [f.drug for f in facts if isinstance(f, Medication)]
    findings = {_cross_kind_term(f) for f in facts if isinstance(f, Finding)}
    facts = [
        f
        for f in facts
        if not (
            isinstance(f, CareMeasure)
            and (
                any(d.lower() in f.action.lower() for d in drugs) or _cross_kind_term(f) in findings
            )
        )
    ]
    unique, seen = [], set()
    for fact in facts:
        key = _meaning(fact)
        if key not in seen:
            seen.add(key)
            unique.append(fact)
    return unique


def _meaning(fact: Fact):
    """What a fact asserts, so that "fieberfrei" and "kein Fieber" count once."""
    match fact:
        case Finding():
            concept, polarity = canonical_finding(fact.concept, fact.polarity)
            site = canonical_site(fact.site) if fact.site else None
            return ("finding", concept, polarity, fact.laterality, site, fact.count)
        case CareMeasure():
            return ("measure", canonical("action", fact.action), _HAPPENED.get(fact.status, False))
        case _:
            return fact.model_copy(update={"context": None})


_LEADING_NEGATION = re.compile(r"^(kein|keine|keinen|nicht)\s+", re.IGNORECASE)


def _cross_kind_term(fact: Finding | CareMeasure) -> str:
    # A care action "kein Stuhlgang seit 3 Tagen" names the same thing as the finding
    # "Stuhlgang" (absent); the negation is carried by status or polarity instead.
    term = fact.concept if isinstance(fact, Finding) else fact.action
    return canonical("finding", _LEADING_NEGATION.sub("", term))


def _reconcile_findings_and_measures(unmatched: dict[str, tuple[list, list]]) -> None:
    """Pair leftovers recorded as a finding on one side and a care action on the other.

    "Kein Stuhlgang" or "4-mal aufgestanden" can be read either way; the information
    is the same, so neither side should count as omitted or added.
    """
    for omitted_kind, added_kind in [("finding", "measure"), ("measure", "finding")]:
        omitted, added = unmatched[omitted_kind][0], unmatched[added_kind][1]
        for fact in list(omitted):
            match = next((o for o in added if _cross_kind_term(o) == _cross_kind_term(fact)), None)
            if match is not None:
                omitted.remove(fact)
                added.remove(match)


def _evidence_in_text(fact: Fact, text: str | None, language: str) -> bool:
    """Whether the text itself contains the fact, so a discrepancy is an extraction slip.

    Numbers alone are not enough for medications: a look-alike drug swap keeps the
    dose ("Metoprolol 500 mg" for "Metamizol 500 mg"), so the drug name must match too.
    """
    values = numeric_values(fact)
    if not values or text is None:
        return False
    if isinstance(fact, Medication) and normalize_drug(fact.drug) not in normalize_drug(text):
        return False
    return all(contains_number(text, language, value) for value in values)


def compare_facts(
    source: list[Fact],
    output: list[Fact],
    source_text: str | None = None,
    source_language: str = "de",
    output_text: str | None = None,
) -> list[Discrepancy]:
    """Discrepancies between two fact sets.

    With the raw texts given, discrepancies the text contradicts are treated as
    extraction disagreements and downgraded to minor: a numeric discrepancy whose values
    occur verbatim in the other text (the note says "2 x 3 cm" even if one extraction
    read 3 x 3), and an omitted finding the note names ("(fieberfrei)" in brackets is
    easily skipped by an extractor). Changed attributes of matched facts are not
    downgraded this way.
    """
    source, output = _normalized(source), _normalized(output)
    discrepancies, unmatched = [], {}
    for fact_type, rules in _RULES.items():
        kind = fact_type.model_fields["kind"].default
        source_facts = [f for f in source if isinstance(f, fact_type)]
        output_facts = [f for f in output if isinstance(f, fact_type)]
        pairs, omitted, added = _align(source_facts, output_facts, rules)
        unmatched[kind] = (omitted, added)
        for s, o in pairs:
            for check in rules.checks:
                problem = check.compare(s, o)
                if problem is None:
                    continue
                severity = check.severity
                if check.numeric and _evidence_in_text(s, output_text, "de"):
                    severity, problem = "minor", f"{problem} (values present in note)"
                discrepancies.append(
                    Discrepancy(
                        "changed",
                        severity,
                        kind,
                        f"{rules.label(s)}: {problem}",
                        source=s,
                        output=o,
                    )
                )
    _reconcile_findings_and_measures(unmatched)
    for fact_type, rules in _RULES.items():
        kind = fact_type.model_fields["kind"].default
        omitted, added = unmatched[kind]
        for fact in omitted:
            mentioned = (
                isinstance(fact, Finding)
                and output_text is not None
                and finding_mentioned(fact.concept, output_text)
            )
            if mentioned:
                label = f"{rules.label(fact)} (term present in note)"
                discrepancies.append(Discrepancy("omitted", "minor", kind, label, source=fact))
            elif _evidence_in_text(fact, output_text, "de"):
                label = f"{rules.label(fact)} (values present in note)"
                discrepancies.append(Discrepancy("omitted", "minor", kind, label, source=fact))
            else:
                discrepancies.append(
                    Discrepancy("omitted", rules.omitted, kind, rules.label(fact), source=fact)
                )
        for fact in added:
            if _evidence_in_text(fact, source_text, source_language):
                label = f"{rules.label(fact)} (values present in dictation)"
                discrepancies.append(Discrepancy("added", "minor", kind, label, output=fact))
            else:
                discrepancies.append(
                    Discrepancy("added", rules.added, kind, rules.label(fact), output=fact)
                )
    return sorted(discrepancies, key=lambda d: _SEVERITY_RANK[d.severity])


def verify(
    source_facts: list[Fact],
    output_facts: list[Fact],
    source_text: str,
    source_language: str,
    output_text: str,
) -> list[Discrepancy]:
    discrepancies = compare_facts(
        source_facts, output_facts, source_text, source_language, output_text
    )
    discrepancies += [
        Discrepancy("added", "major", "number", f"{value:g} not in dictation")
        for value in unsupported_numbers(source_text, source_language, output_text)
    ]
    discrepancies += [
        Discrepancy("added", "major", "date", f"{date} not in dictation")
        for date in unsupported_dates(source_text, output_text)
    ]
    return sorted(discrepancies, key=lambda d: _SEVERITY_RANK[d.severity])


UNVERIFIABLE = Discrepancy("changed", "critical", "extraction", "note could not be checked")


def reverify(record: dict, dictation: str) -> list[Discrepancy]:
    """Discrepancies of a recorded verification, recomputed from its extracted facts."""
    discrepancies = verify(
        FACT_LIST.validate_python(record["source_facts"]),
        FACT_LIST.validate_python(record["output_facts"]),
        dictation,
        record["language"],
        record["output"],
    )
    if record.get("extraction_failed"):
        discrepancies.insert(0, UNVERIFIABLE)
    return discrepancies


def needs_review(discrepancies: list[Discrepancy]) -> bool:
    return any(d.severity in ("critical", "major") for d in discrepancies)
