"""Deterministic single-error variants of gold German notes, for measuring the verifier.

Each injector makes one clinically meaningful edit to a correct note, so the label
is known by construction. Controls (the unchanged note and a benign rewrite with the
same facts) measure false alarms. Injectors return None when a note offers nothing
to edit, so not every case yields every error type.
"""

import random
import re
from collections.abc import Callable
from dataclasses import dataclass

from carelingo.dataset import Case
from carelingo.facts import Intake, Measurement, Medication, Score
from carelingo.numbers import contains_number

SEED = 20261103


@dataclass(frozen=True)
class Variant:
    case_id: str
    error: str  # "none" for controls
    text: str
    description: str


def _german_number(value: float) -> str:
    return f"{value:g}".replace(".", ",")


def _replace_first(pattern: str, replacement, text: str) -> str | None:
    changed, count = re.subn(pattern, replacement, text, count=1)
    return changed if count else None


def _unused_value(value: float, case: Case, rng: random.Random) -> float:
    """A plausible wrong value that appears in none of the case's texts."""
    texts = [(case.german_reference, "de")]
    texts += [(text, language.value) for language, text in case.sources.items()]
    if value == int(value):
        steps = [max(2, round(value * 0.15)), 1, 3, 5] if value >= 10 else [1, 2, 3]
        rng.shuffle(steps)
    else:
        steps = [0.5, 0.3, 1.5]
    for step in steps:
        candidate = round(value + step, 1)
        if not any(contains_number(text, lang, candidate) for text, lang in texts):
            return candidate
    raise ValueError(f"{case.id}: no unused value near {value}")


def change_value(case: Case, rng: random.Random) -> tuple[str, str] | None:
    numeric = []
    for fact in case.facts:
        if isinstance(fact, Measurement):
            numeric.append(fact.values[0])
        elif isinstance(fact, Medication) and fact.dose is not None:
            numeric.append(fact.dose)
        elif isinstance(fact, Intake | Score):
            numeric.append(fact.amount if isinstance(fact, Intake) else fact.value)
    rng.shuffle(numeric)
    for value in numeric:
        old = _german_number(value)
        new = _german_number(_unused_value(value, case, rng))
        text = _replace_first(rf"(?<![\d,]){re.escape(old)}(?![\d,]\d)", new, case.german_reference)
        if text is not None:
            return text, f"{old} → {new}"
    return None


def remove_negation(case: Case, rng: random.Random) -> tuple[str, str] | None:
    match = re.search(r"\b([Kk]eine?n?|nicht)\s+(?!verabreicht)", case.german_reference)
    if match is None:
        return None
    text = case.german_reference[: match.start()] + case.german_reference[match.end() :]
    return text, f"removed '{match.group(1)}'"


_SIDES = {
    "rechten": "linken",
    "linken": "rechten",
    "rechte": "linke",
    "linke": "rechte",
    "Rechtes": "Linkes",
    "Linkes": "Rechtes",
    "rechts": "links",
    "links": "rechts",
}


def flip_laterality(case: Case, rng: random.Random) -> tuple[str, str] | None:
    pattern = r"\b(" + "|".join(_SIDES) + r")\b"
    match = re.search(pattern, case.german_reference)
    if match is None:
        return None
    text = _replace_first(pattern, _SIDES[match.group(1)], case.german_reference)
    return text, f"{match.group(1)} → {_SIDES[match.group(1)]}"


# Most specific first: "nicht verabreicht" before "verabreicht".
_STATUS_FLIPS = [
    (r"nicht verabreicht", "verabreicht"),
    (r"soll informiert werden", "informiert"),
    (r"(mg|IE|Hübe|Beutel|Tropfen) abgelehnt", r"\1 verabreicht"),
    (r"abgelehnt", "durchgeführt"),
    (r"geplant", "durchgeführt"),
    (r"verabreicht", "abgelehnt"),
    (r"eingenommen", "abgelehnt"),
    (r"durchgeführt", "nicht durchgeführt"),
    (r"Hausarzt informiert", "Hausarzt nicht informiert"),
]


def flip_status(case: Case, rng: random.Random) -> tuple[str, str] | None:
    for pattern, replacement in _STATUS_FLIPS:
        text = _replace_first(rf"\b{pattern}\b", replacement, case.german_reference)
        if text is not None:
            return text, f"'{pattern}' → '{replacement}'"
    return None


# Look-alike or sound-alike drugs that a reader could plausibly confuse.
LOOK_ALIKE_DRUGS = {
    "Metamizol": "Metoprolol",
    "Metoprolol": "Metamizol",
    "NovoRapid": "NovoMix",
    "Salbutamol": "Salmeterol",
    "Melperon": "Pipamperon",
    "Pipamperon": "Melperon",
    "Ramipril": "Enalapril",
    "Torasemid": "Furosemid",
    "Fentanyl": "Buprenorphin",
    "Movicol": "Laxoberal",
    "Nitrendipin": "Nifedipin",
    "Apixaban": "Rivaroxaban",
    "Latanoprost": "Timolol",
    "Metformin": "Metoprolol",
    "Pantoprazol": "Omeprazol",
    "Paracetamol": "Metamizol",
    "Tramadol": "Tilidin",
    "Lorazepam": "Lormetazepam",
    "Donepezil": "Memantin",
    "Metoclopramid": "Domperidon",
    "Lantus": "Levemir",
    "Laxoberal": "Lactulose",
}


def swap_drug(case: Case, rng: random.Random) -> tuple[str, str] | None:
    for fact in case.facts:
        if isinstance(fact, Medication) and fact.drug in LOOK_ALIKE_DRUGS:
            replacement = LOOK_ALIKE_DRUGS[fact.drug]
            text = _replace_first(re.escape(fact.drug), replacement, case.german_reference)
            if text is not None:
                return text, f"{fact.drug} → {replacement}"
    return None


def change_room(case: Case, rng: random.Random) -> tuple[str, str] | None:
    room = case.resident.room
    if room is None:
        return None
    new = str(_unused_value(float(room) + 9, case, rng)).removesuffix(".0")
    text = _replace_first(rf"Zi\. {room}\b", f"Zi. {new}", case.german_reference)
    return (text, f"room {room} → {new}") if text else None


# Sentence ends: a period after at least three non-space characters, so "Fr.", "Zi."
# and "ca." do not split.
_SENTENCE_END = re.compile(r"(?<=[^\s.]{3})\.\s+")


def omit_sentence(case: Case, rng: random.Random) -> tuple[str, str] | None:
    sentences = _SENTENCE_END.split(case.german_reference)
    if len(sentences) < 2:
        return None
    # Never drop the first sentence: it names the resident.
    index = rng.randrange(1, len(sentences))
    dropped = sentences.pop(index)
    text = ". ".join(sentence.rstrip(".") for sentence in sentences) + "."
    return text, f"dropped '{dropped.rstrip('.')}'"


INVENTED_FINDINGS = [
    ("Brustschmerzen", "Klagt zusätzlich über Brustschmerzen."),
    ("Rötung", "Zusätzlich Rötung am linken Unterschenkel."),
    ("Bewusstseinsstörung", "Zeitweise nicht ansprechbar."),
    ("Ödem", "Leichte Ödeme an beiden Knöcheln."),
    ("Juckreiz", "Klagt über Juckreiz am Rücken."),
]

INVENTED_ACTIONS = [
    "Angehörige telefonisch informiert.",
    "Vitalzeichenkontrolle in 2 Stunden geplant.",
    "Toilettengang begleitet.",
]

# Claims that are not facts about the resident's state: reasons, sources, intentions.
INVENTED_NARRATIVE = [
    "Vorgehen nach Rücksprache mit dem Hausarzt.",
    "Maßnahme auf Wunsch der Angehörigen.",
    "Laut Bewohner bereits seit gestern so.",
    "Ursache vermutlich unzureichende Flüssigkeitszufuhr.",
]


def _append(case: Case, sentence: str) -> str:
    return f"{case.german_reference.rstrip()} {sentence}"


def invent_finding(case: Case, rng: random.Random) -> tuple[str, str] | None:
    present = {getattr(f, "concept", None) for f in case.facts}
    options = [(concept, s) for concept, s in INVENTED_FINDINGS if concept not in present]
    concept, sentence = rng.choice(options)
    return _append(case, sentence), f"added '{sentence}'"


def invent_action(case: Case, rng: random.Random) -> tuple[str, str] | None:
    sentence = rng.choice(INVENTED_ACTIONS)
    return _append(case, sentence), f"added '{sentence}'"


def invent_narrative(case: Case, rng: random.Random) -> tuple[str, str] | None:
    sentence = rng.choice(INVENTED_NARRATIVE)
    return _append(case, sentence), f"added '{sentence}'"


_BENIGN = [
    (r"\bFr\. ", "Frau "),
    (r"\bHr\. ", "Herr "),
    (r"\bZi\. ", "Zimmer "),
    (r"\bRR ", "Blutdruck "),
    (r"\bBZ ", "Blutzucker "),
    (r"\bverabreicht\b", "gegeben"),
    (r"\bdurchgeführt\b", "erfolgt"),
    (r"\bTemperatur\b", "Körpertemperatur"),
]


def benign_rewrite(case: Case) -> str:
    text = case.german_reference
    for pattern, replacement in _BENIGN:
        text = re.sub(pattern, replacement, text)
    return text


INJECTORS: dict[str, Callable[[Case, random.Random], tuple[str, str] | None]] = {
    "value_change": change_value,
    "negation_removed": remove_negation,
    "laterality_flipped": flip_laterality,
    "status_flipped": flip_status,
    "drug_swapped": swap_drug,
    "room_changed": change_room,
    "sentence_omitted": omit_sentence,
    "finding_invented": invent_finding,
    "action_invented": invent_action,
    "narrative_invented": invent_narrative,
}


def variants(case: Case) -> list[Variant]:
    result = [
        Variant(case.id, "none", case.german_reference, "control: unchanged"),
        Variant(case.id, "none", benign_rewrite(case), "control: benign rewrite"),
    ]
    for error, inject in INJECTORS.items():
        edited = inject(case, random.Random(f"{SEED}:{case.id}:{error}"))
        if edited is not None and edited[0] != case.german_reference:
            result.append(Variant(case.id, error, *edited))
    return result
