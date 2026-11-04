"""Structured extraction of typed facts from a dictation or a German note."""

import json
import types
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Union, get_args, get_origin

from pydantic import TypeAdapter, ValidationError

from carelingo.facts import (
    CareMeasure,
    Fact,
    Finding,
    Intake,
    Measurement,
    Medication,
    ResidentRef,
    Score,
)
from carelingo.llm import ChatModel, Completion
from carelingo.vocabulary import CARE_ACTIONS, FINDINGS, SCALES, SITES, SUBSTANCES, UNITS

FACT_MODELS = (Measurement, Medication, Finding, CareMeasure, Intake, Score, ResidentRef)

EXTRACTION_SYSTEM_PROMPT = f"""\
You extract clinical facts from nursing documentation. The text is either a nurse's \
spoken dictation (English, Polish or German) or a written German nursing note.

Return every fact stated in the text, one object per fact, using these kinds:
- resident: the resident's surname and room number. Exactly one.
- measurement: a measured value. Blood pressure has two values (systolic, diastolic).
- medication: a drug or oxygen with dose and unit as stated. status is "given", \
"refused" (by the resident), "withheld" (not given by staff) or "planned". Set \
as_needed only if the text says it is on-demand (Bedarfsmedikation, PRN). Set route \
only if the text states it.
- finding: an observation or event, including a fall (Sturz). polarity is "absent" \
when negated ("no fever", "kein Fieber") and "possible" when the text hedges \
("maybe", "evtl.", "möglicherweise"). Set count only for how often an event \
happened ("3 times", "2-mal"); otherwise leave it empty.
- measure: a care action. status is "done", "refused", "not_done" or "planned" \
(including "still needs to be done").
- intake: an amount the resident drank or ate.
- score: a scale value. Every pain rating such as "6 out of 10" or "NRS 6/10" is a \
score (scale NRS, value 6, maximum 10), in addition to the pain finding.

Rules:
- Write concept, action, site, scale, substance and unit in German, using these \
terms where one fits; otherwise use a short German term.
  Findings: {", ".join(FINDINGS)}
  Care actions: {", ".join(CARE_ACTIONS)}
  Sites: {", ".join(SITES)}
  Scales: {", ".join(SCALES)}
  Intake substances: {", ".join(SUBSTANCES)}
  Units: {", ".join(UNITS)}
- If the speaker corrects a value, extract only the corrected value.
- Extract only what the text states. Do not infer causes, times or findings. Leave \
route empty unless the text names one (s.c., i.v., oral, ...).
- Extract each fact once.
- Put a time or circumstance that qualifies a fact in context ("03:00", "gestern", \
"im Stehen").
- Ignore document headings, placeholders and signature lines."""


def _value_schema(annotation) -> dict:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        (inner,) = [arg for arg in get_args(annotation) if arg is not type(None)]
        return _value_schema(inner)
    if origin is Literal:
        values = list(get_args(annotation))
        return (
            {"type": "string", "const": values[0]}
            if len(values) == 1
            else {
                "type": "string",
                "enum": values,
            }
        )
    if origin is tuple:
        return {"type": "array", "items": _value_schema(get_args(annotation)[0])}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {"type": "string", "enum": [member.value for member in annotation]}
    return {float: {"type": "number"}, int: {"type": "integer"}, bool: {"type": "boolean"}}.get(
        annotation, {"type": "string"}
    )


# State fields have defaults in the fact models, but the extractor must decide them
# explicitly: an omitted status would silently read as "given", a polarity as "present".
_STATE_FIELDS = {"status", "polarity", "as_needed"}


def _fact_schema(model) -> dict:
    fields = model.model_fields
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "kind",
            *[n for n, f in fields.items() if f.is_required() or n in _STATE_FIELDS],
        ],
        "properties": {name: _value_schema(field.annotation) for name, field in fields.items()},
    }


FACTS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["facts"],
    "properties": {
        "facts": {"type": "array", "items": {"anyOf": [_fact_schema(m) for m in FACT_MODELS]}}
    },
}

_FACT_ADAPTER = TypeAdapter(Fact)


@dataclass(frozen=True)
class Extraction:
    facts: tuple[Fact, ...]
    # Items the model returned that do not validate, e.g. blood pressure with one value.
    rejected: tuple[dict, ...]
    completion: Completion
    # The response was not valid JSON (e.g. a repetition loop cut off at max_tokens).
    # Nothing can be verified, so the note must go to human review.
    failed: bool = False


# A short note has at most a few dozen facts, so this bound only stops runaway
# repetition, which otherwise costs time and money before failing anyway.
MAX_EXTRACTION_TOKENS = 2000


def extract_facts(
    client: ChatModel, model: str, text: str, max_tokens: int = MAX_EXTRACTION_TOKENS
) -> Extraction:
    """Facts in `text`. Reasoning models count their reasoning in `max_tokens`, so they
    need a much larger cap than the default."""
    completion = client.complete(model, EXTRACTION_SYSTEM_PROMPT, text, FACTS_SCHEMA, max_tokens)
    try:
        items = json.loads(completion.text)["facts"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return Extraction((), (), completion, failed=True)
    facts, rejected = [], []
    for item in items:
        try:
            facts.append(_FACT_ADAPTER.validate_python(item))
        except ValidationError:
            rejected.append(item)
    return Extraction(tuple(facts), tuple(rejected), completion)
