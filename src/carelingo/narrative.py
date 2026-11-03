"""Check a German note for statements its source dictation does not support.

Typed facts cannot represent reasons, sources, causes or recommendations ("nach
Rücksprache mit dem Hausarzt", "Ursache vermutlich ..."). This check asks a model
from a different family than the generator to list such statements directly.
"""

import json
from dataclasses import dataclass

from carelingo.llm import ChatModel, Completion

NARRATIVE_SYSTEM_PROMPT = """\
You compare a German nursing note with the nurse's original dictation (English or \
Polish). List every statement in the German note that the dictation does not support.

Unsupported means the note adds content the nurse did not say, for example:
- an observation, symptom or finding
- a reason, cause or explanation
- who decided or was consulted, or where information came from
- a recommendation, plan or follow-up action
- a medication route, time or date

Do not list:
- facts that the dictation states, even if worded differently, abbreviated or \
translated (e.g. "fieberfrei" for "no fever", "RR" for blood pressure)
- standard units added to values the dictation gives (mmHg, /min, °C, mg/dl)
- document structure such as headings

Quote each unsupported statement from the German note. Return an empty list when the \
note adds nothing."""

NARRATIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "unsupported": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"statement": {"type": "string"}, "why": {"type": "string"}},
                "required": ["statement", "why"],
            },
        }
    },
    "required": ["unsupported"],
}


@dataclass(frozen=True)
class NarrativeCheck:
    unsupported: tuple[dict, ...]
    completion: Completion


# Reasoning models occasionally loop; the cap includes hidden reasoning tokens.
MAX_NARRATIVE_TOKENS = 4096


def check_narrative(client: ChatModel, model: str, dictation: str, note: str) -> NarrativeCheck:
    """Unsupported statements; an unusable answer is reported as one, so it gets reviewed."""
    user = f"Dictation:\n{dictation}\n\nGerman note:\n{note}"
    completion = client.complete(
        model, NARRATIVE_SYSTEM_PROMPT, user, NARRATIVE_SCHEMA, MAX_NARRATIVE_TOKENS
    )
    try:
        unsupported = tuple(json.loads(completion.text)["unsupported"])
    except (json.JSONDecodeError, KeyError, TypeError):
        unsupported = ({"statement": "", "why": "narrative check returned no usable answer"},)
    return NarrativeCheck(unsupported, completion)
