"""Single-pass dictation-to-German translation, used as the unchecked baseline."""

from carelingo.llm import ChatModel, Completion

BASELINE_SYSTEM_PROMPT = """\
You turn a nurse's spoken dictation into a German nursing documentation entry \
(Pflegebericht) for a German care home. The dictation may be in English or Polish and \
may contain German nursing terms.

Write the entry in German, in the concise style of German care documentation.
Preserve every clinical fact exactly: measured values and units, medication names and \
doses, whether something was given, refused or withheld, negations, left/right, times, \
and the resident's name and room. Do not add information that is not in the dictation. \
If the speaker corrects themselves, use the corrected value.

Output only the German entry."""


def translate_dictation(client: ChatModel, model: str, dictation: str) -> Completion:
    return client.complete(model, BASELINE_SYSTEM_PROMPT, dictation)
