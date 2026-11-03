import json

from carelingo.llm import Completion
from carelingo.pipeline import ModelChoice, PipelineConfig, process

DICTATION = "Mrs Brandt, room 12. Pulse 88."
FACTS = {
    "facts": [
        {"kind": "resident", "name": "Brandt", "room": "12"},
        {"kind": "measurement", "quantity": "pulse", "values": [88], "unit": "/min"},
    ]
}


class _Scripted:
    """Answers by role: drafts from a queue, extractions by looking at the text."""

    def __init__(self, drafts):
        self.drafts = list(drafts)

    def complete(self, model, system, user, json_schema=None, max_tokens=None):
        if json_schema is None:
            return Completion(self.drafts.pop(0), "ministral-8b-2512", 10, 10, 0.1)
        facts = json.loads(json.dumps(FACTS))
        if "Puls 98" in user:
            facts["facts"][1]["values"] = [98]
        return Completion(json.dumps(facts), "ministral-8b-2512", 10, 10, 0.1)


def _config(drafts, strong_drafts=None):
    draft = ModelChoice(_Scripted(drafts), "ministral-8b-2512")
    strong = ModelChoice(_Scripted(strong_drafts), "ministral-8b-2512") if strong_drafts else None
    return PipelineConfig(draft=draft, extractor=ModelChoice(_Scripted([]), "x"), strong=strong)


def test_clean_draft_is_ready_without_escalation():
    result = process(DICTATION, "en", _config(["Fr. Brandt, Zi. 12: Puls 88/min."]))
    assert (result.decision.tier, result.escalated) == ("ready", False)


def test_critical_draft_is_regenerated_by_the_strong_model():
    result = process(
        DICTATION,
        "en",
        _config(["Fr. Brandt, Zi. 12: Puls 98/min."], ["Fr. Brandt, Zi. 12: Puls 88/min."]),
    )
    assert (result.decision.tier, result.escalated) == ("ready", True)
    assert result.text.endswith("Puls 88/min.")


def test_critical_draft_without_strong_model_is_blocked_with_checklist():
    result = process(DICTATION, "en", _config(["Fr. Brandt, Zi. 12: Puls 98/min."]))
    assert result.decision.tier == "blocked"
    assert "Pulse: 88 /min → Pulse: 98 /min" in result.checklist
