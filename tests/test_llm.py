from carelingo.extraction import extract_facts
from carelingo.llm import Completion, _answer_text


def test_answer_text_drops_reasoning_chunks():
    content = [
        {"type": "thinking", "thinking": [{"type": "text", "text": "The user wants..."}]},
        {"type": "text", "text": "Kein Fieber.\n"},
    ]
    assert _answer_text(content) == "Kein Fieber."


def test_answer_text_accepts_plain_string():
    assert _answer_text("  Kein Fieber. ") == "Kein Fieber."


def test_cost_uses_per_million_prices():
    completion = Completion("", "ministral-8b-2512", 1_000_000, 1_000_000, 1.0)
    assert completion.cost_usd == 0.30


class _FakeClient:
    def __init__(self, text):
        self.text = text

    def complete(self, model, system, user, json_schema=None, max_tokens=None):
        return Completion(self.text, "ministral-8b-2512", 10, 10, 0.1)


def test_malformed_extraction_is_marked_failed_not_raised():
    extraction = extract_facts(_FakeClient('{"facts": [{"kind": "finding", "concept'), "m", "x")
    assert extraction.failed and extraction.facts == ()
