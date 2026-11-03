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
