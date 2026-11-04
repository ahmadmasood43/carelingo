"""A tour of recorded held-out notes, including ones the system gets wrong.

Replays read the run files in results/ and need no API key or GPU. Where a run recorded
the extracted facts, verification, routing and the checklist are computed again by the
real code; the final-system runs recorded only their outcome, which is shown as it was.
"""

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

from carelingo.checklist import render_checklist
from carelingo.dataset import Case
from carelingo.facts import FACT_LIST
from carelingo.pipeline import PipelineResult
from carelingo.routing import Tier, decide
from carelingo.verification import reverify

_FINAL = "Final system: Mistral Large 4 draft, Ministral 14B verification, typed dictation"
_SMALL = "Reference system: Ministral 14B draft and verification, typed dictation"
_SPEECH = (
    "Final system from synthetic speech, with Mistral Large 4 as extractor "
    "(Ministral 14B was unavailable for this run)"
)


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    system: str
    file: str
    case: str
    language: str
    tier: Tier
    notice: str


SCENARIOS = (
    Scenario(
        "clean",
        "A clean note",
        _FINAL,
        "narrative-test-pipeline.jsonl",
        "c013",
        "pl",
        "ready",
        "Nothing is flagged, and the nurse still sees every extracted fact in Polish.",
    ),
    Scenario(
        "caught",
        "A real translation error, caught",
        _SMALL,
        "narrative-test-14b.jsonl",
        "c033",
        "pl",
        "check",
        "Braden 12 means high risk, but the draft says „keine Risikogefährdung“ (no risk). "
        "This draft came from the small reference model, not the final system.",
    ),
    Scenario(
        "narrative",
        "An unsupported claim only the narrative check sees",
        _SMALL,
        "narrative-test-14b.jsonl",
        "c028",
        "en",
        "check",
        "„Keine weiteren Auffälligkeiten“ was never dictated. A blanket statement like this "
        "has no typed fact to compare, so only the narrative check (qwen3:8b) flags it.",
    ),
    Scenario(
        "false-alarm",
        "A correct note, blocked",
        _FINAL,
        "narrative-test-pipeline.jsonl",
        "c029",
        "en",
        "blocked",
        "The German is right. The extractor read the dictation's double negation as confusion "
        "being present, and took „Morgenmedikation“ in the note for an invented drug. The note "
        "is blocked, and the checklist shows the nurse a wrong fact to confirm.",
    ),
    Scenario(
        "missed",
        "A recognition error that reaches the nurse as ready",
        _SPEECH,
        "pipeline-tts-extract-large.jsonl",
        "c025",
        "en",
        "ready",
        "The nurse said 3 cm; recognition heard 30 cm. Draft and dictation agree, so the "
        "verifier has nothing to flag: it compares the note with what was heard. Only the "
        "nurse reading „Wound size: 30 cm“ on the checklist can catch this.",
    ),
)

_TIERS = {
    "ready": "READY (no problems found)",
    "check": "CHECK (the nurse is asked to look at specific items)",
    "blocked": "BLOCKED (held for human review)",
}


@dataclass(frozen=True)
class NoteView:
    title: str
    system: str
    language: str
    said: str
    heard: str | None
    note: str
    tier: Tier
    checklist: str
    notice: str


def replay(scenario: Scenario, results_dir: Path, case: Case) -> NoteView:
    lines = (results_dir / scenario.file).read_text().splitlines()
    record = next(
        r
        for r in map(json.loads, lines)
        if (r["case_id"], r["language"]) == (scenario.case, scenario.language)
    )
    said = case.sources[scenario.language]
    tier, checklist = record.get("tier"), record.get("checklist")
    if "source_facts" in record:
        findings = tuple(record.get("narrative_findings", ()))
        decision = decide(reverify(record, said), findings, escalated=True)
        tier = decision.tier
        checklist = render_checklist(
            FACT_LIST.validate_python(record["source_facts"]),
            list(decision.critical + decision.to_check),
            scenario.language,
            findings,
        )
    return NoteView(
        scenario.title,
        scenario.system,
        scenario.language,
        said,
        record["dictation"] if record.get("from_transcript") else None,
        record["output"],
        tier,
        checklist,
        scenario.notice,
    )


def live_view(result: PipelineResult, dictation: str, language: str) -> NoteView:
    return NoteView(
        "Live dictation",
        "Final system, without the asynchronous narrative check",
        language,
        dictation,
        None,
        result.text,
        result.decision.tier,
        result.checklist,
        f"Cost ${result.cost_usd:.5f}, {result.latency_s:.1f} s.",
    )


def _wrapped(text: str) -> str:
    return textwrap.fill(text, 88, initial_indent="  ", subsequent_indent="  ")


def render(view: NoteView) -> str:
    parts = [f"=== {view.title} ===", view.system, "", f"Dictated ({view.language}):"]
    parts.append(_wrapped(view.said))
    if view.heard:
        parts += ["Heard by speech recognition:", _wrapped(view.heard)]
    parts += ["", "German note:", textwrap.indent(view.note, "  "), ""]
    parts += [f"Result: {_TIERS[view.tier]}", "", view.checklist, ""]
    parts += ["What to notice:", _wrapped(view.notice)]
    return "\n".join(parts)
