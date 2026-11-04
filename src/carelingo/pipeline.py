"""Dictation in, German draft plus nurse checklist out.

Source-fact extraction runs in parallel with the draft, since neither depends on the
other. The default configuration drafts with a strong model and blocks any note with
a critical discrepancy for human review. With a separate strong model
configured, a critical discrepancy instead triggers one regeneration first; that
small-model-first variant was evaluated and is kept for comparison. The narrative
check (narrative.py) runs asynchronously and is not part of this call.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from carelingo.checklist import render_checklist
from carelingo.extraction import MAX_EXTRACTION_TOKENS, Extraction, extract_facts
from carelingo.llm import ChatModel, Completion
from carelingo.routing import Decision, decide
from carelingo.translate import translate_dictation
from carelingo.verification import UNVERIFIABLE, Discrepancy, verify


@dataclass(frozen=True)
class ModelChoice:
    client: ChatModel
    model: str


@dataclass(frozen=True)
class PipelineConfig:
    draft: ModelChoice
    extractor: ModelChoice
    strong: ModelChoice | None = None
    extraction_max_tokens: int = MAX_EXTRACTION_TOKENS


@dataclass(frozen=True)
class PipelineResult:
    text: str
    decision: Decision
    discrepancies: list[Discrepancy]
    checklist: str
    escalated: bool
    completions: tuple[Completion, ...]
    latency_s: float

    @property
    def cost_usd(self) -> float:
        return sum(completion.cost_usd for completion in self.completions)


def _check(
    source: Extraction, draft: str, dictation: str, language: str, config: PipelineConfig
) -> tuple[list[Discrepancy], Extraction]:
    extractor = config.extractor
    output = extract_facts(extractor.client, extractor.model, draft, config.extraction_max_tokens)
    discrepancies = verify(list(source.facts), list(output.facts), dictation, language, draft)
    if source.failed or output.failed:
        discrepancies.insert(0, UNVERIFIABLE)
    return discrepancies, output


def process(dictation: str, language: str, config: PipelineConfig) -> PipelineResult:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as pool:
        source_future = pool.submit(
            extract_facts,
            config.extractor.client,
            config.extractor.model,
            dictation,
            config.extraction_max_tokens,
        )
        draft = translate_dictation(config.draft.client, config.draft.model, dictation)
        source = source_future.result()

    discrepancies, output = _check(source, draft.text, dictation, language, config)
    completions = [source.completion, draft, output.completion]
    text, escalated = draft.text, False
    decision = decide(discrepancies, escalated=config.strong is None)

    if decision.tier == "escalate":
        strong = translate_dictation(config.strong.client, config.strong.model, dictation)
        discrepancies, output = _check(source, strong.text, dictation, language, config)
        completions += [strong, output.completion]
        text, escalated = strong.text, True
        decision = decide(discrepancies, escalated=True)

    checklist = render_checklist(
        list(source.facts), list(decision.critical + decision.to_check), language
    )
    return PipelineResult(
        text,
        decision,
        discrepancies,
        checklist,
        escalated,
        tuple(completions),
        time.perf_counter() - started,
    )
