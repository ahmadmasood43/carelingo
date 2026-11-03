"""Replay recorded runs under different routing policies.

Every step of the pipeline has already been run and recorded per dictation (source
extraction, drafts from each model, output extraction for each draft), so a policy
can be evaluated by choosing which recorded steps it would have taken. Latency
assumes source extraction runs in parallel with the first draft, as the pipeline does.
"""

import statistics
from dataclasses import dataclass

from carelingo.routing import decide

CLINICAL_ERRORS = {"unsupported_claim", "status_changed", "hedge_lost", "site_changed", "prn_lost"}


@dataclass(frozen=True)
class Step:
    """A recorded draft and its verification for one dictation."""

    cost_usd: float
    latency_s: float
    check_cost_usd: float
    check_latency_s: float
    discrepancies: list
    errors: frozenset[str]
    # Asynchronous narrative findings; they arrive too late to trigger escalation, so
    # they only add items to check.
    narrative: tuple[dict, ...] = ()


@dataclass(frozen=True)
class Outcome:
    cost_usd: float
    latency_s: float
    escalated: bool
    tier: str
    errors: frozenset[str]

    @property
    def has_clinical_error(self) -> bool:
        return bool(self.errors & CLINICAL_ERRORS)

    @property
    def unflagged_error(self) -> bool:
        """A clinical error that reaches the nurse with nothing pointing at it."""
        return self.has_clinical_error and self.tier == "ready"


def run_policy(
    policy: str,
    first: Step,
    strong: Step | None,
    source_cost_usd: float,
    source_latency_s: float,
) -> Outcome:
    """policy: "unverified", "verified", "escalate_critical" or "escalate_any"."""
    if policy == "unverified":
        return Outcome(first.cost_usd, first.latency_s, False, "ready", first.errors)

    cost = source_cost_usd + first.cost_usd + first.check_cost_usd
    latency = max(source_latency_s, first.latency_s) + first.check_latency_s
    decision = decide(first.discrepancies)
    escalate = (policy == "escalate_critical" and decision.tier == "escalate") or (
        policy == "escalate_any" and decision.tier in ("escalate", "check")
    )
    if not escalate or strong is None:
        final = decide(first.discrepancies, first.narrative, escalated=policy == "verified")
        return Outcome(cost, latency, False, final.tier, first.errors)

    cost += strong.cost_usd + strong.check_cost_usd
    latency += strong.latency_s + strong.check_latency_s
    final = decide(strong.discrepancies, strong.narrative, escalated=True)
    return Outcome(cost, latency, True, final.tier, strong.errors)


def summarize(outcomes: list[Outcome]) -> dict:
    latencies = sorted(o.latency_s for o in outcomes)
    n = len(outcomes)
    return {
        "n": n,
        "escalated": sum(o.escalated for o in outcomes),
        "clinical_errors": sum(o.has_clinical_error for o in outcomes),
        "unflagged_errors": sum(o.unflagged_error for o in outcomes),
        "needs_attention": sum(o.tier in ("check", "blocked") for o in outcomes),
        "blocked": sum(o.tier == "blocked" for o in outcomes),
        "cost_per_note_usd": sum(o.cost_usd for o in outcomes) / n,
        "latency_p50_s": statistics.median(latencies),
        "latency_max_s": latencies[-1],
    }
