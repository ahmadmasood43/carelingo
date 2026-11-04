"""What happens to a draft once it has been verified.

Critical discrepancies (a changed value, status, side, drug or resident) are rarely
false alarms, though not never: on the held-out split they blocked 3 of 48 notes with
no clinical error, and flagged 2 of 96 injected controls. They are trusted to stop a
note: it is blocked for human review, or, in the small-model-first variant, drafted
once more by a strong model first. Major discrepancies and narrative findings are
noisier; they are shown to the nurse as items to check.
"""

from dataclasses import dataclass
from typing import Literal

from carelingo.verification import Discrepancy

Tier = Literal["ready", "check", "escalate", "blocked"]


@dataclass(frozen=True)
class Decision:
    tier: Tier
    critical: tuple[Discrepancy, ...]
    to_check: tuple[Discrepancy, ...]
    narrative: tuple[dict, ...] = ()


def decide(
    discrepancies: list[Discrepancy],
    narrative_findings: tuple[dict, ...] = (),
    escalated: bool = False,
) -> Decision:
    """Tier for a draft. A strong-model draft that is still critical is blocked."""
    critical = tuple(d for d in discrepancies if d.severity == "critical")
    to_check = tuple(d for d in discrepancies if d.severity == "major")
    if critical:
        tier = "blocked" if escalated else "escalate"
    elif to_check or narrative_findings:
        tier = "check"
    else:
        tier = "ready"
    return Decision(tier, critical, to_check, tuple(narrative_findings))
