"""Evaluation cases: a German reference note, its gold facts, and source dictations."""

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from carelingo.facts import Fact, ResidentRef


class SourceLanguage(StrEnum):
    EN = "en"
    PL = "pl"


class Case(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^c\d{3}$")
    split: Literal["dev", "test"]
    phenomena: tuple[str, ...]
    german_reference: str
    facts: tuple[Fact, ...] = Field(min_length=1)
    sources: dict[SourceLanguage, str]

    @model_validator(mode="after")
    def _single_resident(self) -> Self:
        residents = [fact for fact in self.facts if isinstance(fact, ResidentRef)]
        if len(residents) != 1:
            raise ValueError(f"{self.id}: expected exactly one resident, got {len(residents)}")
        return self

    @property
    def resident(self) -> ResidentRef:
        return next(fact for fact in self.facts if isinstance(fact, ResidentRef))


def load_cases(directory: Path) -> list[Case]:
    cases = []
    for path in sorted(directory.glob("*.yaml")):
        cases.extend(Case.model_validate(raw) for raw in yaml.safe_load(path.read_text()))
    ids = [case.id for case in cases]
    duplicates = {case_id for case_id in ids if ids.count(case_id) > 1}
    if duplicates:
        raise ValueError(f"duplicate case ids: {sorted(duplicates)}")
    return sorted(cases, key=lambda case: case.id)


def fingerprint(cases: list[Case]) -> str:
    """Content hash used to detect edits to a frozen split."""
    payload = json.dumps([case.model_dump(mode="json") for case in cases], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
