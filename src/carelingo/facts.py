"""Typed clinical facts that must survive translation into German documentation.

Facts are language-independent: free-text fields (concepts, actions, sites) use
canonical German nursing terms so that facts extracted from an English or Polish
dictation can be compared directly with facts found in the German output.
"""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Polarity(StrEnum):
    PRESENT = "present"
    ABSENT = "absent"
    POSSIBLE = "possible"


class Laterality(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


class _Fact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # Time or circumstance qualifying the fact, e.g. "08:00", "gestern", "im Stehen".
    context: str | None = None


Quantity = Literal[
    "blood_pressure",
    "pulse",
    "temperature",
    "blood_glucose",
    "oxygen_saturation",
    "respiratory_rate",
    "weight",
    "weight_gain",
    "weight_loss",
    "wound_size",
    "urine_output",
]


class Measurement(_Fact):
    kind: Literal["measurement"] = "measurement"
    quantity: Quantity
    values: tuple[float, ...]
    unit: str

    @model_validator(mode="after")
    def _check_arity(self) -> Self:
        allowed = {"blood_pressure": (2,), "wound_size": (1, 2)}.get(self.quantity, (1,))
        if len(self.values) not in allowed:
            raise ValueError(f"{self.quantity} takes {allowed} values, got {len(self.values)}")
        return self


class Medication(_Fact):
    kind: Literal["medication"] = "medication"
    drug: str
    dose: float | None = None
    unit: str | None = None
    route: str | None = None
    site: str | None = None
    laterality: Laterality | None = None
    as_needed: bool = False
    status: Literal["given", "refused", "withheld", "planned"] = "given"


class Finding(_Fact):
    kind: Literal["finding"] = "finding"
    concept: str
    polarity: Polarity = Polarity.PRESENT
    site: str | None = None
    laterality: Laterality | None = None
    count: int | None = None


class CareMeasure(_Fact):
    kind: Literal["measure"] = "measure"
    action: str
    status: Literal["done", "refused", "not_done", "planned"] = "done"


class Intake(_Fact):
    kind: Literal["intake"] = "intake"
    substance: str
    amount: float
    unit: str


class Score(_Fact):
    kind: Literal["score"] = "score"
    scale: str
    value: float
    maximum: float | None = None


class ResidentRef(_Fact):
    kind: Literal["resident"] = "resident"
    name: str
    room: str | None = None


Fact = Annotated[
    Measurement | Medication | Finding | CareMeasure | Intake | Score | ResidentRef,
    Field(discriminator="kind"),
]


def numeric_values(fact: Fact) -> tuple[float, ...]:
    """Numbers carried by a fact that must appear unchanged in every rendering of it."""
    match fact:
        case Measurement(values=values):
            return values
        case Medication(dose=dose):
            return (dose,) if dose is not None else ()
        case Finding(count=count):
            return (count,) if count is not None else ()
        case Intake(amount=amount):
            return (amount,)
        case Score(value=value, maximum=maximum):
            return (value,) if maximum is None else (value, maximum)
        case ResidentRef(room=room):
            return (float(room),) if room is not None and room.isdigit() else ()
        case _:
            return ()
