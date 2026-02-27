from typing import TypedDict

from pydantic import BaseModel, Field

from .type_aliases import Day, OperationCard, Room, Surgeon

_global_id_counter = 0


def global_id_counter():
    """A simple global ID counter for unique identifiers."""
    global _global_id_counter
    _global_id_counter += 1
    return _global_id_counter


class Surgery(BaseModel):
    id: int = Field(
        default_factory=lambda: global_id_counter(),
        description="Unique identifier for the surgery",
    )
    operation_card_id: OperationCard
    surgeon_id: Surgeon
    expected_duration: int  # in minutes
    days_since_registration: int
    operate_by: Day
    allowed_changes: int
    changes_done: int = 0
    allowed_days_moved_plus: int
    allowed_days_moved_minus: int
    days_moved_plus: int = 0
    days_moved_minus: int = 0
    icu: bool
    ward: bool
    los_icu: int
    los_ward: int
    planned_room: Room | None = None
    planned_day: Day | None = None

    def is_planned(self) -> bool:
        """Check if the surgery is planned."""
        return self.planned_room is not None and self.planned_day is not None


class DurationCell(TypedDict):
    mu: float
    sigma: float
    gamma: float
    kappa: float
