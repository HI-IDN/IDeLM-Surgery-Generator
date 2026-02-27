from __future__ import annotations

from typing import Dict

import numpy as np

from ..type_aliases import OperationCard
from .params import PriorityParams


def generate_priority_data(
    operation_cards: list[OperationCard],
    complexity_scores: np.ndarray,
    params: PriorityParams,
    rng: np.random.Generator | None = None,
) -> Dict[OperationCard, Dict[str, int]]:
    """
    Generate operate_by days and allowed_changes for operation cards based on complexity.

    Priority Assignment for Elective Surgeries
    -------------------------------------------
    Surgeries are prioritized by **operational complexity** (scheduling difficulty),
    derived from the case mix classification of Leeftink and Hans (2018). Complexity
    combines two normalized metrics:

    1. Relative duration (m_t / c): How much of an OR block the surgery consumes
    2. Coefficient of variation (s_t / m_t): Duration variability

    Each surgery's priority parameters are derived from its complexity score using
    continuous linear mappings with random noise:

        param = min + complexity * (max - min) + uniform_noise(-noise, +noise)

    More complex surgeries receive:
    - Shorter operate_by windows (must be scheduled sooner)
    - Fewer allowed_changes (less rescheduling tolerance)

    This reflects operational constraints: complex surgeries are harder to schedule
    due to longer durations, specific resource requirements, higher variability, and
    greater disruption if rescheduled.

    Parameters
    ----------
    operation_cards : list[OperationCard]
        Ordered list of operation card identifiers.
    complexity_scores : np.ndarray
        Array of shape (T,) with complexity scores in [0, 1], one per operation card
        in the same order as operation_cards. Computed by helpers.compute_complexity_scores.
    params : PriorityParams
        Parameters defining min/max bounds and noise levels for linear mappings.
    rng : np.random.Generator | None
        Random number generator for reproducibility.

    Returns
    -------
    Dict[OperationCard, Dict[str, int]]
        Dictionary mapping operation cards to their priority parameters:
        - "operate_by": Maximum days from waitlist entry to surgery date (integer)
        - "allowed_changes": Number of times the surgery can be rescheduled (integer)

    """
    if rng is None:
        rng = np.random.default_rng()

    T = len(operation_cards)
    if complexity_scores.shape != (T,):
        raise ValueError(
            f"complexity_scores must have shape ({T},), got {complexity_scores.shape}."
        )

    priority_data: Dict[OperationCard, Dict[str, int]] = {}

    for card, complexity in zip(operation_cards, complexity_scores):
        # Operate-by days: linear mapping + noise, rounded to integer
        # Higher complexity → shorter window (so we map to max - complexity * range)
        operate_by_base = params.operate_by_max - complexity * (
            params.operate_by_max - params.operate_by_min
        )
        operate_by = int(
            round(
                np.clip(
                    operate_by_base
                    + rng.uniform(-params.operate_by_noise, params.operate_by_noise),
                    params.operate_by_min,
                    params.operate_by_max,
                )
            )
        )

        # Allowed changes: linear mapping + noise, rounded to integer
        # Higher complexity → fewer changes (so we map to max - complexity * range)
        allowed_changes_base = params.allowed_changes_max - complexity * (
            params.allowed_changes_max - params.allowed_changes_min
        )
        allowed_changes = int(
            round(
                np.clip(
                    allowed_changes_base
                    + rng.uniform(
                        -params.allowed_changes_noise, params.allowed_changes_noise
                    ),
                    params.allowed_changes_min,
                    params.allowed_changes_max,
                )
            )
        )

        priority_data[card] = {
            "operate_by": operate_by,
            "allowed_changes": allowed_changes,
        }

    return priority_data
