from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np

from ..type_aliases import OperationCard, Surgeon
from .params import PriorityParams


def generate_priority_data(
    frequency_data: Dict[Tuple[OperationCard, Surgeon], float],
    params: PriorityParams,
    rng: Optional[np.random.Generator] = None,
) -> Dict[OperationCard, Dict[str, int]]:
    """Generate operate_by day and allowed_changes for operation cards based on their frequencies.

    Parameters
    ----------
    frequency_data : Dict[Tuple[OperationCard, Surgeon], float]
        Dictionary mapping (operation card, surgeon) pairs to their frequencies.
        This indicates how often each surgeon performs each operation card.
        Frequencies are used to determine the planning and operation windows.
    params : PriorityParams
        Parameters for generating the priority data, including splits for different percentiles.
    rng: Optional[np.random.Generator], optional
        An optional random number generator instance for reproducibility, by default None

    Returns
    -------
    Dict[OperationCard, Dict[str, int]]
        A dictionary mapping operation cards to their operate_by day and allowed_changes
        Each entry contains a dictionary with keys "operate_by" and "allowed_changes".
        - "operate_by" is an integer indicating the day by which the surgery should be performed.
        - "allowed_changes" is an integer indicating how many changes are allowed in the plan.
    """

    if rng is None:
        rng = np.random.default_rng()

    # Aggregate frequency per operation card
    card_totals: Dict[str, float] = defaultdict(float)
    for (card, _), freq in frequency_data.items():
        card_totals[card] += freq

    # Sort cards by frequency
    sorted_cards = sorted(card_totals.items(), key=lambda x: -x[1])
    total_cards = len(sorted_cards)

    # Prepare percentiles and sort
    sorted_percentiles = sorted(params.splits.keys())

    def get_split_for_percentile(p: float) -> Dict[str, Tuple[int, int]]:
        for perc in reversed(sorted_percentiles):
            if p >= perc:
                return params.splits[perc]  # type: ignore
        return params.splits[sorted_percentiles[0]]  # type: ignore

    # Result
    priority_data: Dict[str, Dict[str, int]] = {}

    for i, (card, _) in enumerate(sorted_cards):
        percentile = (i / total_cards) * 100
        split = get_split_for_percentile(percentile)

        def sample_range(split: Tuple[int, int]) -> int:
            return int(rng.integers(split[0], split[1] + 1))

        operate_by = sample_range(split["operate_by_range"])
        allowed_changes = sample_range(split["allowed_changes_range"])

        priority_data[card] = {
            "operate_by": operate_by,
            "allowed_changes": allowed_changes,
        }

    return priority_data
