from collections import defaultdict
from typing import Dict, Optional, Tuple, Union

import numpy as np

from ..type_aliases import OperationCard, Surgeon, TimeWindow
from .params import WindowParams


def generate_window_data(
    frequency_data: Dict[Tuple[OperationCard, Surgeon], float],
    params: WindowParams,
    rng: Optional[np.random.Generator] = None,
) -> Dict[OperationCard, Dict[str, Union[TimeWindow, int]]]:
    """Generate planning and operation windows for operation cards based on their frequencies.

    Parameters
    ----------
    frequency_data : Dict[Tuple[OperationCard, Surgeon], float]
        Dictionary mapping (operation card, surgeon) pairs to their frequencies.
        This indicates how often each surgeon performs each operation card.
        Frequencies are used to determine the planning and operation windows.
    params : WindowParams
        Parameters for generating the windows, including splits for different percentiles.
    rng: Optional[np.random.Generator], optional
        An optional random number generator instance for reproducibility, by default None

    Returns
    -------
    Dict[OperationCard, Dict[str, Union[TimeWindow, int]]]
        A dictionary mapping operation cards to their planning and operation windows, and allowed changes.
        Each entry contains a dictionary with keys "planning_window", "operation_window", and "allowed_changes".
        - "planning_window" is a tuple (start, end) indicating the planning period.
        - "operation_window" is a tuple (start, end) indicating the operation period.
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
    window_data: Dict[str, Dict[str, Union[Tuple[int, int], int]]] = {}

    for i, (card, _) in enumerate(sorted_cards):
        percentile = (i / total_cards) * 100
        split = get_split_for_percentile(percentile)

        def sample_range(split: Tuple[int, int]) -> int:
            return int(rng.integers(split[0], split[1] + 1))

        plan_start = sample_range(split["plan_start_range"])
        plan_len = sample_range(split["plan_len_range"])
        op_start = sample_range(split["op_start_range"])
        op_len = sample_range(split["op_len_range"])
        allowed_changes = sample_range(split["allowed_changes_range"])

        window_data[card] = {
            "planning_window": (plan_start, plan_start + plan_len),
            "operation_window": (op_start, op_start + op_len),
            "allowed_changes": allowed_changes,
        }

    return window_data
