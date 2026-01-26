from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np

from ..type_aliases import OperationCard, Surgeon

from .params import DurationParams


def generate_duration_data(
    frequency_data: Dict[Tuple[OperationCard, Surgeon], float],
    params: DurationParams,
    rng: Optional[np.random.Generator] = None,
) -> Dict[OperationCard, Tuple[float, float]]:
    """Generate duration distributions for operation cards based on their frequency data.
    More frequent surgeries tend to have mean and sd closer to the base mean and sd.

    Parameters
    ----------
    frequency_data : Dict[Tuple[OperationCard, Surgeon], float]
        Dictionary mapping (operation card, surgeon) pairs to their frequencies.
    params : DurationParams
        Parameters for duration generation.
    rng : Optional[np.random.Generator], optional
        Random number generator, by default None.

    Returns
    -------
    Dict[OperationCard, Tuple[float, float]]
        Dictionary mapping operation cards to their duration distribution parameters
        (mean log duration, standard deviation of log duration).
    """
    if rng is None:
        rng = np.random.default_rng()
    # Step 1: Aggregate frequency per operation card
    card_totals: Dict[OperationCard, float] = defaultdict(float)
    for (card, _), freq in frequency_data.items():
        card_totals[card] += freq

    # Step 2: Normalize frequencies (so scale is stable)
    total = sum(card_totals.values())
    normalized_freq = {card: freq / total for card, freq in card_totals.items()}

    # Step 3: Generate duration distribution parameters
    durations: Dict[OperationCard, Tuple[float, float]] = {}

    # Precompute min/max for rarity scaling
    max_freq = max(normalized_freq.values())
    min_freq = min(normalized_freq.values())

    for card, norm_freq in normalized_freq.items():
        # 0 for the most frequent card, 1 for the rarest card
        if max_freq == min_freq:
            rarity = 1.0
        else:
            rarity = (max_freq - norm_freq) / (max_freq - min_freq)

        # Scale deviation by rarity: frequent -> small deviation, rare -> up to alpha
        dev_scale = rarity  # in [0, 1]
        max_dev = params.max_dev_mean * dev_scale

        # Mean log deviation shrinks for frequent surgeries
        deviation = rng.uniform(-max_dev, max_dev)
        mean_log = params.base_mean_log + deviation

        # Std dev deviation also shrinks for frequent surgeries
        max_dev = params.max_dev_sd * dev_scale
        std_deviation = rng.uniform(-max_dev, max_dev)
        std_log = params.base_std_log + std_deviation
        std_log = np.clip(std_log, params.min_std, params.max_std)

        durations[card] = (float(mean_log), float(std_log))

    return durations
