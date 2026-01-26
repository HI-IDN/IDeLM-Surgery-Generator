from typing import Dict, List, Optional, Tuple

import numpy as np

from ..type_aliases import OperationCard, Surgeon

from .params import FrequencyParams

# Rewrite only the grouping logic (step 2) in the provided function


def create_card_groups(
    card_freq: Dict[OperationCard, float], n_groups: int
) -> Dict[OperationCard, int]:
    """
    Assign operation cards to frequency-based groups of roughly equal size.
    Groups are ordered so that:
    - Group 0 contains the most frequent cards
    - Group N-1 contains the least frequent cards

    Parameters:
        card_freq: Dict mapping operation card IDs to their global frequency
        n_groups: Number of groups

    Returns:
        Dict mapping operation card ID to group ID
    """
    sorted_cards = sorted(card_freq.items(), key=lambda x: -x[1])  # most freq → least
    total_cards = len(sorted_cards)

    # Compute uniform group sizes
    base_size = total_cards // n_groups
    remainder = total_cards % n_groups
    group_sizes = [base_size + (1 if i < remainder else 0) for i in range(n_groups)]

    # Assign cards to groups
    card_groups = {}
    idx = 0
    for group_id, size in enumerate(group_sizes):
        for _ in range(size):
            card, _ = sorted_cards[idx]
            card_groups[card] = group_id
            idx += 1

    return card_groups


def generate_frequency_data(
    num_operation_cards: int,
    num_surgeons: int,
    params: FrequencyParams,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Dict[Tuple[OperationCard, Surgeon], float], Dict[OperationCard, int]]:
    """Generate frequency data for operation cards and surgeons.

    Parameters
    ----------
    num_operation_cards : int
        The number of operation cards to generate frequency data for.
    num_surgeons : int
        The number of surgeons to generate frequency data for.
    params : FrequencyParams
    rng: Optional[np.random.Generator], optional
        An optional random number generator instance for reproducibility, by default None

    Returns
    -------
    Tuple[Dict[Tuple[OperationCard, Surgeon], float], Dict[Surgeon, List[int]]]
        A dictionary mapping (operation card, surgeon) pairs to their frequencies,
        and a dictionary mapping operation cards to their group IDs.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Step 1: Generate global operation card frequencies (Zipf)
    operation_cards = [f"Operation_{i}" for i in range(num_operation_cards)]
    raw_freq = rng.zipf(params.skewness, size=num_operation_cards)
    global_freq = raw_freq / raw_freq.sum()
    card_freq = dict(zip(operation_cards, global_freq))

    # Step 2: Group operation cards by frequency
    card_groups = create_card_groups(card_freq, params.n_groups)

    # Apply mixing
    for card in operation_cards:
        if rng.random() < params.mixing_rate:
            card_groups[card] = int(rng.integers(1, params.n_groups))

    # Step 3: Assign surgeons to groups and give activity profiles
    surgeons = list(range(num_surgeons))
    surgeon_groups: Dict[Surgeon, List[int]] = {}
    surgeon_activity: Dict[Surgeon, float] = {}

    for s in surgeons:
        groups = rng.choice(
            range(params.n_groups),
            size=rng.integers(1, params.max_groups_per_surgeon + 1),
            replace=False,
        ).tolist()
        surgeon_groups[s] = groups
        surgeon_activity[s] = float(
            rng.lognormal(params.surgeon_activity_mu, params.surgeon_activity_sigma)
        )

    # Step 4: Distribute frequencies to surgeons
    frequency: Dict[Tuple[OperationCard, Surgeon], float] = {}

    for card, card_group in card_groups.items():
        eligible_surgeons = [s for s in surgeons if card_group in surgeon_groups[s]]
        if not eligible_surgeons:
            continue

        weights = {
            s: surgeon_activity[s] * rng.uniform(0.8, 1.2) for s in eligible_surgeons
        }
        total_weight = sum(weights.values())
        normalized_weights = {s: w / total_weight for s, w in weights.items()}

        for s in eligible_surgeons:
            frequency[(card, s)] = float(card_freq[card] * normalized_weights[s])

    return frequency, card_groups
