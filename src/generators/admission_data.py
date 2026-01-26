from typing import Dict, Optional, Tuple, Union

import numpy as np

from ..type_aliases import OperationCard, Surgeon

from .params import AdmissionParams


def generate_admission_data(
    frequency_data: Dict[Tuple[OperationCard, Surgeon], float],
    params: AdmissionParams,
    rng: Optional[np.random.Generator] = None,
) -> Dict[OperationCard, Dict[str, Union[float, Tuple[float, float]]]]:
    """
    Generate postoperative care data per operation card, including:
    - Probability of ICU admission
    - Probability of ward admission
    - ICU length-of-stay distribution (log-normal mean and std)
    - Ward length-of-stay distribution (log-normal mean and std)

    Parameters:
        frequency_data: Frequency data from which operation card frequencies are extracted.
        params: Parameters for admission data generation.
        rng: Optional random number generator for reproducibility.

    Returns:
        Dict mapping operation card ID to postoperative data.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Compute normalized frequency per operation card
    card_freq: Dict[str, float] = {}
    for (card, _), freq in frequency_data.items():
        card_freq[card] = card_freq.get(card, 0.0) + freq

    total_freq = sum(card_freq.values())
    normalized_freq = {card: freq / total_freq for card, freq in card_freq.items()}

    # Sort percentile thresholds
    sorted_thresholds = sorted(params.splits.keys())

    result: Dict[OperationCard, Dict[str, Union[float, Tuple[float, float]]]] = {}

    for card, freq in normalized_freq.items():
        # Compute percentile position
        percentile = int(100 * freq)

        # Find the nearest lower threshold
        applicable_threshold = max(t for t in sorted_thresholds if t <= percentile)

        dist_params = params.splits[applicable_threshold]

        # Sample ICU LOS log-normal parameters
        icu_mean = rng.uniform(*dist_params["mean_los_icu"])
        icu_std = rng.uniform(*dist_params["sd_los_icu"])

        # Sample Ward LOS log-normal parameters
        ward_mean = rng.uniform(*dist_params["mean_los_ward"])
        ward_std = rng.uniform(*dist_params["sd_los_ward"])

        result[card] = {
            "p_icu": float(rng.uniform(*dist_params["p_icu"])),
            "p_ward": float(rng.uniform(*dist_params["p_ward"])),
            "icu_los": (float(np.log(icu_mean)), icu_std),
            "ward_los": (float(np.log(ward_mean)), ward_std),
        }

    return result
