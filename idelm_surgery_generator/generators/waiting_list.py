from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..models import DurationCell, Surgery
from ..type_aliases import OperationCard, Surgeon


def generate_waiting_list(
    n: int,
    frequency_data: Dict[Tuple[OperationCard, Surgeon], float],
    duration_data: Dict[Tuple[OperationCard, Surgeon], DurationCell],
    priority_data: Dict[OperationCard, Dict[str, int]],
    admission_data: Dict[OperationCard, Dict[str, float]],
    rng: Optional[np.random.Generator] = None,
) -> List[Surgery]:
    """
    Generate a waiting list of surgeries by sampling from generated distributions.

    This function creates a list of Surgery objects by:
    1. Sampling (operation_card, surgeon) pairs from frequency_data
    2. Computing expected duration from lognormal parameters
    3. Extracting priority parameters (operate_by, allowed_changes)
    4. Sampling ICU/ward admission and length of stay

    The waiting list represents patients waiting for surgery, each with:
    - Surgical procedure and assigned surgeon
    - Expected operation duration
    - Priority deadline (operate_by days)
    - Rescheduling flexibility (allowed_changes)
    - Postoperative care needs (ICU/ward admission and LOS)

    Parameters
    ----------
    n : int
        Number of surgeries to generate.
    frequency_data : Dict[Tuple[OperationCard, Surgeon], float]
        Joint frequencies from generate_frequency_data(). Normalized probabilities
        for sampling (operation_card, surgeon) pairs.
    duration_data : Dict[Tuple[OperationCard, Surgeon], DurationCell]
        Lognormal duration parameters from generate_duration_data(). Each cell
        contains {"mu", "sigma", "gamma"} for 3-parameter lognormal.
    priority_data : Dict[OperationCard, Dict[str, int]]
        Priority parameters from generate_priority_data(). Each card has:
        - "operate_by": Maximum days from registration to surgery
        - "allowed_changes": Number of times surgery can be rescheduled
    admission_data : Dict[OperationCard, Dict[str, float]]
        Admission parameters from generate_admission_data(). Each card has:
        - "p_icu": Probability of ICU admission
        - "p_ward": Probability of ward admission
        - "icu_los_mu": ICU LOS lognormal μ parameter
        - "icu_los_sigma": ICU LOS lognormal σ parameter
        - "ward_los_mu": Ward LOS lognormal μ parameter
        - "ward_los_sigma": Ward LOS lognormal σ parameter
    rng : Optional[np.random.Generator]
        Random number generator for reproducibility.

    Returns
    -------
    List[Surgery]
        List of Surgery objects with sampled attributes. Each Surgery has:
        - operation_card_id, surgeon_id: Sampled pair
        - expected_duration: Integer duration in minutes
        - days_since_registration: Always 0 (all just registered)
        - operate_by: Deadline in days
        - allowed_changes: Number of reschedules allowed
        - allowed_days_moved_plus/minus: Days surgery can be moved (derived)
        - icu, ward: Boolean admission flags
        - los_icu, los_ward: Integer length of stay in days (0 if not admitted)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Normalize frequencies for sampling
    pairs = list(frequency_data.keys())
    weights = np.array([frequency_data[pair] for pair in pairs], dtype=np.float64)
    weights /= weights.sum()

    waiting_list: List[Surgery] = []

    for _ in range(n):
        # Sample (operation_card, surgeon) pair
        operation_card, surgeon = pairs[rng.choice(len(pairs), p=weights)]

        # Get duration parameters and compute expected duration
        cell = duration_data[(operation_card, surgeon)]
        mu = cell["mu"]
        sigma = cell["sigma"]
        gamma = cell["gamma"]
        # Expected value of 3-parameter lognormal: E[X] = γ + exp(μ + σ²/2)
        expected_duration = int(round(gamma + np.exp(mu + 0.5 * sigma**2)))

        # Get priority parameters
        priority_info = priority_data[operation_card]
        operate_by: int = priority_info["operate_by"]
        allowed_changes: int = priority_info["allowed_changes"]

        # Compute allowed days moved (flexibility for rescheduling)
        # These multipliers reflect asymmetric scheduling flexibility:
        # - Moving forward (delay): More flexible, can delay by ~1 week per change
        # - Moving backward (expedite): Less flexible, can only expedite by ~1 day per change
        allowed_days_moved_plus: int = allowed_changes * 7
        allowed_days_moved_minus: int = allowed_changes * 1

        # Get admission parameters
        admission_info = admission_data[operation_card]

        # Sample ICU admission
        icu: bool = rng.random() < admission_info["p_icu"]
        if icu:
            # Sample ICU LOS from lognormal, round to integer days
            icu_mu = admission_info["icu_los_mu"]
            icu_sigma = admission_info["icu_los_sigma"]
            los_icu = math.ceil(np.exp(rng.normal(icu_mu, icu_sigma)))
        else:
            los_icu = 0

        # Sample ward admission
        ward: bool = rng.random() < admission_info["p_ward"]
        if ward:
            # Sample ward LOS from lognormal, round to integer days
            ward_mu = admission_info["ward_los_mu"]
            ward_sigma = admission_info["ward_los_sigma"]
            los_ward = math.ceil(np.exp(rng.normal(ward_mu, ward_sigma)))
        else:
            los_ward = 0

        waiting_list.append(
            Surgery(
                operation_card_id=operation_card,
                surgeon_id=surgeon,
                expected_duration=expected_duration,
                days_since_registration=0,
                operate_by=operate_by,
                allowed_changes=allowed_changes,
                allowed_days_moved_plus=allowed_days_moved_plus,
                allowed_days_moved_minus=allowed_days_moved_minus,
                icu=icu,
                ward=ward,
                los_icu=los_icu,
                los_ward=los_ward,
            )
        )

    return waiting_list
