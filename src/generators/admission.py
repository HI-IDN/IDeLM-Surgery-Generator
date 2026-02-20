from __future__ import annotations

from typing import Dict

import numpy as np

from ..type_aliases import OperationCard
from .helpers import rng_or_default
from .params import AdmissionParams


def generate_admission_data(
    operation_cards: list[OperationCard],
    complexity_scores: np.ndarray,
    params: AdmissionParams,
    rng: np.random.Generator | None = None,
) -> Dict[OperationCard, Dict[str, float]]:
    """
    Generate postoperative care parameters based on surgical complexity.

    Postoperative Resource Needs
    -----------------------------
    Each surgery's postoperative parameters are derived from its operational complexity
    score using continuous linear mappings with random noise. While operational
    complexity (duration + variability) is not identical to clinical complexity,
    procedure duration is a validated predictor of ICU admission and hospital length
    of stay in the literature.

    Parameters are mapped as:
        param = min + complexity * (max - min) + uniform_noise(-noise, +noise)

    Length of Stay Distributions
    -----------------------------
    Both ICU and ward length of stay follow lognormal distributions with parameters
    (μ, σ) in log-space. The relationship to real-space statistics:
    - Median LOS = exp(μ)
    - Mean LOS = exp(μ + σ²/2)
    - Std dev = sqrt((exp(σ²) - 1) * exp(2μ + σ²))

    Example: μ=1.0, σ=0.4 → median ≈ 2.7 days, mean ≈ 3.0 days

    To sample actual LOS in days:
        days = int(round(exp(N(μ, σ))))  # Round to nearest integer

    Parameters
    ----------
    operation_cards : list[OperationCard]
        Ordered list of operation card identifiers.
    complexity_scores : np.ndarray
        Array of shape (T,) with complexity scores in [0, 1], one per operation card.
        Computed by helpers.compute_complexity_scores.
    params : AdmissionParams
        Parameters defining min/max bounds and noise levels for linear mappings.
    rng : np.random.Generator | None
        Random number generator for reproducibility.

    Returns
    -------
    Dict[OperationCard, Dict[str, float]]
        Dictionary mapping operation cards to postoperative care parameters:
        - "p_icu": Probability of ICU admission (float in [0,1])
        - "p_ward": Probability of ward admission (float in [0,1])
        - "icu_los_mu": ICU LOS lognormal location parameter (log-space)
        - "icu_los_sigma": ICU LOS lognormal scale parameter (log-space)
        - "ward_los_mu": Ward LOS lognormal location parameter (log-space)
        - "ward_los_sigma": Ward LOS lognormal scale parameter (log-space)
    """
    rng = rng_or_default(rng)

    T = len(operation_cards)
    if complexity_scores.shape != (T,):
        raise ValueError(
            f"complexity_scores must have shape ({T},), got {complexity_scores.shape}."
        )

    result: Dict[OperationCard, Dict[str, float]] = {}

    for card, complexity in zip(operation_cards, complexity_scores):
        # ICU admission probability: linear mapping + noise, clipped to [0, 1]
        p_icu_base = params.p_icu_min + complexity * (
            params.p_icu_max - params.p_icu_min
        )
        p_icu = float(
            np.clip(
                p_icu_base + rng.uniform(-params.p_icu_noise, params.p_icu_noise),
                0.0,
                1.0,
            )
        )

        # Ward admission probability: linear mapping + noise, clipped to [0, 1]
        p_ward_base = params.p_ward_min + complexity * (
            params.p_ward_max - params.p_ward_min
        )
        p_ward = float(
            np.clip(
                p_ward_base + rng.uniform(-params.p_ward_noise, params.p_ward_noise),
                0.0,
                1.0,
            )
        )

        # ICU LOS lognormal μ: linear mapping + noise
        icu_mu_base = params.icu_los_mu_min + complexity * (
            params.icu_los_mu_max - params.icu_los_mu_min
        )
        icu_mu = float(
            icu_mu_base + rng.uniform(-params.icu_los_mu_noise, params.icu_los_mu_noise)
        )

        # ICU LOS lognormal σ: linear mapping + noise, ensure positive
        icu_sigma_base = params.icu_los_sigma_min + complexity * (
            params.icu_los_sigma_max - params.icu_los_sigma_min
        )
        icu_sigma = float(
            max(
                0.01,
                icu_sigma_base
                + rng.uniform(-params.icu_los_sigma_noise, params.icu_los_sigma_noise),
            )
        )

        # Ward LOS lognormal μ: linear mapping + noise
        ward_mu_base = params.ward_los_mu_min + complexity * (
            params.ward_los_mu_max - params.ward_los_mu_min
        )
        ward_mu = float(
            ward_mu_base
            + rng.uniform(-params.ward_los_mu_noise, params.ward_los_mu_noise)
        )

        # Ward LOS lognormal σ: linear mapping + noise, ensure positive
        ward_sigma_base = params.ward_los_sigma_min + complexity * (
            params.ward_los_sigma_max - params.ward_los_sigma_min
        )
        ward_sigma = float(
            max(
                0.01,
                ward_sigma_base
                + rng.uniform(
                    -params.ward_los_sigma_noise, params.ward_los_sigma_noise
                ),
            )
        )

        result[card] = {
            "p_icu": p_icu,
            "p_ward": p_ward,
            "icu_los_mu": icu_mu,
            "icu_los_sigma": icu_sigma,
            "ward_los_mu": ward_mu,
            "ward_los_sigma": ward_sigma,
        }

    return result
