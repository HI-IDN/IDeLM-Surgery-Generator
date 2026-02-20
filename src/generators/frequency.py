from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..type_aliases import OperationCard, Surgeon
from .params import FrequencyParams


def generate_frequency_data(
    operation_cards: list[OperationCard],
    surgeons: list[Surgeon],
    complexity_scores: np.ndarray,  # Now taken as input
    params: FrequencyParams,
    rng: Optional[np.random.Generator] = None,
) -> Dict[Tuple[OperationCard, Surgeon], float]:
    """Generate f_{t,s} with optional complexity-based surgeon concentration."""
    T, S = len(operation_cards), len(surgeons)
    if T <= 0 or S <= 0:
        raise ValueError("operation_cards and surgeons must be non-empty.")

    if complexity_scores.shape != (T,):
        raise ValueError(
            f"complexity_scores must have shape ({T},), got {complexity_scores.shape}."
        )

    rng_ = rng if rng is not None else np.random.default_rng()

    # Compute concentrations (no longer need to generate baseline params here)
    if params.complexity_scaling > 0.0:
        concentrations = params.surgeon_split_dirichlet_concentration / (
            1.0 + params.complexity_scaling * complexity_scores
        )
    else:
        concentrations = np.full(T, params.surgeon_split_dirichlet_concentration)

    # Generate frequencies
    f_t = rng_.dirichlet(np.full(T, params.case_mix_dirichlet_concentration))

    out: Dict[Tuple[OperationCard, Surgeon], float] = {}
    for t, operation_card in enumerate(operation_cards):
        concentration = float(concentrations[t])

        p_s_given_t = rng_.dirichlet(np.full(S, concentration))

        for s, surgeon in enumerate(surgeons):
            out[(operation_card, surgeon)] = float(f_t[t] * p_s_given_t[s])

    return out
