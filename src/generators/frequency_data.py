from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..type_aliases import OperationCard, Surgeon
from .helpers import dirichlet_uniform, rng_or_default
from .params import FrequencyParams


def generate_frequency_data(
    num_operation_cards: int,
    num_surgeons: int,
    params: FrequencyParams,
    rng: Optional[np.random.Generator] = None,
) -> Dict[Tuple[OperationCard, Surgeon], float]:
    """
    Generates:
      - f_t ~ Dirichlet(case_mix_dirichlet_concentration * 1)
      - p(s|t) ~ Dirichlet(surgeon_split_dirichlet_concentration * 1)
      - f_{t,s} = f_t * p(s|t)

    Returns mapping: (t, s) -> frequency
    """
    if num_operation_cards <= 0 or num_surgeons <= 0:
        raise ValueError("num_operation_cards and num_surgeons must be positive.")

    rng_ = rng_or_default(rng)
    T, S = num_operation_cards, num_surgeons
    operation_cards = [f"Operation_{i}" for i in range(num_operation_cards)]

    f_t = dirichlet_uniform(T, params.case_mix_dirichlet_concentration, rng_)

    out: Dict[Tuple[OperationCard, Surgeon], float] = {}
    for operation_card, t in zip(operation_cards, range(T)):
        p_s_given_t = dirichlet_uniform(
            S, params.surgeon_split_dirichlet_concentration, rng_
        )
        for s in range(S):
            out[(operation_card, s)] = f_t[t] * p_s_given_t[s]

    return out
