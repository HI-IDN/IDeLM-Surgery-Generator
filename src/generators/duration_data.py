from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..models import DurationCell
from ..type_aliases import OperationCard, Surgeon
from .helpers import rng_or_default
from .params import DurationParams


def _infer_index_maps_from_frequency_data(
    frequency_data: Dict[Tuple[OperationCard, Surgeon], float],
) -> Tuple[
    List[OperationCard], List[Surgeon], Dict[OperationCard, int], Dict[Surgeon, int]
]:
    """
    Build stable index maps for operation cards and surgeons from frequency_data.

    Returns:
      operation_cards: list of unique OperationCard in sorted order
      surgeons:        list of unique Surgeon in sorted order
      op_to_i:         mapping OperationCard -> row index
      s_to_j:          mapping Surgeon -> col index
    """
    if not frequency_data:
        raise ValueError("frequency_data is empty.")

    operation_cards = sorted({op for (op, _s) in frequency_data.keys()})
    surgeons = sorted({s for (_op, s) in frequency_data.keys()})

    op_to_i = {op: i for i, op in enumerate(operation_cards)}
    s_to_j = {s: j for j, s in enumerate(surgeons)}

    return operation_cards, surgeons, op_to_i, s_to_j


def _build_frequency_matrix(
    operation_cards: List[OperationCard],
    surgeons: List[Surgeon],
    op_to_i: Dict[OperationCard, int],
    s_to_j: Dict[Surgeon, int],
    frequency_data: Dict[Tuple[OperationCard, Surgeon], float],
) -> np.ndarray:
    """
    Build f_{t,s} matrix of shape (T,S) from frequency_data.
    Missing pairs get 0.
    """
    T, S = len(operation_cards), len(surgeons)
    f_ts = np.zeros((T, S), dtype=float)

    for (op, s), f in frequency_data.items():
        ff = float(f)
        if ff < 0:
            raise ValueError(
                f"Negative frequency for (operation_card={op}, surgeon={s})."
            )
        f_ts[op_to_i[op], s_to_j[s]] = ff

    return f_ts


def generate_duration_data(
    frequency_data: Dict[Tuple[OperationCard, Surgeon], float],
    params: DurationParams,
    rng: Optional[np.random.Generator] = None,
) -> Dict[Tuple[OperationCard, Surgeon], DurationCell]:
    """
    Baseline per operation card t:
        D_t = gamma_t + exp(N(mu_t, sigma_t^2))

    Surgeon/type effect with normalized specialization:
        kappa_{t,s} = clip(
            g_s * h_{t,s} * ((f_{t,s}+eps)/(mean_s f_{t,s}+eps))^(-b),
            kappa_min, kappa_max
        )

    Transform to preserve 3-parameter lognormal:
        mu_{t,s}    = mu_t + ln(kappa_{t,s})
        sigma_{t,s} = sigma_t
        gamma_{t,s} = kappa_{t,s} * gamma_t

    Returns mapping: (operation_card, surgeon) -> {"mu","sigma","gamma","kappa"}
    """
    rng_ = rng_or_default(rng)

    # Build index maps from string operation cards and surgeon ids
    operation_cards, surgeons, op_to_i, s_to_j = _infer_index_maps_from_frequency_data(
        frequency_data
    )
    T, S = len(operation_cards), len(surgeons)

    f_ts = _build_frequency_matrix(
        operation_cards, surgeons, op_to_i, s_to_j, frequency_data
    )

    # --- Baseline per operation card (indexed by i for operation_cards[i]) ---
    mu_t = rng_.normal(loc=params.mu_mean, scale=params.mu_sd, size=T)

    sigma_low, sigma_high = sorted((params.sigma_low, params.sigma_high))
    sigma_t = rng_.uniform(low=sigma_low, high=sigma_high, size=T)

    gamma_low, gamma_high = sorted((params.gamma_low, params.gamma_high))
    gamma_t = rng_.uniform(low=gamma_low, high=gamma_high, size=T)

    # --- Multipliers g_s and h_{t,s} ---
    g_s = (
        np.ones(S, dtype=float)
        if params.global_skill_sigma == 0.0
        else rng_.lognormal(mean=0.0, sigma=params.global_skill_sigma, size=S)
    )
    h_ts = (
        np.ones((T, S), dtype=float)
        if params.type_skill_sigma == 0.0
        else rng_.lognormal(mean=0.0, sigma=params.type_skill_sigma, size=(T, S))
    )

    # --- Normalized specialization effect ---
    b = params.specialization_exponent_b
    eps = params.specialization_epsilon

    mean_f_t = f_ts.mean(axis=1, keepdims=True)  # (T,1)
    spec_factor = ((f_ts + eps) / (mean_f_t + eps)) ** (-b)  # (T,S)

    kappa_ts = g_s[None, :] * h_ts * spec_factor
    kappa_ts = np.clip(kappa_ts, params.kappa_min, params.kappa_max)

    # Output in the same key-space as frequency_data
    out: Dict[Tuple[OperationCard, Surgeon], DurationCell] = {}
    for op, s in frequency_data.keys():
        i = op_to_i[op]
        j = s_to_j[s]
        kap = float(kappa_ts[i, j])
        out[(op, s)] = {
            "kappa": kap,
            "mu": float(mu_t[i] + np.log(kap)),
            "sigma": float(sigma_t[i]),
            "gamma": float(kap * gamma_t[i]),
        }

    return out
