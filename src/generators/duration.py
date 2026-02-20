from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..models import DurationCell
from ..type_aliases import OperationCard, Surgeon
from .params import DurationParams


def generate_duration_data(
    frequency_data: Dict[Tuple[OperationCard, Surgeon], float],
    mu_t: np.ndarray,
    sigma_t: np.ndarray,
    gamma_t: np.ndarray,
    params: DurationParams,
    rng: np.random.Generator,
) -> Dict[Tuple[OperationCard, Surgeon], DurationCell]:
    """
    Generate duration parameters with surgeon speed multipliers.

    kappa_{t,s} = clip(g_s * h_{t,s} * ((f_{t,s}+eps)/(mean_s f_{t,s}+eps))^(-b), min, max)
    """
    operation_cards = sorted({op for (op, _) in frequency_data.keys()})
    surgeons = sorted({s for (_, s) in frequency_data.keys()})
    T, S = len(operation_cards), len(surgeons)

    op_to_i = {op: i for i, op in enumerate(operation_cards)}
    s_to_j = {s: j for j, s in enumerate(surgeons)}

    f_ts = np.zeros((T, S))
    for (op, s), f in frequency_data.items():
        f_ts[op_to_i[op], s_to_j[s]] = f

    g_s = (
        np.ones(S)
        if params.global_skill_sigma == 0
        else rng.lognormal(0, params.global_skill_sigma, S)
    )
    h_ts = (
        np.ones((T, S))
        if params.type_skill_sigma == 0
        else rng.lognormal(0, params.type_skill_sigma, (T, S))
    )

    mean_f_t = f_ts.mean(axis=1, keepdims=True)
    spec_factor = (
        (f_ts + params.specialization_epsilon)
        / (mean_f_t + params.specialization_epsilon)
    ) ** (-params.specialization_exponent_b)

    kappa_ts = np.clip(
        g_s[None, :] * h_ts * spec_factor, params.kappa_min, params.kappa_max
    )

    out: Dict[Tuple[OperationCard, Surgeon], DurationCell] = {}
    for op, s in frequency_data.keys():
        i, j = op_to_i[op], s_to_j[s]
        kap = float(kappa_ts[i, j])
        out[(op, s)] = DurationCell(
            kappa=kap,
            mu=float(mu_t[i] + np.log(kap)),
            sigma=float(sigma_t[i]),
            gamma=float(kap * gamma_t[i]),
        )

    return out
