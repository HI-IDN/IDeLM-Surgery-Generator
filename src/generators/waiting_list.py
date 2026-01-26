import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..models import Surgery
from ..type_aliases import OperationCard, Surgeon, TimeWindow
from .params import WaitingListParams


def generate_waiting_list(
    n: int,
    frequency_data: Dict[Tuple[OperationCard, Surgeon], float],
    duration_data: Dict[str, Tuple[float, float]],
    window_data: Dict[OperationCard, Dict[str, Union[TimeWindow, int]]],
    admission_data: Dict[OperationCard, Dict[str, Union[float, Tuple[float, float]]]],
    params: WaitingListParams,
    admission_dist: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> List[Surgery]:
    """Generate a waiting list of surgeries based on frequency, duration, and window data.

    Parameters
    ----------
    n : int
        Number of surgeries to generate.
    frequency_data : Dict[Tuple[OperationCard, Surgeon], float]
        Dictionary mapping (operation card, surgeon) pairs to their frequencies.
    duration_data : Dict[str, Tuple[float, float]]
        Dictionary mapping operation cards to their duration distributions (mean log and std log).
    window_and_data : Dict[OperationCard, Dict[str, Union[TimeWindow, int]]]
        Dictionary mapping operation cards to their planning and operation windows, and allowed changes.
    admission_data : Dict[OperationCard, Dict[str, Union[float, Tuple[float, float]]]]
        Dictionary mapping operation cards to their admission data.
        Each entry contains a dictionary with keys "p_icu", "p_ward", "icu_los", and "ward_los".
        - "p_icu" is the probability of ICU admission.
        - "p_ward" is the probability of ward admission.
        - "icu_los" is a tuple (mean, std) for ICU length of stay.
        - "ward_los" is a tuple (mean, std) for ward length of stay.
    params : WaitingListParams
        Parameters for waiting list generation.
    admission_dist : bool, optional
        Whether to interpret the values in admission_data as distribution parameters, by default True.
        If False, one of the values is used directly.
    rng : Optional[np.random.Generator], optional
        Numpy random number generator for reproducibility, by default None

    Returns
    -------
    List[Surgery]
        List of Surgery objects representing the generated waiting list.
    """

    if rng is None:
        rng = np.random.default_rng()

    # Normalize frequencies for sampling
    tuples = list(frequency_data.keys())
    weights = np.array([frequency_data[t] for t in tuples], dtype=np.float64)
    weights /= weights.sum()

    # Sample all registration days at once
    raw_days = params.registration_distribution(n)
    raw_days = np.clip(np.round(raw_days), 0, params.max_days_registered).astype(int)

    waiting_list = []
    for i in range(n):
        operation_card, surgeon = rng.choice(tuples, p=weights, size=1)[0]

        # Duration (log-normal expected value)
        mean_log, std_log = duration_data[operation_card]
        expected_duration = int(np.exp(mean_log + 0.5 * std_log**2))

        # Planning and operation windows, allowed changes
        window_info = window_data[operation_card]
        planning_window: TimeWindow = window_info["planning_window"]  # type: ignore
        operation_window: TimeWindow = window_info["operation_window"]  # type: ignore
        allowed_changes: int = window_info["allowed_changes"]  # type: ignore
        allowed_days_moved_plus: int = allowed_changes * 7
        allowed_days_moved_minus: int = allowed_changes * 1

        # Admission information
        admission_info = admission_data[operation_card]
        icu: bool = rng.random() < admission_info["p_icu"]  # type: ignore
        ward: bool = rng.random() < admission_info["p_ward"]  # type: ignore
        if not admission_dist:
            los_icu = (
                int(math.ceil(rng.choice(admission_info["icu_los"]))) if icu else 0
            )
            los_ward = (
                int(math.ceil(rng.choice(admission_info["ward_los"]))) if ward else 0
            )
        else:
            los_icu: int = (
                math.ceil(
                    rng.lognormal(
                        mean=admission_info["icu_los"][0],  # type: ignore
                        sigma=admission_info["icu_los"][1],  # type: ignore
                    )
                )
                if icu
                else 0
            )
            los_ward: int = (
                math.ceil(
                    rng.lognormal(
                        mean=admission_info["ward_los"][0],  # type: ignore
                        sigma=admission_info["ward_los"][1],  # type: ignore
                    )
                )
                if ward
                else 0
            )

        # Clip registration date to ensure it's within operation window end
        max_reg_day = max(0, operation_window[1] - 5)
        reg_day = min(raw_days[i], max_reg_day)

        waiting_list.append(
            Surgery(
                operation_card_id=operation_card,
                surgeon_id=surgeon,
                expected_duration=expected_duration,
                days_since_registration=reg_day,
                planning_window=planning_window,
                operation_window=operation_window,
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
