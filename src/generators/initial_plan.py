import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..models import Surgery
from ..type_aliases import (
    OperationCard,
    Pattern,
    Room,
    Schedule,
    Surgeon,
    TimeWindow,
    Weekday,
)

from .params import InitialPlanParams


def generate_initial_plan(
    pattern_data: Dict[Tuple[Weekday, Room], List[Pattern]],
    schedule: Schedule,
    frequency_data: Dict[Tuple[OperationCard, Surgeon], float],
    duration_data: Dict[OperationCard, Tuple[float, float]],
    window_data: Dict[OperationCard, Dict[str, Union[TimeWindow, int]]],
    admission_data: Dict[OperationCard, Dict[str, Union[float, Tuple[float, float]]]],
    params: InitialPlanParams,
    admission_dist: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> List[Surgery]:
    """Generate an initial plan of surgeries based on patterns, surgeon availability, and operation cards.

    Parameters
    ----------
    pattern_data : Dict[Tuple[Weekday, Room], List[Pattern]]
        Dictionary mapping (weekday, room) pairs to lists of operation patterns.
        Each pattern is a list of operation cards that can be performed on that day in that room.
    schedule : Schedule
        Dictionary mapping (surgeon, room, weekday) to a score indicating the surgeon's availability.
        A score of 0 means the surgeon is not available on that day in that room.
    frequency_data : Dict[Tuple[OperationCard, Surgeon], float]
        Dictionary mapping (operation card, surgeon) pairs to their frequencies.
        This indicates how often each surgeon performs each operation card.
    duration_data : Dict[OperationCard, Tuple[float, float]]
        Dictionary mapping operation cards to their duration distributions.
        Each entry contains a tuple of (mean log duration, standard deviation of log duration).
    window_data : Dict[OperationCard, Dict[str, Union[TimeWindow, int]]]
        Dictionary mapping operation cards to their planning and operation windows, and allowed changes.
        Each entry contains a dictionary with keys "planning_window", "operation_window", and "allowed_changes".
    admission_data : Dict[OperationCard, Dict[str, Union[float, Tuple[float, float]]]]
        Dictionary mapping operation cards to their admission data.
        Each entry contains a dictionary with keys "p_icu", "p_ward", "icu_los", and "ward_los".
        - "p_icu" is the probability of ICU admission.
        - "p_ward" is the probability of ward admission.
        - "icu_los" is a tuple (mean, std) for ICU length of stay.
        - "ward_los" is a tuple (mean, std) for ward length of stay.
    params : InitialPlanParams
        Parameters for generating the initial plan, including:
    admission_dist : bool, optional
        Whether to intepret the values in admission_data as distribution parameters, by default True.
        If False, one of the values is used directly.
    rng : Optional[np.random.Generator], optional
        Numpy random number generator for reproducibility, by default None

    Returns
    -------
    List[Surgery]
        List of Surgery objects representing the planned surgeries.
    """

    if rng is None:
        rng = np.random.default_rng()

    if isinstance(params.fullness, float):

        def fullness_func(week):
            return params.fullness

    else:
        assert callable(params.fullness), (
            "fullness must be a float or a callable function"
        )
        fullness_func = params.fullness  # type: ignore

    # Build mapping of operation_card -> list of (surgeon, normalized freq)
    card_to_surgeons: Dict[OperationCard, List[Tuple[Surgeon, float]]] = defaultdict(
        list
    )
    for (card, surgeon), freq in frequency_data.items():
        card_to_surgeons[card].append((surgeon, freq))

    for card in card_to_surgeons:
        total = sum(w for _, w in card_to_surgeons[card])
        card_to_surgeons[card] = [(s, w / total) for s, w in card_to_surgeons[card]]

    # Build surgeon availability per (room, weekday)
    room_day_to_surgeons: Dict[Tuple[Weekday, Room], List[Surgeon]] = defaultdict(list)
    for (surgeon, room, day), score in schedule.items():
        if score > 0:
            room_day_to_surgeons[(day, room)].append(surgeon)

    surgeries: List[Surgery] = []

    weekdays = range(7)

    for week in range(params.plan_horizon_weeks):
        week_fullness: float = fullness_func(week)  # type: ignore

        for (day, room), patterns in pattern_data.items():
            absolute_day = week * len(weekdays) + day
            active_surgeons = set(room_day_to_surgeons.get((day, room), []))
            if not active_surgeons or not patterns:
                continue

            np_patterns = np.asarray(patterns, dtype=object)
            selected_pattern = rng.choice(np_patterns)

            for card in selected_pattern:
                if rng.random() > week_fullness:
                    continue

                eligible = [
                    (s, w)
                    for s, w in card_to_surgeons.get(card, [])
                    if s in active_surgeons
                ]
                if not eligible:
                    continue

                surgeons, weights = zip(*eligible)
                surgeon = rng.choice(
                    surgeons, p=np.array(weights) / sum(weights), size=1
                )[0]

                mean_log, std_log = duration_data[card]
                expected_duration = int(np.exp(mean_log + 0.5 * std_log**2))

                days_since_registration = int(params.registration_distribution(1)[0])
                days_since_registration = np.clip(
                    np.round(days_since_registration), 0, params.max_days_registered
                )

                win = window_data[card]
                planning_window: TimeWindow = win["planning_window"]  # type: ignore
                operation_window: TimeWindow = win["operation_window"]  # type: ignore
                allowed_changes: int = win["allowed_changes"]  # type: ignore
                allowed_days_moved_plus: int = allowed_changes * 7
                allowed_days_moved_minus: int = allowed_changes * 1

                # Admission information
                admission_info = admission_data[card]
                icu: bool = rng.random() < admission_info["p_icu"]  # type: ignore
                ward: bool = rng.random() < admission_info["p_ward"]  # type: ignore
                if not admission_dist:
                    los_icu = (
                        int(math.ceil(rng.choice(admission_info["icu_los"])))
                        if icu
                        else 0
                    )
                    los_ward = (
                        int(math.ceil(rng.choice(admission_info["ward_los"])))
                        if ward
                        else 0
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

                surgeries.append(
                    Surgery(
                        operation_card_id=card,
                        surgeon_id=surgeon,
                        expected_duration=expected_duration,
                        days_since_registration=days_since_registration,
                        planning_window=planning_window,
                        operation_window=operation_window,
                        allowed_changes=allowed_changes,
                        allowed_days_moved_plus=allowed_days_moved_plus,
                        allowed_days_moved_minus=allowed_days_moved_minus,
                        icu=icu,
                        ward=ward,
                        los_icu=los_icu,
                        los_ward=los_ward,
                        planned_room=room,
                        planned_day=absolute_day,
                    )
                )

    return surgeries
