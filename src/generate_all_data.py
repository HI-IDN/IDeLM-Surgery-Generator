from typing import Dict, List, Tuple, TypeAlias, Union

import numpy as np

from .type_aliases import (
    OperationCard,
    Pattern,
    Room,
    Schedule,
    Surgeon,
    TimeWindow,
    Weekday,
)

from . import generators
from .generators.params import (
    AdmissionParams,
    DurationParams,
    FrequencyParams,
    InitialPlanParams,
    PatternParams,
    ScheduleParams,
    WaitingListParams,
    WindowParams,
)
from .models import Surgery

AllData: TypeAlias = Tuple[
    Dict[Tuple[OperationCard, Surgeon], float],  # frequency_data
    Dict[OperationCard, int],  # card_groups
    Dict[OperationCard, Tuple[float, float]],  # duration_data
    Schedule,  # schedule
    Dict[Tuple[Weekday, Room], List[Pattern]],  # patterns
    Dict[OperationCard, Dict[str, Union[TimeWindow, int]]],  # window_data
    Dict[OperationCard, Dict[str, float | Tuple[float, float]]],  # admission_data
    List[Surgery],  # waiting_list
    List[Surgery],  # initial_plan
]


def generate_all_data(
    n_rooms: int,
    n_surgeons: int,
    n_operation_cards: int,
    waiting_list_size: int,
    seed: int,
    frequency_params: FrequencyParams,
    duration_params: DurationParams,
    schedule_params: ScheduleParams,
    pattern_params: PatternParams,
    window_params: WindowParams,
    admission_params: AdmissionParams,
    waiting_list_params: WaitingListParams,
    initial_plan_params: InitialPlanParams,
) -> AllData:
    """Generates all data required for the simulation based on the given parameters.

    Parameters
    ----------
    n_rooms : int
        The number of rooms available for surgeries.
    n_surgeons : int
        The number of surgeons available.
    n_operation_cards : int
        The number of different operation cards (surgery types).
    waiting_list_size : int
        The number of surgeries to generate for the waiting list.
    seed : int
        RNG seed.
    frequency_params : FrequencyParams
        Parameters for generating frequency data.
    duration_params : DurationParams
        Parameters for generating duration data.
    schedule_params : ScheduleParams
        Parameters for generating the schedule.
    pattern_params : PatternParams
        Parameters for generating surgery patterns.
    window_params : WindowParams
        Parameters for generating time windows.
    admission_params : AdmissionParams
        Parameters for generating admission data.
    waiting_list_params : WaitingListParams
        Parameters for generating the waiting list.
    initial_plan_params : InitialPlanParams
        Parameters for generating the initial plan.

    Returns
    -------
    AllData
        A tuple containing the generated data.
    """
    ss = np.random.SeedSequence(seed)
    rngs = [np.random.default_rng(s) for s in ss.spawn(8)]
    weekdays: list[int] = [0, 1, 2, 3, 4]  # Monday to Friday
    frequency_data, card_groups = generators.generate_frequency_data(
        num_operation_cards=n_operation_cards,
        num_surgeons=n_surgeons,
        params=frequency_params,
        rng=rngs[0],
    )
    duration_data = generators.generate_duration_data(
        frequency_data=frequency_data,
        params=duration_params,
        rng=rngs[1],
    )
    schedule = generators.generate_schedule(
        frequency_data=frequency_data,
        rooms=[i for i in range(n_rooms)],
        weekdays=weekdays,
        params=schedule_params,
        rng=rngs[2],
    )
    patterns = generators.generate_patterns(
        frequency_data=frequency_data,
        schedule=schedule,
        duration_distributions=duration_data,
        params=pattern_params,
        rng=rngs[3],
    )
    window_data = generators.generate_window_data(
        frequency_data=frequency_data,
        params=window_params,
        rng=rngs[4],
    )
    admission_data = generators.generate_admission_data(
        frequency_data=frequency_data,
        params=admission_params,
        rng=rngs[5],
    )
    waiting_list = generators.generate_waiting_list(
        n=waiting_list_size,
        frequency_data=frequency_data,
        duration_data=duration_data,
        window_data=window_data,
        admission_data=admission_data,
        params=waiting_list_params,
        admission_dist=True,
        rng=rngs[6],
    )

    initial_plan = generators.generate_initial_plan(
        pattern_data=patterns,
        schedule=schedule,
        frequency_data=frequency_data,
        duration_data=duration_data,
        window_data=window_data,
        admission_data=admission_data,
        params=initial_plan_params,
        admission_dist=True,
        rng=rngs[7],
    )

    return (
        frequency_data,
        card_groups,
        duration_data,
        schedule,
        patterns,
        window_data,
        admission_data,
        waiting_list,
        initial_plan,
    )
