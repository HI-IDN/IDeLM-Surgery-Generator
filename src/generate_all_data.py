from typing import Dict, List, Tuple, TypeAlias

import numpy as np

from . import generators
from .generators.params import (
    AdmissionParams,
    DurationParams,
    FrequencyParams,
    PriorityParams,
    ScheduleParams,
)
from .models import DurationCell, Surgery
from .type_aliases import (
    OperationCard,
    Schedule,
    Surgeon,
)

AllData: TypeAlias = Tuple[
    Dict[Tuple[OperationCard, Surgeon], float],  # frequency_data
    Dict[Tuple[OperationCard, Surgeon], DurationCell],  # duration_data
    Schedule,  # schedule
    Dict[OperationCard, Dict[str, int]],  # priority_data
    Dict[OperationCard, Dict[str, float | Tuple[float, float]]],  # admission_data
    List[Surgery],  # waiting_list
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
    priority_params: PriorityParams,
    admission_params: AdmissionParams,
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
    priority_params : PriorityParams
        Parameters for generating priority data.
    admission_params : AdmissionParams
        Parameters for generating admission data.

    Returns
    -------
    AllData
        A tuple containing the generated data.
    """
    ss = np.random.SeedSequence(seed)
    rngs = [np.random.default_rng(s) for s in ss.spawn(8)]
    weekdays: list[int] = [0, 1, 2, 3, 4]  # Monday to Friday
    frequency_data = generators.generate_frequency_data(
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
    priority_data = generators.generate_priority_data(
        frequency_data=frequency_data,
        params=priority_params,
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
        priority_data=priority_data,
        admission_data=admission_data,
        admission_dist=True,
        rng=rngs[6],
    )

    return (
        frequency_data,
        duration_data,
        schedule,
        priority_data,
        admission_data,
        waiting_list,
    )
