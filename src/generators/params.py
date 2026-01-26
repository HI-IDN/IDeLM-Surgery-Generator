from typing import Callable, Dict, Tuple, TypeAlias, Union

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

Split: TypeAlias = Dict[int, Dict[str, Tuple[int | float, int | float]]]


class FrequencyParams(BaseModel):
    skewness: float = 2.119  # Skewness for operation card frequencies, higher values lead to more skewed distributions
    n_groups: int = 8  # Number of groups for operation cards
    mixing_rate: float = 0.1  # Mixing rate for operation card groups
    max_groups_per_surgeon: int = 2  # Maximum number of groups a surgeon can belong
    surgeon_activity_mu: float = (
        1.0  # Mean for surgeon activity (log-normal distribution)
    )
    surgeon_activity_sigma: float = (
        0.7  # Standard deviation for surgeon activity (log-normal distribution)
    )


class DurationParams(BaseModel):
    base_mean_log: float = 4.12  # Base mean log duration for surgeries
    base_std_log: float = 0.783  # Base standard deviation of log duration for surgeries
    max_dev_mean: float = 1.8  # How much the mean log duration can deviate from the base mean log based on frequency
    max_dev_sd: float = 0.7  # How much the sd log duration can deviate from the base sd log based on frequency
    min_std: float = 0.0771  # Minimum log standard deviation for surgery durations
    max_std: float = 1.56  # Maximum log standard deviation for surgery durations


class ScheduleParams(BaseModel):
    slots_per_day: int = 8  # Number of time slots available per day for surgeries
    entropy: float = 0.05  # Entropy for schedule generation
    slot_randomness_scale: float = (
        0.1  # Randomness scale for the number of slots per surgeon
    )


class PatternParams(BaseModel):
    max_minutes_per_day: int = 480  # Maximum minutes available for surgeries in a day
    num_patterns_per_room_day: int = (
        50  # Number of patterns to generate per room per day
    )


class WindowParams(BaseModel):
    splits: Split = {
        0: {
            "plan_start_range": (1, 10),
            "plan_len_range": (5, 14),
            "op_start_range": (0, 20),
            "op_len_range": (10, 25),
            "allowed_changes_range": (0, 1),
        },
        20: {
            "plan_start_range": (3, 20),
            "plan_len_range": (7, 20),
            "op_start_range": (0, 25),
            "op_len_range": (14, 40),
            "allowed_changes_range": (1, 2),
        },
        80: {
            "plan_start_range": (5, 30),
            "plan_len_range": (10, 30),
            "op_start_range": (0, 30),
            "op_len_range": (20, 60),
            "allowed_changes_range": (2, 3),
        },
    }


def default_registration_distribution(n: int) -> npt.NDArray[np.float64]:
    """Function for generating days since registration

    Parameters
    ----------
    n : int
        Number of surgeries to generate days since registration for.

    Returns
    -------
    npt.NDArray[np.float_]
        Array of days since registration for each surgery.
    """
    return np.random.lognormal(mean=2.5, sigma=0.8, size=n)


def decaying_fullness(week: int) -> float:
    """Fullness function that returns a decreasing fullness value based on the week number."""
    return max(0.2, 0.8 - 0.15 * week)


class WaitingListParams(BaseModel):
    max_days_registered: int = 90  # Maximum number of days since registration
    registration_distribution: Callable[[int], npt.NDArray[np.float64]] = Field(
        default=default_registration_distribution, exclude=True
    )


class InitialPlanParams(BaseModel):
    plan_horizon_weeks: int = 8  # Number of weeks to plan ahead
    max_days_registered: int = 90  # Maximum number of days since registration
    registration_distribution: Callable[[int], npt.NDArray[np.float64]] = Field(
        default=default_registration_distribution, exclude=True
    )
    fullness: Union[float, Callable[[int], float]] = Field(
        default=decaying_fullness, exclude=True
    )


class AdmissionParams(BaseModel):
    splits: Split = {
        0: {
            "p_icu": (0.0, 0.05),
            "p_ward": (0.1, 0.15),
            "mean_los_icu": (0.5, 1.0),
            "sd_los_icu": (0.1, 0.3),
            "mean_los_ward": (1, 3),
            "sd_los_ward": (0.1, 0.3),
        },
        20: {
            "p_icu": (0.05, 0.1),
            "p_ward": (0.2, 0.3),
            "mean_los_icu": (1, 2),
            "sd_los_icu": (0.2, 0.4),
            "mean_los_ward": (2, 4),
            "sd_los_ward": (0.2, 0.4),
        },
        80: {
            "p_icu": (0.1, 0.3),
            "p_ward": (0.4, 0.6),
            "mean_los_icu": (2, 3),
            "mean_los_ward": (3, 6),
            "sd_los_icu": (0.3, 0.5),
            "sd_los_ward": (0.3, 0.5),
        },
    }
