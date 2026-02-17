from __future__ import annotations

from typing import Dict, Tuple, TypeAlias

from pydantic import BaseModel, Field

Split: TypeAlias = Dict[int, Dict[str, Tuple[int | float, int | float]]]


class FrequencyParams(BaseModel):
    case_mix_dirichlet_concentration: float = Field(
        default=1.0,
        gt=0.0,
        description="Dirichlet concentration for operation-card frequencies. Lower = more uneven.",
    )

    surgeon_split_dirichlet_concentration: float = Field(
        default=1.0,
        gt=0.0,
        description="Dirichlet concentration for splitting operation cards across surgeons. Lower = more specialization.",
    )


class DurationParams(BaseModel):
    mu_mean: float = Field(
        default=3.5,
        description="Mean of baseline log-duration parameter mu_t.",
    )
    mu_sd: float = Field(
        default=0.35,
        ge=0.0,
        description="Std dev of baseline log-duration parameter mu_t.",
    )

    sigma_low: float = Field(
        default=0.20,
        gt=0.0,
        description="Lower bound for sigma_t.",
    )
    sigma_high: float = Field(
        default=0.60,
        gt=0.0,
        description="Upper bound for sigma_t.",
    )

    gamma_low: float = Field(
        default=0.0,
        ge=0.0,
        description="Lower bound for gamma_t.",
    )
    gamma_high: float = Field(
        default=10.0,
        ge=0.0,
        description="Upper bound for gamma_t.",
    )

    global_skill_sigma: float = Field(
        default=0.10,
        ge=0.0,
        description="Std dev of global surgeon speed multiplier.",
    )
    type_skill_sigma: float = Field(
        default=0.05,
        ge=0.0,
        description="Std dev of type-specific surgeon multiplier.",
    )

    kappa_min: float = Field(
        default=0.70,
        gt=0.0,
        description="Minimum multiplier (faster limit).",
    )
    kappa_max: float = Field(
        default=1.40,
        gt=0.0,
        description="Maximum multiplier (slower limit).",
    )

    specialization_exponent_b: float = Field(
        default=0.10,
        ge=0.0,
        description="Strength of specialization effect.",
    )
    specialization_epsilon: float = Field(
        default=1e-6,
        gt=0.0,
        description="Small constant to avoid division by zero.",
    )


class ScheduleParams(BaseModel):
    slots_per_day: int = Field(
        default=8,
        ge=1,
        description="Number of slots per day.",
    )
    entropy: float = Field(
        default=0.05,
        ge=0.0,
        description="Schedule randomness parameter.",
    )
    slot_randomness_scale: float = Field(
        default=0.1,
        ge=0.0,
        description="Variation in slots per surgeon.",
    )


class PriorityParams(BaseModel):
    splits: Split = Field(
        default_factory=lambda: {
            0: {
                "operate_by_range": (0, 45),
                "allowed_changes_range": (0, 1),
            },
            20: {
                "operate_by_range": (0, 65),
                "allowed_changes_range": (1, 2),
            },
            80: {
                "operate_by_range": (0, 90),
                "allowed_changes_range": (2, 3),
            },
        },
        description="Priority parameter ranges by split.",
    )


class AdmissionParams(BaseModel):
    splits: Split = Field(
        default_factory=lambda: {
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
        },
        description="Admission parameter ranges by split.",
    )
