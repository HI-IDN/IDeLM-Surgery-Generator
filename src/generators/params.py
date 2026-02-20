from __future__ import annotations

from pydantic import BaseModel, Field


class FrequencyParams(BaseModel):
    case_mix_dirichlet_concentration: float = Field(
        default=1.0,
        gt=0.0,
        description="Dirichlet concentration for operation-card frequencies. Lower = more uneven.",
    )
    surgeon_split_dirichlet_concentration: float = Field(
        default=1.0,
        gt=0.0,
        description="Base Dirichlet concentration for splitting operation cards across surgeons. Lower = more specialization.",
    )
    complexity_scaling: float = Field(
        default=0.0,
        ge=0.0,
        description="Scales surgeon concentration by complexity: concentration_t = base / (1 + scaling * complexity_t). 0 = disabled.",
    )


class DurationParams(BaseModel):
    mu_mean: float = Field(default=3.5, description="Mean of baseline mu_t.")
    mu_sd: float = Field(default=0.35, ge=0.0, description="Std dev of baseline mu_t.")
    sigma_low: float = Field(
        default=0.20, gt=0.0, description="Lower bound for sigma_t."
    )
    sigma_high: float = Field(
        default=0.60, gt=0.0, description="Upper bound for sigma_t."
    )
    gamma_low: float = Field(
        default=0.0, ge=0.0, description="Lower bound for gamma_t."
    )
    gamma_high: float = Field(
        default=10.0, ge=0.0, description="Upper bound for gamma_t."
    )
    global_skill_sigma: float = Field(
        default=0.10, ge=0.0, description="Std dev of global surgeon speed multiplier."
    )
    type_skill_sigma: float = Field(
        default=0.05, ge=0.0, description="Std dev of type-specific surgeon multiplier."
    )
    kappa_min: float = Field(
        default=0.70, gt=0.0, description="Minimum multiplier (faster limit)."
    )
    kappa_max: float = Field(
        default=1.40, gt=0.0, description="Maximum multiplier (slower limit)."
    )
    specialization_exponent_b: float = Field(
        default=0.10, ge=0.0, description="Strength of specialization effect."
    )
    specialization_epsilon: float = Field(
        default=1e-6, gt=0.0, description="Small constant to avoid division by zero."
    )


class ScheduleParams(BaseModel):
    """
    Parameters for generating master surgical schedules via Dirichlet sampling.

    Each surgeon's (room, weekday) preferences are sampled from a Dirichlet distribution.
    The concentration parameter controls how focused vs. spread out their schedule is:
    - Low concentration → highly focused (few preferred room-day combinations)
    - High concentration → spread out (operates across many room-day combinations)

    Busier surgeons (higher workload) can optionally have more spread-out schedules.
    """

    base_concentration: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Base Dirichlet concentration for surgeon-room-day assignments. "
            "Lower = more focused schedules (surgeons have strong preferences). "
            "Higher = more uniform schedules (surgeons operate everywhere)."
        ),
    )

    workload_scaling: float = Field(
        default=0.5,
        ge=0.0,
        description=(
            "How surgeon workload affects concentration. "
            "concentration = base / (1 + scaling * workload). "
            "Higher = busy surgeons have more spread-out schedules. "
            "0 = all surgeons use base concentration."
        ),
    )

    sparsity_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description=(
            "Zero out (room, day) assignments below this threshold. "
            "Creates sparse schedules where surgeons only operate on specific days/rooms. "
            "0 = no sparsification (all assignments kept)."
        ),
    )


class PriorityParams(BaseModel):
    """
    Parameters for continuous linear mapping of complexity to priority parameters.

    Each parameter follows: value = base + complexity_effect + noise
    where higher complexity leads to shorter operate_by windows and fewer allowed_changes.

    Note: For priority, we use inverse mapping (high complexity → low values) because
    complex surgeries need shorter waiting times and less flexibility.
    """

    # Operate-by days (waiting time target)
    operate_by_min: int = Field(
        default=14, ge=1, description="Min operate_by days (most complex surgeries)"
    )
    operate_by_max: int = Field(
        default=90, ge=1, description="Max operate_by days (least complex surgeries)"
    )
    operate_by_noise: int = Field(
        default=7, ge=0, description="Uniform noise range in days (±noise)"
    )

    # Allowed changes (rescheduling tolerance)
    allowed_changes_min: int = Field(
        default=0, ge=0, description="Min allowed changes (most complex surgeries)"
    )
    allowed_changes_max: int = Field(
        default=5, ge=0, description="Max allowed changes (least complex surgeries)"
    )
    allowed_changes_noise: int = Field(
        default=1, ge=0, description="Uniform noise range for changes (±noise)"
    )


class AdmissionParams(BaseModel):
    """
    Parameters for continuous linear mapping of complexity to admission/LOS parameters.

    Each parameter follows: value = min + complexity * (max - min) + noise
    where noise ~ Uniform(-noise_param, +noise_param)

    ICU admission probability ranges from p_icu_min (low complexity) to p_icu_max (high).
    Ward admission follows the same pattern.
    LOS parameters use lognormal (μ, σ) in log-space where median LOS = exp(μ).
    """

    # ICU admission probability
    p_icu_min: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Min ICU admission probability"
    )
    p_icu_max: float = Field(
        default=0.30, ge=0.0, le=1.0, description="Max ICU admission probability"
    )
    p_icu_noise: float = Field(
        default=0.05, ge=0.0, description="Uniform noise range for ICU probability"
    )

    # Ward admission probability
    p_ward_min: float = Field(
        default=0.20, ge=0.0, le=1.0, description="Min ward admission probability"
    )
    p_ward_max: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Max ward admission probability"
    )
    p_ward_noise: float = Field(
        default=0.10, ge=0.0, description="Uniform noise range for ward probability"
    )

    # ICU LOS lognormal μ (log-space location parameter)
    icu_los_mu_min: float = Field(
        default=0.0,
        description="Min ICU LOS μ. Median LOS = exp(μ), so μ=0 → 1 day median.",
    )
    icu_los_mu_max: float = Field(
        default=1.0,
        description="Max ICU LOS μ. μ=1.0 → ~2.7 day median.",
    )
    icu_los_mu_noise: float = Field(
        default=0.15, ge=0.0, description="Uniform noise range for ICU LOS μ"
    )

    # ICU LOS lognormal σ (log-space scale parameter)
    icu_los_sigma_min: float = Field(
        default=0.1, gt=0.0, description="Min ICU LOS σ (must be positive)"
    )
    icu_los_sigma_max: float = Field(default=0.5, gt=0.0, description="Max ICU LOS σ")
    icu_los_sigma_noise: float = Field(
        default=0.05, ge=0.0, description="Uniform noise range for ICU LOS σ"
    )

    # Ward LOS lognormal μ
    ward_los_mu_min: float = Field(
        default=0.5,
        description="Min ward LOS μ. μ=0.5 → ~1.6 day median.",
    )
    ward_los_mu_max: float = Field(
        default=2.0,
        description="Max ward LOS μ. μ=2.0 → ~7.4 day median.",
    )
    ward_los_mu_noise: float = Field(
        default=0.20, ge=0.0, description="Uniform noise range for ward LOS μ"
    )

    # Ward LOS lognormal σ
    ward_los_sigma_min: float = Field(
        default=0.2, gt=0.0, description="Min ward LOS σ (must be positive)"
    )
    ward_los_sigma_max: float = Field(default=0.6, gt=0.0, description="Max ward LOS σ")
    ward_los_sigma_noise: float = Field(
        default=0.05, ge=0.0, description="Uniform noise range for ward LOS σ"
    )
