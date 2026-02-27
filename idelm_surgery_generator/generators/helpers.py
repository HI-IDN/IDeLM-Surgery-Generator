from typing import Optional

import numpy as np


def rng_or_default(rng: Optional[np.random.Generator]) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def dirichlet_uniform(
    n: int, concentration: float, rng: np.random.Generator
) -> np.ndarray:
    """Dirichlet(concentration * 1_n)."""
    alpha = np.full(n, concentration, dtype=float)
    return rng.dirichlet(alpha)


def compute_complexity_scores(
    mu_t: np.ndarray,
    sigma_t: np.ndarray,
    gamma_t: np.ndarray,
    or_capacity: float,
) -> np.ndarray:
    """
    Compute complexity scores in [0,1] from Leeftink & Hans (2018) classification:
    - relative duration: m_t / c
    - coefficient of variation: s_t / m_t

    Returns: 0.5 * relative_duration + 0.5 * cv, clipped to [0,1]
    """
    if or_capacity <= 0:
        raise ValueError(f"or_capacity must be positive, got {or_capacity}.")

    mu_t = np.asarray(mu_t, dtype=float)
    sigma_t = np.asarray(sigma_t, dtype=float)
    gamma_t = np.asarray(gamma_t, dtype=float)

    exp_term = np.exp(mu_t + 0.5 * sigma_t**2)
    m_t = gamma_t + exp_term
    s_t = np.sqrt(np.maximum(np.exp(sigma_t**2) - 1.0, 0.0)) * exp_term

    relative_duration = np.clip(m_t / or_capacity, 0.0, 1.0)
    cv = np.clip(s_t / np.maximum(m_t, 1e-12), 0.0, 1.0)

    return 0.5 * relative_duration + 0.5 * cv


def generate_baseline_parameters(
    num_operation_cards: int,
    mu_mean: float = 3.5,
    mu_sd: float = 0.35,
    sigma_low: float = 0.20,
    sigma_high: float = 0.60,
    gamma_low: float = 0.0,
    gamma_high: float = 10.0,
    or_capacity: float = 480.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate baseline lognormal parameters and complexity scores for surgery types.

    This function samples the baseline duration distribution parameters that are
    shared across all modules (frequency, duration, priority generation).

    Parameters
    ----------
    num_operation_cards : int
        Number of surgery types (T).
    mu_mean, mu_sd : float
        Mean and std dev for sampling mu_t.
    sigma_low, sigma_high : float
        Bounds for uniform sampling of sigma_t.
    gamma_low, gamma_high : float
        Bounds for uniform sampling of gamma_t.
    or_capacity : float
        OR block capacity in minutes, used for complexity calculation.
    rng : np.random.Generator | None
        Random number generator for reproducibility.

    Returns
    -------
    mu_t : np.ndarray
        Shape (T,), log-duration location parameters.
    sigma_t : np.ndarray
        Shape (T,), log-duration scale parameters.
    gamma_t : np.ndarray
        Shape (T,), threshold (location) parameters.
    complexity_scores : np.ndarray
        Shape (T,), operational complexity scores in [0,1].
    """
    if rng is None:
        rng = np.random.default_rng()

    T = num_operation_cards

    mu_t = rng.normal(loc=mu_mean, scale=mu_sd, size=T)
    sigma_t = rng.uniform(
        low=min(sigma_low, sigma_high), high=max(sigma_low, sigma_high), size=T
    )
    gamma_t = rng.uniform(
        low=min(gamma_low, gamma_high), high=max(gamma_low, gamma_high), size=T
    )

    complexity_scores = compute_complexity_scores(mu_t, sigma_t, gamma_t, or_capacity)

    return mu_t, sigma_t, gamma_t, complexity_scores
