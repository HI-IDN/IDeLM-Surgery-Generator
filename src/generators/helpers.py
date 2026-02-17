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
