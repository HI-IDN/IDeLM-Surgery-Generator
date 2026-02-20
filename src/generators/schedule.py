from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..type_aliases import OperationCard, Room, Schedule, Surgeon, Weekday
from .params import ScheduleParams


def generate_schedule(
    frequency_data: Dict[Tuple[OperationCard, Surgeon], float],
    rooms: List[Room],
    weekdays: List[Weekday],
    params: ScheduleParams,
    rng: Optional[np.random.Generator] = None,
) -> Schedule:
    """
    Generate master surgical schedule as desirability scores.

    Master Surgical Schedule
    -------------------------
    This function generates a "master surgical schedule" that captures surgeon
    preferences and availability patterns. For each surgeon, it assigns desirability
    weights to (room, weekday) pairs, representing:

    - Historical patterns: Which room-day combinations a surgeon typically uses
    - Preferences: Preferred operating rooms and days
    - Availability: When/where a surgeon is available to operate

    The output serves as a soft constraint in surgical scheduling optimization:
    - High weight → surgeon typically operates here (preferred assignment)
    - Low weight → surgeon rarely operates here (unusual, may incur penalty)
    - Zero weight → surgeon never operates here (hard constraint)

    Dirichlet-Based Generation
    ---------------------------
    Each surgeon's preferences are sampled from a Dirichlet distribution over all
    (room, weekday) pairs. The concentration parameter controls schedule focus:

    - Low concentration → Focused schedule (surgeon has 1-2 primary room-day combos)
    - High concentration → Spread out schedule (surgeon operates across many combos)

    Busier surgeons optionally have more spread-out schedules (controlled by
    workload_scaling parameter), reflecting that high-volume surgeons need more
    operating time and thus use more room-day slots.

    The sparsity_threshold parameter zeros out low-probability assignments, creating
    realistic sparse schedules where surgeons operate only on specific days.

    Parameters
    ----------
    frequency_data : Dict[Tuple[OperationCard, Surgeon], float]
        Dictionary mapping (operation card, surgeon) pairs to their frequencies.
        Used to compute each surgeon's total workload (sum of frequencies).
    rooms : List[Room]
        List of available operating rooms.
    weekdays : List[Weekday]
        List of weekdays for scheduling.
    params : ScheduleParams
        Parameters controlling schedule generation (concentration, workload scaling,
        sparsity threshold).
    rng : Optional[np.random.Generator]
        Random number generator for reproducibility.

    Returns
    -------
    Schedule
        Dictionary mapping (surgeon, room, weekday) tuples to desirability weights.
        For each surgeon, weights sum to 1.0 across all (room, day) pairs.
        Only non-zero weights are included (sparse representation).

        Example:
        {
            ("Surgeon_A", "OR_1", "Monday"): 0.45,      # Primary slot
            ("Surgeon_A", "OR_1", "Wednesday"): 0.40,   # Secondary slot
            ("Surgeon_A", "OR_2", "Monday"): 0.10,      # Occasional
            ("Surgeon_A", "OR_3", "Friday"): 0.05,      # Rare
            ("Surgeon_B", "OR_2", "Tuesday"): 0.70,     # Highly focused
            ("Surgeon_B", "OR_2", "Thursday"): 0.25,
            ("Surgeon_B", "OR_1", "Tuesday"): 0.05,
        }

    Notes
    -----
    This representation provides a probabilistic desirability measure that can be
    used in optimization models as:

    - Objective function weights (maximize assignment to preferred room-days)
    - Soft constraint penalties (penalize deviation from typical patterns)
    - Feasibility checks (zero weight = infeasible assignment)

    The Dirichlet-based approach ensures:
    - Weights are non-negative and sum to 1 per surgeon (valid probability distribution)
    - Natural variance between surgeons (different concentration parameters)
    - Consistency with other generator modules (frequency, duration all use Dirichlet)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Step 1: Compute total workload per surgeon
    surgeon_workloads: Dict[Surgeon, float] = {}
    for (_, surgeon), freq in frequency_data.items():
        surgeon_workloads[surgeon] = surgeon_workloads.get(surgeon, 0.0) + freq

    surgeons = list(surgeon_workloads.keys())

    # Step 2: Create all (room, weekday) pairs
    room_day_pairs = [(room, day) for day in weekdays for room in rooms]
    num_slots = len(room_day_pairs)

    if num_slots == 0:
        raise ValueError(
            "No room-day pairs available. Provide non-empty rooms and weekdays."
        )

    # Step 3: Generate schedule for each surgeon via Dirichlet sampling
    schedule: Schedule = {}

    for surgeon in surgeons:
        workload = surgeon_workloads[surgeon]

        # Compute concentration: busier surgeons have lower concentration (more spread out)
        # concentration = base / (1 + scaling * workload)
        concentration = params.base_concentration / (
            1.0 + params.workload_scaling * workload
        )

        # Sample from Dirichlet: weights for each (room, day) pair
        alpha = np.full(num_slots, concentration)
        weights = rng.dirichlet(alpha)

        # Apply sparsity threshold: zero out low-probability assignments
        if params.sparsity_threshold > 0:
            weights[weights < params.sparsity_threshold] = 0.0

            # Renormalize (only if there are non-zero weights remaining)
            total = weights.sum()
            if total > 0:
                weights = weights / total
            else:
                # If all weights were below threshold, keep the largest one
                max_idx = np.argmax(rng.random(num_slots))  # Random fallback
                weights = np.zeros(num_slots)
                weights[max_idx] = 1.0

        # Store non-zero weights in flat dictionary with (surgeon, room, day) keys
        for (room, day), weight in zip(room_day_pairs, weights):
            if weight > 0:
                schedule[(surgeon, room, day)] = float(weight)

    return schedule
