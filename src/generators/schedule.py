from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .params import ScheduleParams
from ..type_aliases import (
    OperationCard,
    Room,
    Schedule,
    Surgeon,
    Weekday,
)


def generate_schedule(
    frequency_data: Dict[Tuple[OperationCard, Surgeon], float],
    rooms: List[Room],
    weekdays: List[Weekday],
    params: ScheduleParams,
    rng: Optional[np.random.Generator] = None,
) -> Schedule:
    """Generate a schedule based on surgeon frequencies and room availability.

    Parameters
    ----------
    frequency_data : Dict[Tuple[OperationCard, Surgeon], float]
        Dictionary mapping (operation card, surgeon) pairs to their frequencies.
        This indicates how often each surgeon performs each operation card.
    rooms : List[Room]
        List of available rooms for surgeries.
    weekdays : List[Weekday]
        List of weekdays for scheduling surgeries.
    params : ScheduleParams
        Parameters for schedule generation
    rng: Optional[np.random.Generator], optional
        An optional random number generator instance for reproducibility, by default None

    Returns
    -------
    Schedule
        A dictionary mapping (surgeon, room, weekday) to the fraction of slots assigned to that surgeon
        in that room on that weekday. The values are normalized to sum to 1 across all surgeons for each room and weekday.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Step 1: Compute total frequency per surgeon
    surgeon_totals: Dict[Surgeon, float] = {}
    for (_, surgeon), freq in frequency_data.items():
        surgeon_totals[surgeon] = surgeon_totals.get(surgeon, 0.0) + freq

    total_weight = sum(surgeon_totals.values())
    surgeons = list(surgeon_totals.keys())

    # Step 2: Create all slots: (room, weekday, slot_index)
    slots: List[Tuple[Room, Weekday, int]] = [
        (room, day, slot_idx)
        for day in weekdays
        for room in rooms
        for slot_idx in range(params.slots_per_day)
    ]
    total_slots = len(slots)

    # Step 3: Calculate number of slots per surgeon, with randomness
    slots_per_surgeon: Dict[Surgeon, int] = {}
    for s in surgeons:
        base = (surgeon_totals[s] / total_weight) * total_slots
        variation = 1.0 + rng.uniform(
            -params.slot_randomness_scale, params.slot_randomness_scale
        )
        slots_per_surgeon[s] = max(1, int(base * variation))

    # Step 4: Greedy compact assignment using randomized room-day buckets
    # Track per-surgeon, per-weekday assignment counts
    surgeon_weekday_counts: Dict[Surgeon, Dict[Weekday, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    slot_buckets: Dict[Tuple[Room, Weekday], List[int]] = defaultdict(list)
    for room, day, slot_idx in slots:
        slot_buckets[(room, day)].append(slot_idx)

    slot_assignments: Dict[Tuple[Room, Weekday, int], Surgeon] = {}
    used_slots = set()
    sorted_surgeons = sorted(
        surgeons, key=lambda s: -surgeon_totals[s]
    )  # busiest first

    for surgeon in sorted_surgeons:
        remaining = slots_per_surgeon[surgeon]
        room_day_keys = list(slot_buckets.keys())
        rng.shuffle(room_day_keys)  # random order per surgeon

        for room, day in room_day_keys:
            # Skip if surgeon already has max slots for this weekday
            if surgeon_weekday_counts[surgeon][day] >= params.slots_per_day:
                continue
            free_slots = [
                idx
                for idx in slot_buckets[(room, day)]
                if (room, day, idx) not in used_slots
            ]
            for slot_idx in free_slots:
                # stop if we have reached the day's limit
                if surgeon_weekday_counts[surgeon][day] >= params.slots_per_day:
                    break
                # stop if surgeon has enough slots
                if remaining <= 0:
                    break
                slot = (room, day, slot_idx)
                slot_assignments[slot] = surgeon
                used_slots.add(slot)
                remaining -= 1
                surgeon_weekday_counts[surgeon][day] += 1
            if remaining <= 0:
                break

    # Step 5: Entropy - swap slot assignments between surgeons
    num_swaps = int(params.entropy * len(slot_assignments))
    assigned_slots = list(slot_assignments.keys())

    for _ in range(num_swaps):
        a, b = map(tuple, rng.choice(assigned_slots, 2, replace=False))
        sa, sb = slot_assignments[a], slot_assignments[b]
        if sa != sb:
            slot_assignments[a], slot_assignments[b] = sb, sa

    # Step 6: Aggregate into final schedule (surgeon, room, weekday)
    counts: Dict[Tuple[Surgeon, Room, Weekday], int] = defaultdict(int)
    for (room, day, _), surgeon in slot_assignments.items():
        counts[(surgeon, room, day)] += 1

    total_assigned = sum(counts.values())
    schedule: Schedule = {key: val / total_assigned for key, val in counts.items()}

    return schedule
