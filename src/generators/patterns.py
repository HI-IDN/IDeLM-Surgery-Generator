from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .params import PatternParams
from ..type_aliases import (
    OperationCard,
    Pattern,
    Room,
    Schedule,
    Surgeon,
    Weekday,
)


def generate_patterns(
    schedule: Schedule,
    frequency_data: Dict[Tuple[OperationCard, Surgeon], float],
    duration_distributions: Dict[OperationCard, Tuple[float, float]],
    params: PatternParams,
    rng: Optional[np.random.Generator] = None,
) -> Dict[Tuple[Weekday, Room], List[Pattern]]:
    """Generate operation patterns for each room on each weekday based on surgeon availability and operation card frequencies.

    Parameters
    ----------
    schedule : Schedule
        Dictionary mapping (surgeon, room, weekday) to a score indicating the surgeon's availability.
        A score of 0 means the surgeon is not available on that day in that room.
    frequency_data : Dict[Tuple[OperationCard, Surgeon], float]
        Dictionary mapping (operation card, surgeon) pairs to their frequencies.
        This indicates how often each surgeon performs each operation card.
    duration_distributions : Dict[OperationCard, Tuple[float, float]]
        Dictionary mapping operation cards to their duration distributions.
        Each entry contains a tuple of (mean log duration, standard deviation of log duration).
    params : PatternParams
        Parameters for pattern generation
    rng: Optional[np.random.Generator], optional
        An optional random number generator instance for reproducibility, by default None

    Returns
    -------
    Dict[Tuple[Weekday, Room], List[Pattern]]
        A dictionary mapping (weekday, room) pairs to lists of operation patterns.
        Each pattern is a list of operation cards that can be performed on that day in that room.
    """
    if rng is None:
        rng = np.random.default_rng()

    pattern_data: Dict[Tuple[Weekday, Room], List[Pattern]] = defaultdict(list)

    # Surgeon → normalized operation card weights
    surgeon_card_weights: Dict[Surgeon, Dict[OperationCard, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for (card, surgeon), freq in frequency_data.items():
        surgeon_card_weights[surgeon][card] += freq

    # Normalize each surgeon's distribution
    for surgeon, card_dict in surgeon_card_weights.items():
        total = sum(card_dict.values())
        if total > 0:
            for card in card_dict:
                card_dict[card] /= total

    # Build mapping: (room, weekday) → [(surgeon, raw score)]
    room_day_to_surgeons: Dict[Tuple[Weekday, Room], List[Tuple[Surgeon, float]]] = (
        defaultdict(list)
    )
    for (surgeon, room, day), score in schedule.items():
        room_day_to_surgeons[(day, room)].append((surgeon, score))

    for (day, room), surgeon_list in room_day_to_surgeons.items():
        if not surgeon_list:
            continue

        # Normalize scores within this (room, day)
        raw_scores = {s: sc for s, sc in surgeon_list}
        total_score = sum(raw_scores.values())
        if total_score == 0:
            normalized_scores = {s: 1 / len(raw_scores) for s in raw_scores}
        else:
            normalized_scores = {s: sc / total_score for s, sc in raw_scores.items()}

        # Combine operation card weights across all surgeons
        combined_card_weights: Dict[OperationCard, float] = defaultdict(float)
        for surgeon, norm_weight in normalized_scores.items():
            for card, weight in surgeon_card_weights.get(surgeon, {}).items():
                combined_card_weights[card] += norm_weight * weight

        if not combined_card_weights:
            continue

        cards, weights = zip(*combined_card_weights.items())

        for _ in range(params.num_patterns_per_room_day):
            total_duration: float = 0.0
            sequence: List[OperationCard] = []

            while total_duration < params.max_minutes_per_day:
                card = str(rng.choice(cards, p=weights, size=1)[0])
                mean_log, std_log = duration_distributions[card]
                duration = rng.lognormal(mean_log, std_log)

                if total_duration + duration > params.max_minutes_per_day:
                    break

                sequence.append(card)
                total_duration += duration

            if sequence:
                pattern_data[(day, room)].append(tuple(sequence))

    pattern_data = remove_subpatterns(pattern_data)

    return pattern_data


def _is_proper_multiset_subset(a: Counter, b: Counter) -> bool:
    """True if multiset a is a proper subset of multiset b (duplicates respected)."""
    # Quick length check: if a has more elements than b, it cannot be a subset
    if sum(a.values()) > sum(b.values()):
        return False
    # a ⊆ b (multiset) means every count in a is ≤ the corresponding count in b
    if any(a[k] > b[k] for k in a):
        return False
    # Proper subset: not equal as multisets
    return a != b


def remove_subpatterns(
    pattern_data: Dict[Tuple[Weekday, Room], List[Pattern]],
) -> Dict[Tuple[Weekday, Room], List[Pattern]]:
    """Remove patterns that are (proper) sub-multisets of other patterns.

    Subpattern is checked regardless of order, and duplicates are respected.
    Equal-multiset patterns (even in different orders) are NOT removed.
    """
    # Collect unique patterns across all buckets (keeps work bounded by unique count)
    all_patterns = {p for patterns in pattern_data.values() for p in patterns}

    # Precompute Counters and lengths once
    counters = {p: Counter(p) for p in all_patterns}
    lengths = {p: len(p) for p in all_patterns}

    # Sort unique patterns by length ascending so we only compare to same/longer
    unique_sorted = sorted(all_patterns, key=lambda p: lengths[p])

    # For efficient comparison, bucket candidates by length
    buckets_by_len: Dict[int, List[Pattern]] = defaultdict(list)
    for p in unique_sorted:
        buckets_by_len[lengths[p]].append(p)

    # Prepare a set of patterns that survive
    non_subpatterns = set(all_patterns)

    # Build an ordered list of candidate superpatterns for each length L:
    # all patterns with length >= L (concatenated buckets). This avoids O(n^2) on many length-mismatched pairs.
    sorted_lengths = sorted(buckets_by_len)
    suffix_candidates_for_len: Dict[int, List[Pattern]] = {}
    running: List[Pattern] = []
    for L in reversed(sorted_lengths):
        # prepend current bucket to running super-list
        running = buckets_by_len[L] + running
        suffix_candidates_for_len[L] = running

    # Main check: for each pattern, look only at same-or-longer candidates
    for p in unique_sorted:
        if p not in non_subpatterns:
            continue  # already discarded
        p_len = lengths[p]
        p_ctr = counters[p]

        # Compare against candidates with length >= p_len
        for q in suffix_candidates_for_len[p_len]:
            if q is p:
                continue  # identical object; don't compare to itself
            # Quick skip: if q already known to be a subpattern of someone else,
            # it can still be a superpattern of p; don't skip it.
            # Use multiset proper-subset test:
            if _is_proper_multiset_subset(p_ctr, counters[q]):
                non_subpatterns.discard(p)
                break

    # Reassemble filtered dict (preserving original per-(day,room) order)
    filtered_pattern_data: Dict[Tuple[Weekday, Room], List[Pattern]] = defaultdict(list)
    for key, patterns in pattern_data.items():
        filtered_pattern_data[key] = [p for p in patterns if p in non_subpatterns]

    return filtered_pattern_data
