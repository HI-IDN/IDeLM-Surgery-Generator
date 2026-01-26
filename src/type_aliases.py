from typing import Dict, Tuple, TypeAlias

Surgeon: TypeAlias = int
Room: TypeAlias = int
Weekday: TypeAlias = int  # 0 for Monday, 1 for Tuesday, ..., 6 for Sunday
Day: TypeAlias = int  # 0 for the first day, 1 for the second day, etc.
OperationCard: TypeAlias = str
OperationId: TypeAlias = int

# A schedule is a mapping from (surgeon_index, room_index, weekday_index)
# to a desirability score.
Schedule: TypeAlias = Dict[Tuple[Surgeon, Room, Weekday], float]

# A pattern is a sequence of surgery cards, represented as a tuple of strings.
Pattern: TypeAlias = Tuple[OperationCard, ...]

# TimeWindow
TimeWindow: TypeAlias = Tuple[int, int]  # (start_time, end_time)
