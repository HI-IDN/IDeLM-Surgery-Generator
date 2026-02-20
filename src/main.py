from .generate_all_data import generate_all_data
from .generators import params as pm


def main():
    data = generate_all_data(
        n_rooms=5,
        n_surgeons=10,
        or_capacity=480.0,
        n_operation_cards=100,
        waiting_list_size=50,
        seed=42,
        frequency_params=pm.FrequencyParams(),
        duration_params=pm.DurationParams(),
        schedule_params=pm.ScheduleParams(),
        priority_params=pm.PriorityParams(),
        admission_params=pm.AdmissionParams(),
    )
    print("Data generation complete.")
    print(f"Generated {len(data[5])} surgeries in the waiting list.")


if __name__ == "__main__":
    main()
