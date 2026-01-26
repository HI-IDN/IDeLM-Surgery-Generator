from .generate_all_data import generate_all_data
from .generators import params as pm


def main():
    data = generate_all_data(
        n_rooms=5,
        n_surgeons=10,
        n_operation_cards=100,
        waiting_list_size=50,
        seed=42,
        frequency_params=pm.FrequencyParams(),
        duration_params=pm.DurationParams(),
        schedule_params=pm.ScheduleParams(),
        pattern_params=pm.PatternParams(),
        window_params=pm.WindowParams(),
        admission_params=pm.AdmissionParams(),
        waiting_list_params=pm.WaitingListParams(),
        initial_plan_params=pm.InitialPlanParams(),
    )
    print("Data generation complete.")
    print(f"Generated {len(data[7])} surgeries in the waiting list.")


if __name__ == "__main__":
    main()
