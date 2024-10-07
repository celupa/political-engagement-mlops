import traceback
from scenarios import simulate
from scripts import reset_project


def check_app_pulse() -> None:
    """See if the app is still operational after heavy changes."""

    try:
        simulate.simulate_batches()
        reset_project.reset_data(hard_reset="true")
        print("-----VITALS OK-----")
    except Exception as e:
        print(f"-----SOS-----: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    check_app_pulse()
