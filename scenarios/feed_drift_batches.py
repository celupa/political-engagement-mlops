import os
import shutil
from helpers import load_transform_predict, dossier


def predict_drift_batches(
        source_path: str=dossier.TEST_BATCHES_LOCATION,
        dest_path: str=dossier.NEW_BATCHES_LOCATION,
        ) -> None:
    """Transfer drift testing_batches to new_batches and predict them."""

    batch_logic = load_transform_predict.Predictor(
        dossier.ARTIFACTS_LOCATION,
        dossier.NEW_BATCHES_LOCATION,
        dossier.PREDICTIONS_LOCATION
        )

    for batch in os.listdir(source_path):
        # target new batches
        if "new" in batch:
            batch_source = f"{source_path}/{batch}"
            batch_dest = f"{dest_path}/{batch}"
            # copy batches from testing_batches to new_batches
            shutil.copy(batch_source, batch_dest)

    # predict
    batch_logic.predict_batches()


if __name__ == "__main__":
    predict_drift_batches()
