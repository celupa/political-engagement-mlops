import os 
import shutil
from helpers import load_transform_predict


ARTIFACTS_FOLDER_PATH = "./mlflow"
NEW_BATCHES_PATH = "./data/batch_data/new_batches"
PREDS_PATH = "./data/batch_data/predictions"


def predict_drift_batches(
        source_path: str= "./data/batch_data/testing_batches",
        dest_path: str="./data/batch_data/new_batches",
        ) -> None:
    """Transfer new testing_batches to new_batches."""

    batch_logic = load_transform_predict.Predictor(
        ARTIFACTS_FOLDER_PATH,
        NEW_BATCHES_PATH,
        PREDS_PATH
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