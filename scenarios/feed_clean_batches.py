import os 
import shutil
from helpers import load_transform_predict


ARTIFACTS_FOLDER_PATH = "./mlflow"
PREDS_PATH = "./data/batch_data/predictions"
NEW_BATCHES_PATH = "./data/batch_data/new_batches"


def predict_prod_batches(
        source_path: str= "./data/batch_data/testing_batches",
        dest_path: str="./data/batch_data/new_batches",
        ) -> None:
    """Transfer prod testing_batches to new_batches. 
    If reset, delete the batches from new_batches after predictions."""

    for batch in os.listdir(source_path):
        # target prod batches
        if "prod" in batch:
            batch_source = f"{source_path}/{batch}"
            batch_dest = f"{dest_path}/{batch}"
            # copy batches from testing_batches to new_batches
            shutil.copy(batch_source, batch_dest)
            # predict
            load_transform_predict.predict(
                ARTIFACTS_FOLDER_PATH,
                NEW_BATCHES_PATH,
                PREDS_PATH
                )


if __name__ == "__main__":
    predict_prod_batches()