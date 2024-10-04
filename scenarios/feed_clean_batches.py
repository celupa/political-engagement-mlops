import os 
import shutil
from helpers import load_transform_predict, dossier


def predict_prod_batches(
        source_path: str=dossier.TEST_BATCHES_LOCATION,
        dest_path: str=dossier.NEW_BATCHES_LOCATION,
        ) -> None:
    """Transfer prod testing_batches to new_batches and predict the batches"""

    batch_logic = load_transform_predict.Predictor(
        dossier.ARTIFACTS_LOCATION,
        dossier.NEW_BATCHES_LOCATION,
        dossier.PREDICTIONS_LOCATION
        )
    
    for batch in os.listdir(source_path):
        # target prod batches
        if "prod" in batch:
            batch_source = f"{source_path}/{batch}"
            batch_dest = f"{dest_path}/{batch}"
            # copy batches from testing_batches to new_batches
            shutil.copy(batch_source, batch_dest)

    # predict
    batch_logic.predict_batches()


if __name__ == "__main__":
    predict_prod_batches()