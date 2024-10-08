if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from datetime import datetime
from helpers import load_transform_predict, dossier


@custom
def predict(*args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your custom logic here

    batch_logic = load_transform_predict.Predictor(
        dossier.ARTIFACTS_FOLDER_PATH,
        dossier.NEW_BATCHES_PATH,
        dossier.PREDS_PATH
        )
    
    outcome = batch_logic.predict_batches()
    return outcome


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
