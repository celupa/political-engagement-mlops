import os


# data locations
ORIGINAL_DATA_LOCATION = "./data/original_data.parquet"
PRODUCTION_DATA_LOCATION = "./data/training_data/starting_data/production_data.parquet"
NEW_DATA_LOCATION = "./data/training_data/new_data/new_data.parquet"
LIVE_DATA_LOCATION = "./data/training_data/live_data.parquet"
# batch locations
TEST_BATCHES_LOCATION = "./data/batch_data/testing_batches"
NEW_BATCHES_LOCATION = "./data/batch_data/new_batches"
PREDICTIONS_LOCATION = "./data/batch_data/predictions"
# artifact location
ARTIFACTS_LOCATION = "./mlflow"
# we use os because the database hardcodes the hostname
MLFLOW_ARTIFACTS_LOCATION = f"{os.getcwd()}/mlflow/mlruns/poleng"
MLFLOW_TRACKING_URI = "sqlite:///mlflow/mlflow.db"
MLFLOW_DB_LOCATION = "./mlflow/mlflow.db"
# prediction label
PREDICTION_COL_NAME = "prediction"
# best results
BEST_RESULTS = {
    "learning_rate": 0.005132133544114984,
    "max_depth": 84.0,
    "min_child_weight": 9.971863457487139,
    "reg_alpha": 2.960929919747157,
    "scale_pos_weight": 1.0299401967261155,
    "objective": "binary:logistic",
    "seed": 99,
    }
