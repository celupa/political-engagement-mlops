import pickle
import os

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb

from typing import Tuple


def load_artifacts(model_path: str, vectorizer_path: str) -> Tuple[xgb.Booster, DictVectorizer]:
    """Load the latest xgb model and the related vectorizer."""

    model = xgb.Booster()
    model.load_model(model_path)

    with open(vectorizer_path, "rb") as fin:
        dv = pickle.load(fin)

    return model, dv 

def get_dmatrix(df: pd.DataFrame, dv: DictVectorizer) -> pd.DataFrame:
    """Get the XGB DMatrix of a dataframe."""

    df_dict = df.to_dict(orient="records")
    df_vect = dv.transform(df_dict)
    df_xgbdm = xgb.DMatrix(df_vect)
    
    return df_xgbdm

def predict(
        model_path: str,
        vectorizer_path: str, 
        new_batches_folder_path: str,
        predictions_output_path: str
        ) -> pd.DataFrame:
    """Make predictions for new batches."""

    batch_dict = {}
    model, dv = load_artifacts(model_path, vectorizer_path)

    for batch in os.listdir(new_batches_folder_path):
        # label data
        new_batch_path = f"{new_batches_folder_path}/{batch}"
        batch_name = batch.split(".")[0]
        output_batch_name = f"{batch_name}_predicted.parquet"
        predicted_batch_path = f"{predictions_output_path}/{output_batch_name}"
        # read data and wrangle data
        print(f"Reading: {new_batch_path}")
        df = pd.read_parquet(new_batch_path)
        print("Predicting...")
        df_dmat = get_dmatrix(df, dv)
        # predict (0=subject doesn't need intervention (-), 1=subject could benefit from intervention (CONTACT))
        df["prediction"] = np.round(model.predict(df_dmat)).astype(int)
        df["prediction"] = df.prediction.replace([0, 1], ["-", "CONTACT"])
        # make things easier for the agents by only keeping the target subjects
        # moreover, keep only essential contact info
        df = df[df.prediction == "CONTACT"]
        df = df[[
            "country", 
            "sex", 
            "education",
            "citizenship",
            "subject_id"
            ]]
        # output predicted batch
        print(f"Writing predictions to: {predicted_batch_path}")
        df.to_parquet(predicted_batch_path, index=False)
        # remove new batch
        os.remove(new_batch_path)
        # store predictions (troubleshooting)
        batch_dict[output_batch_name] = df

    return batch_dict 
