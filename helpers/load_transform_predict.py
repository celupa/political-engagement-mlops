import pickle
import os
import shutil

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb

from typing import Tuple

from helpers.drift_detection import DriftHandler


class Loader():
    @staticmethod
    def _load_model(model_path: str) -> xgb.Booster:
        model = xgb.Booster()
        model.load_model(model_path)
        return model

    @staticmethod
    def _load_preprocessor(preprocessor_path: str) -> DictVectorizer:
        with open(preprocessor_path, "rb") as fin:
            dv = pickle.load(fin)
        return dv

    @staticmethod 
    def load_artifacts(artifacts_folder_path: str) -> Tuple[xgb.Booster, DictVectorizer]:
        """
        Load the latest xgb model and the related vectorizer.
        This wrapper exists because the name of the model & vectorizer changes constantly.
        """

        data = {"xgb": Loader._load_model, "bin": Loader._load_preprocessor}
        loaded_artifacts = {}

        for file in os.listdir(artifacts_folder_path):
            if file.endswith("xgb") or file.endswith("bin"):
                extension = file.split(".")[1]
                file_path = f"{artifacts_folder_path}/{file}" 
                loaded_artifacts[extension] = data[extension](file_path)

        model, dv = loaded_artifacts["xgb"], loaded_artifacts["bin"]
        return model, dv 
    
    @staticmethod
    def load_live_data(
            live_data_path: str="./data/training_data/live_data.parquet", 
            backup_data_path: str="./data/training_data/starting_data/production_data.parquet"
            ) -> pd.DataFrame:
        
        if not os.path.exists(live_data_path):
            print("Live data not found. Creating data...")
            shutil.copy(backup_data_path, live_data_path)

        print(f"Loaded: {live_data_path}")
        df = pd.read_parquet(live_data_path)
        return df

class Transformer():
    @staticmethod
    def get_dmatrix(df: pd.DataFrame, dv: DictVectorizer) -> pd.DataFrame:
        """Get the XGB DMatrix of a dataframe."""

        df_dict = df.to_dict(orient="records")
        df_vect = dv.transform(df_dict)
        df_xgbdm = xgb.DMatrix(df_vect)
        
        return df_xgbdm

class Predictor():
    def __init__(
        self,
        artifacts_folder_path: str,
        new_batches_folder_path: str,
        predictions_output_path: str
        ):
        self.artifacts_folder_path = artifacts_folder_path
        self.model, self.dv = Loader.load_artifacts(self.artifacts_folder_path)
        self.new_batches_folder_path = new_batches_folder_path
        self.predictions_output_path = predictions_output_path
        self.live_data = self.predict(Loader.load_live_data())
        
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for new batches."""
        
        df_dmat = Transformer.get_dmatrix(df, self.dv)
        # predict (0=subject doesn't need intervention (-), 1=subject could benefit from intervention (CONTACT))
        df["prediction"] = self.model.predict(df_dmat)
        return df
    
    def predict_batches(self):
        """Build on self.predict by outputing a more customized output."""

        for batch in os.listdir(self.new_batches_folder_path):
            # label data
            new_batch_path = f"{self.new_batches_folder_path}/{batch}"
            batch_name = batch.split(".")[0]
            output_batch_name = f"{batch_name}_predicted.parquet"
            predicted_batch_path = f"{self.predictions_output_path}/{output_batch_name}"
            # read data and wrangle data
            print(f"Reading: {new_batch_path}")
            batch_df = pd.read_parquet(new_batch_path)
            print("Predicting...")
            batch_df = self.predict(batch_df)
            # handle drift
            drift_report = DriftHandler.detect_drift(self.live_data, batch_df)
            if "drift":
                pass 
            batch_df["prediction_simplified"] = np.round(batch_df.prediction).astype(int)
            batch_df["outcome"] = batch_df.prediction_simplified.replace([0, 1], ["-", "CONTACT"])
            # make things easier for the agents by only keeping the target subjects
            # moreover, keep only essential contact info
            batch_df = batch_df[batch_df.outcome == "CONTACT"]
            end_user_df = batch_df[[
                "country", 
                "sex", 
                "education",
                "citizenship",
                "subject_id", 
                "prediction"
                ]]
            # output predicted batch
            print(f"Writing predictions to: {predicted_batch_path}")
            end_user_df.to_parquet(predicted_batch_path, index=False)
            # remove new batch
            os.remove(new_batch_path)
