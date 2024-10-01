import shutil
import os
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer 
import pandas as pd
from typing import Tuple

from helpers import mageai_supports

class DriftHandler():
    """
    Detects data drifts between live data and incoming batches.
    """

    def __init__(self):
        self.live_data = DriftHandler.get_live_data()
        self.artifacts_folder_path = "./mlflow"
        self.model, self.dv = self.load_artifacts()

    @staticmethod
    def get_live_data(
            live_data_path: str="./data/training_data/live_data.parquet", 
            backup_data_path: str="./data/training_data/starting_data/production_data.parquet"
            ) -> pd.DataFrame:
        
        if not os.path.exists(live_data_path):
            print("Live data not found. Creating data...")
            shutil.copy(backup_data_path, live_data_path)

        print(f"Loaded: {live_data_path}")
        df = pd.read_parquet(live_data_path)
        return df
    
    def load_artifacts(self) -> Tuple[xgb.Booster, DictVectorizer]:
        model, dv = mageai_supports.load_artifacts(self.artifacts_folder_path)
        return model, dv

    def detect_drift(self, batch_df: pd.DataFrame):
        pass

