import shutil
import os
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer 
import pandas as pd
from typing import Tuple

class DriftHandler():
    """
    Detects data drifts between live data and incoming batches.
    """

    def __init__(self):
        self.live_data = DriftHandler.get_live_data()

    def detect_drift(self, live_data: pd.DataFrame, batch: pd.DataFrame):
        pass

