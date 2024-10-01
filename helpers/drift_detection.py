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

    def detect_drift(live_data: pd.DataFrame, batch: pd.DataFrame):
        pass

