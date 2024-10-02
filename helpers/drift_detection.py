import pandas as pd

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric

import os 
import sys

from helpers import dossier


class DriftHandler():
    """Detects data drifts between live data and incoming batches."""
    
    def __init__(
        self, 
        prod_data_path: str=dossier.PROD_DATA_LOCATION,
        new_data_path: str=dossier.NEW_DATA_LOCATION,
        live_data_path: str=dossier.LIVE_DATA_LOCATION
        ):
        
        self.prod_data_path = prod_data_path
        self.new_data_path = new_data_path
        self.live_data_path = live_data_path

    def detect_drift(self, live_data: pd.DataFrame, batch: pd.DataFrame):
        """Awesome description."""
        
        print("---Checking for data drift...")
        # label columns
        cat_cols = [col for col in live_data.columns if live_data[col].dtype == "object"]
        num_cols = [col for col in live_data.columns if live_data[col].dtype != "object"]
        all_cols = cat_cols + num_cols
        
        # setup evidently
        column_mapping = ColumnMapping(
        target=None,
        prediction="prediction",
        numerical_features=num_cols,
        categorical_features=cat_cols
        )

        self.report = Report(metrics=[
        ColumnDriftMetric(column_name="prediction"),
        DatasetDriftMetric(columns=all_cols, drift_share=0.1)
        ])
        
        self.report.run(
        reference_data=live_data, 
        current_data=batch, 
        column_mapping=column_mapping
        )
        
        retrain_model = self.report_drift(self.report)
        return retrain_model
    
    def report_drift(self, evidently_report: Report):
        self.results = evidently_report.as_dict()
        
        pred_drift_results = self.results["metrics"][0]
        pred_drift_status = pred_drift_results["result"]["drift_detected"]
        pred_drift_col = pred_drift_results["result"]["column_name"]
        pred_drift_test = pred_drift_results["result"]["stattest_name"]
        pred_drift_threshold = pred_drift_results["result"]["stattest_threshold"]
        pred_drift_score = pred_drift_results["result"]["drift_score"]

        dataset_drift_results = self.results["metrics"][1]
        dataset_drift_status = dataset_drift_results["result"]["dataset_drift"]
        dataset_drift_columns = dataset_drift_results["result"]["number_of_drifted_columns"]
        dataset_drift_treshold = dataset_drift_results["result"]["drift_share"]
        dataset_drift_share = dataset_drift_results["result"]["share_of_drifted_columns"]

        prediction_drift_warning = f"""---Prediction Drift Detected: 
        Target name: {pred_drift_col}
        Test used: {pred_drift_test}
        Test threshold: {pred_drift_threshold}
        Test score: {pred_drift_score}
        """
        dataset_drift_warning = f"""---Dataset Drift Detected: 
        Columns drifted: {dataset_drift_columns}
        Drift threshold: {dataset_drift_treshold}
        Drift Share: {dataset_drift_share}
        """

        if pred_drift_status > 0:
            print(prediction_drift_warning)
        if dataset_drift_status > 0:
            print(dataset_drift_warning)  
        
        retrain_model = pred_drift_status + dataset_drift_status
        return retrain_model
      
    def retrain_model(self):
        print("---Retraining model...")
        live_data = pd.read_parquet(self.live_data_path)
        
        try:
            new_data = pd.read_parquet(self.new_data_path)
        except:
            print("---No new data found. Please reach out the MLOPS team.")
            
        live_data = pd.concat([live_data, new_data])
        live_data.to_parquet(self.live_data_path, index=False)
        os.remove(self.new_data_path)
        print(f"---New live data wrote to: {self.live_data_path}")
        
        os.system("python train.py")
        
        print("---Model retraining completed.")
