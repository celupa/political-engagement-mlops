import os

import pandas as pd

import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss, root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer

from hyperopt import STATUS_OK

import mlflow

from datetime import datetime
import pickle
from typing import Any, Tuple
import sqlite3


def get_objname(obj: Any) -> str:
    """Retrieve the name of an object (for formating purposes)."""

    obj_name =[v for v in globals() if globals()[v] is obj][0]    
    return obj_name

def set_mlflow_artifact_location(
        db_path: str, 
        experiment_name: str,
        artifact_location: str) -> None:
    
    # connect to db
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # set artifact location
    query = """UPDATE experiments
            SET artifact_location = ?
            WHERE name = ?"""
    
    cursor.execute(query, (artifact_location, experiment_name))
    # close connection
    conn.commit()
    conn.close()

def clear_outdated_artifacts(artifacts_path: str) -> None:
    for file in os.listdir(artifacts_path):
        # isolate models and preprocessors
        if file.endswith("xgb") or file.endswith("bin"):
            file_path = f"{artifacts_path}/{file}"
            os.remove(file_path)

def objective(
        search_space: dict,
        xtrain: xgb.DMatrix,
        xtest: xgb.DMatrix,
        ytrain: pd.Series,
        ytest: pd.Series,
        num_boost_round: int,
        tags: dict={},
        save_artifacts: Tuple[bool, str, DictVectorizer]=(False, "", None)
        ) -> dict:
    """
    Set-up the mlflow/hyperopt optimization process.
    
    params:
        save_artifacts: bool, artifacts_path, DictVectorizer
    """

    with mlflow.start_run():
        mlflow.set_tags(tags)
        mlflow.log_params(search_space)
        booster = xgb.train(
            params=search_space,
            dtrain=xtrain,
            num_boost_round=num_boost_round,
            evals=[(xtest, "test")],
            early_stopping_rounds=50
            )
        
        # get metrics
        ytrain_pred = booster.predict(xtrain)
        yval_pred = booster.predict(xtest)
        # get auc
        train_auc = roc_auc_score(ytrain, ytrain_pred)
        test_auc = roc_auc_score(ytest, yval_pred)
        # get loss
        train_log_loss = log_loss(ytrain, ytrain_pred)
        test_log_loss = log_loss(ytest, yval_pred)
        # get rmse
        train_rmse = root_mean_squared_error(ytrain, ytrain_pred)
        test_rmse = root_mean_squared_error(ytest, yval_pred)
        
        # store metrics
        metrics = {
            "train_auc": train_auc,
            "test_auc": test_auc,
            "train_log_loss": train_log_loss,
            "test_log_loss": test_log_loss,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse
            }
        
        # log metrics to mlflow
        for name, metric in metrics.items():
            mlflow.log_metric(name, metric)

        if save_artifacts[0]:
            # tag data
            # yymmddhhmmss
            model_creation_time = datetime.now().strftime("%y%m%d%H%M%S")
            model_name = f"poleng_xgb_{model_creation_time}"
            preprocessor_name = f"preprocessor_xgb_{model_creation_time}"
            artifact_paths = save_artifacts[1]
            dv = save_artifacts[2]
            
            clear_outdated_artifacts(artifact_paths)

            # log artifacts  
            with open(f"{artifact_paths}/{preprocessor_name}.bin", "wb") as fout:
                    pickle.dump(dv, fout)
            booster.save_model(f"{artifact_paths}/{model_name}.xgb")
            mlflow.log_artifact(f"{artifact_paths}/{preprocessor_name}.bin")
            mlflow.log_artifact(f"{artifact_paths}/{model_name}.xgb")
            mlflow.set_tag("model", model_name)

        return {"loss": test_log_loss, "status": STATUS_OK}
        