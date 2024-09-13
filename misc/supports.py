import pandas as pd

import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss, root_mean_squared_error

from hyperopt import STATUS_OK

import mlflow

from typing import Any


def get_objname(obj: Any) -> str:
    """Retrieve the name of an object (for formating purposes)."""

    obj_name =[v for v in globals() if globals()[v] is obj][0]    
    return obj_name

def objective(
        search_space: dict,
        xtrain: xgb.DMatrix,
        xtest: xgb.DMatrix,
        ytrain: pd.Series,
        ytest: pd.Series,
        num_boost_round: int
        ) -> dict:
    """Set-up the mlflow/hyperopt optimization process."""

    with mlflow.start_run():
        mlflow.log_params(search_space)
        booster = xgb.train(
            params=search_space,
            dtrain=xtrain,
            num_boost_round=num_boost_round,
            evals=[(xtest, "test")],
            early_stopping_rounds=50
            )
        
        # get predictions
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

        return {"loss": test_log_loss, "status": STATUS_OK}
        