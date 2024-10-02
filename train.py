#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys
import os
import pickle
from datetime import datetime 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction import DictVectorizer 
from sklearn.metrics import roc_auc_score, log_loss, root_mean_squared_error
import xgboost as xgb
from hyperopt import fmin, hp, tpe, Trials
from hyperopt.pyll import scope

import mlflow

from helpers import cfg, supports, load_transform_predict


# In[2]:


# initialize configuration
# mode=test will minimize parameters (ex: 20 VS 1000 booster runs)
# skip_optimization=True will skip model tuning
CONFIG = cfg.init_config(mode="test", skip_optimization=False)

# output config
print("---CONFIG:")
for k, v in CONFIG.items():
    print(f"---{k} > {v}")


# In[3]:


# # launch mlflow
# mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root mlflow
experiment = "political_engagement"
artifact_location = CONFIG["mlflow_artifacts_path"]

mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
mlflow.set_experiment(experiment)
supports.set_mlflow_artifact_location(
    "./mlflow/mlflow.db",
    experiment,
    artifact_location
)


# In[4]:


# read data
data = load_transform_predict.Loader.load_live_data()


# In[5]:


# setup training context
# get datasets
dftrain, dftest = train_test_split(data, test_size=0.2, random_state=99)

# get targets
ytrain = dftrain["political_engagement"].values
ytest = dftest["political_engagement"].values
dftrain.drop(columns=["political_engagement"], inplace=True)
dftest.drop(columns=["political_engagement"], inplace=True)

# vectorize
dv = DictVectorizer(sparse=False)
train_dict = dftrain.to_dict(orient="records")
test_dict = dftest.to_dict(orient="records")
xtrain = dv.fit_transform(train_dict)
xtest = dv.transform(test_dict)

# get dmatrix
xtrain = xgb.DMatrix(xtrain, label=ytrain)
xtest = xgb.DMatrix(xtest, label=ytest)


# In[6]:


# optimize
if not CONFIG["skip_optimization"]:
    search_space = {
        "learning_rate": hp.loguniform("learning_rate", -7, 10),
        "max_depth": scope.int(hp.quniform("max_depth", 0, 100, 1)),
        "min_child_weight": hp.loguniform("min_child_weight", -1, 4.6),
        "reg_alpha": hp.loguniform("reg_alpha", -5, 4.6), 
        "scale_pos_weight": hp.loguniform("scale_pos_weight", 0, 4.6),
        "objective": "binary:logistic",
        "seed": 99
    }

    best_result = fmin(
        fn=lambda search_space: supports.objective(
            search_space=search_space,
            xtrain=xtrain,
            xtest=xtest,
            ytrain=ytrain,
            ytest=ytest,
            num_boost_round=CONFIG["booster_rounds"]
            ),
        space=search_space,
        algo=tpe.suggest,
        max_evals=CONFIG["mlflow_evals_nbr"],
        trials=Trials()
        )


# In[7]:


# save the model with the best params
artifacts_path = "./mlflow"
tags = {
        "author": "andrei lupascu",
        "mode": CONFIG["mode"]
}
# update best_result
best_result["objective"] = search_space["objective"]
best_result["seed"] = search_space["seed"]
# format best results (some params need to be cast as int)
int_params = ["max_depth"]
for int_param in int_params:
        best_result[int_param] = int(best_result[int_param])

supports.objective(
     search_space=best_result,
     xtrain=xtrain,
     xtest=xtest,
     ytrain=ytrain,
     ytest=ytest,
     num_boost_round=CONFIG["booster_rounds"],
     tags=tags,
     save_artifacts=(True, artifacts_path, dv)
     )
    


# In[ ]:




