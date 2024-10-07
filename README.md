# Political Engagement (MLOPS Simulation)
This project simulates a real-world application where a model predicts the political engagement of individuals.
Data comes in batches from the field, predictions are appended to it and sent to the contact centers in an expected format. The focus here is on Machine Learning Operations.
<br/>

## TLDR;
* Duplicate Repo 
* Build Docker-Compose image with: **docker-compose build**
* Launch container with **docker up**
* Explore scenarios on **localhost:6789** (path_to_scenarios)
* Observe effects in model registry **localhost:5000**
<br/>

## Context
XGButler is - ficticious - organization gathering data on the general population.
Some of the variables they are looking into are the education and income levels, public beliefs, political engagement...
They trained an **XGBoost** model to predict the political engagement of their subjects based on the prior features.
Their aim is to identify **politically disengaged** people and attempte to re-engage them.
Why? They believe that a nation composed of politically engaged and educated individuals fares better in the longterm.
How? The pilosophy of this organization is on the... debatable side of things. The batches originate from both reputable and less reputable sources. The organization leans on "the end justifies the means" approach.
<br/>

## Data 
The original data comes from https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp.
This data has been heavily transformed and customized in a previous project: https://github.com/celupa/political-engagement-analysis
<br/>

## Flow
The initial study offered the data on wich the starting XGB model has been trained.
However, since this is a continuous study, additional data keeps coming from the field in batch format.
These batches are fed to the model for predictions. Between the batch and the model resides the *omnious overseer* assessing wether or not the new batch data has drifted (new values in the dataset, prediction drift...).
If there happens to be data drift, an automated correcting mechanism will look for new data, assimilate it to the original training data and retrain a fresh model. 
Automated parameter tuning will make sure the best model is picked for the job. 
<image=/>
<br/>

## How to 

