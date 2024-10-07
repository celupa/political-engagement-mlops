# Methodological and Demographic Predictors of Political Engagement
<br/>

## TL;DR
An XGBoost predictor of political engagement has been trained and deployed to AWS Lambda. 
The classifier assesses whether a person would benefit from an intervention aiming to educate on political engagement and stimulate political involvement.

This project mainly **focuses on model deployment**.

Test the model by downloading and running **lambda/test_aws_lambda.py** (python and "requests" package needed).
<br/>

## Context 
The **World Values Survey** (WVS) is an organization that has been gathering socio-economic and psychological data across the world for over 40 years.
Their questionnaire covers features such as personal values, wellbeing, political perceptions and much more.

This is the first part of 3-step project where we attempt to predict the political engagement level of a person via surface dimensions like sex, education and profession.
The second project will analyse political engagement by adding more complex variables to the variable pool (e.g., political perception).
The final project will inspect the evolution of political engagement over the years. 
<br/>

## Rationale
Understanding the variables that make a person politically engaged is valuable for a society and its members.

Whether we are aware of it or not, we are political creatures living in a political world.
By means of a simplified example, both voting and asking for a piece of bread at the table rely on political undertones. Both actions involve one's relationship with their past, their rapport to the current world and shape their future interactions.  

Among other valuable things, political apathy may result in losing the tools useful in nudging one's life in a more fulfilling direction (economic, social...).

Understanding political profiles is the first step in a comprehensive process that would help us to:
* **Inform** about the consequences of political apathy 
* **Prevent** political manipulation by identifying vulnerable populations 
* **Support** intervention campaigns reinvigorating political engagement
* **Protect** systems and citizens from political decay by keeping track of political sentiment over time
<br/>

## Data
The World Values Survey is a questionnaire that covers a wide range of topics.
The data is free to use but not distributable. The pristine version of the data (Wave 7) and other related documentation can be found here: https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp. 
Alternatively, a custom version of the data is available on this repository (misc/custom_wvs7_data.parquet) that can be used in some parts of the analysis. Additionally, variable definitions and other supporting documentation can be found in the **misc** folder.

The 7th wave ran from 2017 to 2022, after which the results were consolidated in one comprehensive dataset. For this project, only a handful of variables have been retained (see **assists/dossier.py**)
<br/>

## How to
The environment used for analysis is stored in **env.yml**. If you have anaconda installed run the following conda cmd (windows): *conda env create -f “path_to_env.yml_file”*. This will create a custom conda environemnt with all the dependencies.

Replicate the analyses via **notebook.ipynb**.

Skip the main analyses and run **train.ipynb** to train the model with the final version of the data (wrangled & imputed) and optimized parameters.

Deploy the app to a local server by first saving the model in train.py with bentoml and running the following cmds at the dev level (docker required):
* **Build**: bentoml build
* **Containerize**: bentoml containerize MODEL_NAME:TAG
* **Serve**: docker run -it --rm -p 3000:3000 MODEL_NAME:TAG

The model has been deployed on **AWS Lambda**. Try it out via **lambda/test_aws_lambda.py**.
<br/>

## Shoutout 
Thank you to the WVS organization for keeping the academic playground accessible and transparent.
<br/>

## Sources
Haerpfer, C., Inglehart, R., Moreno, A., Welzel, C., Kizilova, K., Diez-Medrano J., M. Lagos, P. Norris, E. Ponarin & B. Puranen (eds.). 2022. World Values Survey: Round Seven - Country-Pooled Datafile Version 5.0. Madrid, Spain & Vienna, Austria: JD Systems Institute & WVSA Secretariat. doi:10.14281/18241.20
