import os
import pandas as pd
import uuid 
from helpers import dossier


def generate_data(original_data_path: str=dossier.ORIGINAL_DATA_LOCATION) -> None:
    """
    Split data/original_data into 3 datasets:
        - production_data represents the starting data of the project
        - new_data represents drift data coming from the field
        - live_data represents the data the model is trained on and which can be enriched
    """

    # read data
    original_data = pd.read_parquet(original_data_path)
    # split data 
    prod_data = original_data[original_data.country.astype(int) < 500]
    new_data = original_data[original_data.country.astype(int) >= 500]
    # save data
    prod_data.to_parquet(dossier.PRODUCTION_DATA_LOCATION)
    prod_data.to_parquet(dossier.LIVE_DATA_LOCATION)
    new_data.to_parquet(dossier.NEW_DATA_LOCATION)

def generate_batches(
        prod_data_path: str=dossier.PRODUCTION_DATA_LOCATION,
        new_data_path: str=dossier.NEW_DATA_LOCATION,
        batches_path: str=dossier.TEST_BATCHES_LOCATION
        ) -> None:
    """Generate batches containing expected and drift data."""
    
    # read data
    prod_data = pd.read_parquet(prod_data_path)
    new_data = pd.read_parquet(new_data_path)
    # shuffle data
    sprod_data = prod_data.sample(frac=1).reset_index(drop=True)
    snew_data = new_data.sample(frac=1).reset_index(drop=True)
    df_dict = {"prod_data" : sprod_data, 
               "new_data": snew_data}
    # generate batches
    splits = 3
    for df_name, df in df_dict.items():
        lower_bound = 0
        upper_bound = int(round(len(df) / splits, 0))
        for i in range(splits):
            batch_name = f"{df_name}_batch_{i + 1}.parquet"
            subset = df.iloc[lower_bound:upper_bound].copy()
            subset.drop(columns="political_engagement", inplace=True)
            # link subject ID
            subset["subject_id"] = [str(uuid.uuid4()) for i in range(len(subset))]
            # updates slices
            lower_bound = upper_bound
            upper_bound += upper_bound 
            # export batches
            subset.to_parquet(f"{batches_path}/{batch_name}", index=False)
