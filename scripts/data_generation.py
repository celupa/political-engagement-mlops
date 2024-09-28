import pandas as pd
import os
import uuid 


def generate_data(original_data_path: str="./data/original_data.parquet") -> None:
    """Split the original_data into 2 subsets for testing (prod and new)"""

    # read data
    original_data = pd.read_parquet(original_data_path)
    # split data 
    prod_data = original_data[original_data.country.astype(int) < 500]
    new_data = original_data[original_data.country.astype(int) >= 500]
    # save data
    data_path = os.path.dirname(original_data_path)
    prod_data.to_parquet(f"{data_path}/training_data/starting_data/production_data.parquet")
    new_data.to_parquet(f"{data_path}/training_data/new_data/new_data.parquet")

def generate_batches(
        prod_data_path: str="./data/training_data/starting_data/production_data.parquet",
        new_data_path: str="./data/training_data/new_data/new_data.parquet",
        batches_path: str="./data/batch_data/testing_batches"
        ) -> None:
    # read data
    prod_data = pd.read_parquet(prod_data_path)
    new_data = pd.read_parquet(new_data_path)

    # shuffle data
    sprod_data = prod_data.sample(frac=1).reset_index(drop=True)
    snew_data = new_data.sample(frac=1).reset_index(drop=True)
    df_dict = {"prod_data" : sprod_data, 
               "new_data": snew_data}

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
