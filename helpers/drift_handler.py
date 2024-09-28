import pandas as pd
import shutil
import os


def get_live_data(
        live_data_path: str="./data/training_data/live_data.parquet", 
        backup_data_path: str="./data/training_data/starting_data/production_data.parquet"
        ) -> pd.DataFrame:
    
    if not os.path.exists(live_data_path):
        print("Live data not found. Creating data...")
        shutil.copy(backup_data_path, live_data_path)

    print(f"Loaded: {live_data_path}")
    df = pd.read_parquet(live_data_path)

    return df
