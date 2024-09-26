import pandas as pd


def get_data(data_path: str) -> pd.DataFrame:
    df = pd.read_parquet(data_path)

    return df
