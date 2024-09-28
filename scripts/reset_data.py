import os


def reset_data(
        predictions_folder: str="./data/batch_data/predictions",
        hard_reset: bool=False
        ) -> None:
    """
    Reset the data for the project:
        - clear batch folders
        - recreate batches
    If hard_reset:
        - recreate production and new data
    """
    
    for file in os.listdir(predictions_folder):
        file_path = f"{predictions_folder}/{file}"
        os.remove(file_path)


if __name__ == "__main__":
    reset_data()