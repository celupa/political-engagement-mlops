import os
from sys import argv
from scripts import data_generation


def reset_data(
        predictions_folder: str="./data/batch_data/predictions",
        hard_reset: str="false"
        ) -> None:
    """
    Clear predictions folder.
    If hard_reset, recreate datasets and batches.
    """
    
    pfolder_size = len(os.listdir(predictions_folder))

    if pfolder_size > 0:
        for file in os.listdir(predictions_folder):
            file_path = f"{predictions_folder}/{file}"
            os.remove(file_path)
        print("---Flushed predictions")
    else:
        print("---No predictions to delete")

    if len(argv) > 1:
        hard_reset = argv[1]

    if hard_reset == "true":
        data_generation.generate_data()
        data_generation.generate_batches()
        print("---Recreated datasets")


if __name__ == "__main__":
    reset_data()