import os 
from helpers import dossier 


def init_config(mode: str="test", skip_optimization: bool=True) -> dict:
    """Initialize configuration. If user doesn't provide config, use default settings."""

    CONFIG = {
        "test": {
            "mode": mode,
            "mlflow_evals_nbr": 2,
            "booster_rounds": 10,
            "skip_optimization": skip_optimization,
        },
        "prod": {
            "mode": mode,
            "mlflow_evals_nbr": 100,
            "booster_rounds": 1000,
            "skip_optimization": skip_optimization,
        }
    }

    return CONFIG[mode]


if __name__ == "__main__":
    init_config()
