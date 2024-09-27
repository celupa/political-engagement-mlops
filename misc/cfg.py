# from config import config_me, default_config


def init_config(mode: str="test", skip_optimization: bool=True) -> dict:
    """Initialize configuration. If user doesn't provide config, use default settings."""

    CONFIG = {
        "test": {
            "mode": mode,
            "mlflow_evals_nbr": 2,
            "booster_rounds": 10,
            "skip_optimization": skip_optimization
        },
        "prod": {
            "mode": mode,
            "mlflow_evals_nbr": 100,
            "booster_rounds": 1000,
            "skip_optimization": skip_optimization
        }
    }

    return CONFIG[mode]
    # user_config_completion = 0
    # missed_settings = []

    # for key, val in config_me.USER_CONFIG.items():
    #     setting_length = len(val)
    #     user_config_completion += setting_length

    #     if setting_length == 0:
    #         missed_settings.append(key)
        
    # if user_config_completion == 0:
    #     print("---User configuration not found. Loading default settings.")
    #     return default_config.DEFAULT_CONFIG
    # elif len(missed_settings) > 0:
    #     print(f"---Using default profile because the following settings are missing: {', '.join(missed_settings)}")
    #     return default_config.DEFAULT_CONFIG
    
    # return config_me.USER_CONFIG


if __name__ == "__main__":
    init_config()
