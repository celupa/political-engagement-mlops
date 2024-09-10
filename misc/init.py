from config import config_me, default_config


def init_config():
    """Initialize configuration by either defaulting to the demo setup or user
    defined setup."""

    user_config_completion = 0
    missed_settings = []

    # if user doesn't provide variables (data path...), pick default config
    for key, val in config_me.USER_CONFIG.items():
        setting_length = len(val)
        user_config_completion += setting_length

        if setting_length == 0:
            missed_settings.append(key)
        
    if user_config_completion == 0:
        print("User configuration not found. Loading default settings.")
        return default_config.DEFAULT_CONFIG
    elif len(missed_settings) > 0:
        print(f"Missing the following settings: {', '.join(missed_settings)}")
    
    return config_me.USER_CONFIG


if __name__ == "__main__":
    init_config()
