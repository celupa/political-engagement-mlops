from misc import config_me, default_config


def init_config():
    """Initialize configuration by either defaulting to the demo setup or user
    defined setup."""
    CONFIG = {}

    # if user doesn't provide variables (data path...), pick default config
    for key, val in config_me.USER_CONFIG.items():
        if len(val) == 0:
            CONFIG[key] = default_config.DEFAULT_CONFIG[key]
    
    return CONFIG


if __name__ == "__main__":
    init_config()
