import yaml
import logging
import logging.config


from pathlib import Path


logging_config_path=Path(__file__).parent.parent.parent / 'config' / 'logging_config.yml'


def setup_logging(module_name):
    # Load the logging configuration from the YAML file
    with open(logging_config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Configure logging for the specific module
    logger = None
    if module_name in config['loggers']:
        logging.config.dictConfig(config)
        logger = logging.getLogger(module_name)
    else:
        logging.basicConfig(level=logging.WARNING)  # Fallback if module not found in config

    return logger