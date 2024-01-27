from altair import Union
import yaml
import logging
import logging.config
import pandas as pd


from pathlib import Path


logging_config_path=Path(__file__).parent.parent.parent / 'config' / 'logging_config.yml'


def setup_logging(module_name):
    # Load the logging configuration from the YAML file
    with open(logging_config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    logger = logging.getLogger('default')
    # Configure logging for the specific module
    if module_name in config['loggers']:
        logging.config.dictConfig(config)
        logger = logging.getLogger(module_name)
    else:
        logging.basicConfig(level=logging.WARNING)  # Fallback if module not found in config

    return logger

logger = setup_logging(__name__)

def read_data(file_config: dict) -> pd.DataFrame:
    data = pd.DataFrame()
    try:
        if file_config["type"] == "csv":
            data = pd.read_csv(file_config["path"], sep=file_config["sep"])
    except Exception as e:
        logger.error(f"Error in read_data: {e}")

    return data