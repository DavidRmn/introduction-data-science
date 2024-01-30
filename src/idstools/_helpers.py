from altair import Union
import yaml
import logging
import logging.config
import pandas as pd


from pathlib import Path


logging_config_path=Path(__file__).parent.parent.parent / 'config' / 'logging' / 'config.yml'

def setup_logging(module_name):
    package_root = Path(__file__).resolve().parent.parent.parent
    log_file_path = package_root / 'results' / 'idstools.log'
    # Ensure the /results directory exists
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    # Load the logging configuration from the YAML file
    with open(logging_config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Update the filename in the file_handler
    if 'handlers' in config['default'] and 'file_handler' in config['default']['handlers']:
        config['default']['handlers']['file_handler']['filename'] = str(log_file_path)

    logger = logging.getLogger('default')
    # Configure logging for the specific module
    if module_name in config['default']['loggers']:
        logging.config.dictConfig(config['default'])
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

def write_data(data: pd.DataFrame, output_path: Path, filename: str):
    try:
        data.to_csv(output_path / filename, index=False)
        logger.info(f"Data written to: {output_path / filename}")
    except Exception as e:
        logger.error(f"Error in write_data: {e}")