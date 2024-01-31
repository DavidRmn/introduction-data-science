import logging
import logging.config
import pandas as pd
from pathlib import Path
from idstools._config import _logging

def setup_logging(module_name):
    # Set the logfile path
    logfile_path = Path(__file__).resolve().parent.parent.parent / 'results' / 'idstools.log'
    logfile_path.parent.mkdir(parents=True, exist_ok=True)

    # Update the filename in the file_handler
    _logging.config.handlers.file_handler.filename = str(logfile_path)

    # Apply the logging configuration
    logging.config.dictConfig(_logging.config.items())

    # Get the logger for the module
    logger = logging.getLogger(module_name)
    return logger

logger = setup_logging(__name__)

def read_data(file_path: str, file_type: str, separator: str) -> pd.DataFrame:
    data = pd.DataFrame()
    try:
        if file_type == "csv":
            data = pd.read_csv(
                file_path,
                sep=separator
                )
    except Exception as e:
        logger.error(f"Error in read_data: {e}")

    return data

def write_data(data: pd.DataFrame, output_path: str):
    try:
        data.to_csv(
            Path(output_path),
            index=False
            )
        logger.info(f"Data written to: {output_path}")
    except Exception as e:
        logger.error(f"Error in write_data: {e}")