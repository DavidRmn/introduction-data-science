import functools
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
    _logging.default.handlers.file_handler.filename = str(logfile_path)

    # Apply the logging configuration
    logging.config.dictConfig(_logging.default.to_dict())

    # Get the logger for the module
    logger = logging.getLogger(module_name)
    return logger

logger = setup_logging(__name__)

def emergency_logger(func):
    """
    A decorator that logs exceptions at the emergency level.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log the exception with a message indicating it's an emergency
            logger.error(f"Emergency in '{func.__name__}': {e}", exc_info=True)
            # Optionally, you can add additional emergency handling logic here
            # Reraise the exception
            raise
    return wrapper

@emergency_logger
def read_data(file_path: str, file_type: str, separator: str | None) -> pd.DataFrame:
    data = pd.DataFrame()
    try:
        if file_type in ['csv']:
            logger.info(f"Reading {file_type} file:\n{file_path}")
            data = pd.read_csv(
                Path(file_path).resolve(),
                sep=separator
                )
    except Exception as e:
        logger.error(f"Error in read_data: {e}")
    return data

@emergency_logger
def write_data(data: pd.DataFrame, output_path: str):
    try:
        logger.info(f"Writing data to:\n{output_path}")
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(
            path,
            index=False
            )
    except Exception as e:
        logger.error(f"Error in write_data: {e}")