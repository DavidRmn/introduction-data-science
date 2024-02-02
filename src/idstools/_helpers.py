import functools
import logging
import logging.config
import pandas as pd
from pathlib import Path
from idstools._config import _logging

def setup_logging(module_name):
    """
    This function sets up the logging configuration for the module.
    
    Args:
        module_name (str): The name of the module to set up logging for.
    Returns:
        logger (logging.Logger): The logger object for the module.
    """
    logfile_path = Path(__file__).resolve().parent.parent.parent / 'results' / 'idstools.log'
    logfile_path.parent.mkdir(parents=True, exist_ok=True)

    _logging.default.handlers.file_handler.filename = str(logfile_path)

    logging.config.dictConfig(_logging.default.to_dict())

    logger = logging.getLogger(module_name)
    return logger

logger = setup_logging(__name__)

def emergency_logger(func):
    """
    A decorator that logs exceptions at the emergency level.

    Args:
        func (function): The function to decorate.
    Returns:
        wrapper (function): The decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Emergency in '{func.__name__}': {e}", exc_info=True)
            raise
    return wrapper

@emergency_logger
def read_data(file_path: Path, file_type: str | None, separator: str | None) -> pd.DataFrame | None:
    """
    This function reads data from a file and returns a DataFrame.
    
    Args:
        file_path (Path): The path to the file to read.
        file_type (str): The type of file to read.
        separator (str): The separator for the file.
        data (pd.DataFrame): The data from the file.
    """
    try:
        if file_type in ['csv']:
            logger.info(f"Reading {file_type} file:\n{file_path}")
            data = pd.read_csv(
                file_path,
                sep=separator
                )
            return data
    except Exception as e:
        logger.error(f"Error in read_data: {e}")

@emergency_logger
def write_data(data: pd.DataFrame, output_path: Path):
    """
    This function writes data to a file.
    
    Args:
        data (pd.DataFrame): The data to write to the file.
        output_path (Path): The path to the file to write the data to.
    """
    try:
        logger.info(f"Writing data to:\n{output_path}")
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
            )
        data.to_csv(
            output_path,
            index=False
            )
    except Exception as e:
        logger.error(f"Error in write_data: {e}")