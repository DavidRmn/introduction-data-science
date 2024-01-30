import logging
import logging.config
import pandas as pd
from pathlib import Path
from idstools._config import settings

def setup_logging(module_name):
    """Setup logging with the provided module name"""
    logfile_path = Path(__file__).parent.parent.parent / 'results' / 'idstools.log'
    # Ensure the /results directory exists
    logfile_path.parent.mkdir(parents=True, exist_ok=True)

    # Update the filename in the file_handler
    if 'handlers' in settings.logging and 'file_handler' in settings.logging.handlers:
        settings.set('logging.handlers.file_handler.filename', str(logfile_path))

    logger = logging.getLogger('default')

    if module_name in settings.logging.loggers:
        logging.config.dictConfig(settings.logging)
        logger = logging.getLogger(module_name)
    else:
        logging.basicConfig(level=logging.WARNING)

    return logger

logger = setup_logging(__name__)

def read_data(file_path: str, file_type: str, separator: str) -> pd.DataFrame:
    data = pd.DataFrame()
    try:
        if file_type == "csv":
            data = pd.read_csv(file_path, sep=separator)
    except Exception as e:
        logger.error(f"Error in read_data: {e}")

    return data

def write_data(data: pd.DataFrame, output_path: Path, filename: str):
    try:
        data.to_csv(output_path / filename, index=False)
        logger.info(f"Data written to: {output_path / filename}")
    except Exception as e:
        logger.error(f"Error in write_data: {e}")