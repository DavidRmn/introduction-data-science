import functools
import logging
import logging.config
import pandas as pd
from pathlib import Path
from idstools._config import _logging

def setup_logging(module_name, env_name: str = None, step_name: str = None, filename: str = None) -> logging.Logger:
    """
    This function sets up the logging configuration for the module.
    
    Args:
        module_name (str): The name of the module to set up logging for.
    Returns:
        logger (logging.Logger): The logger object for the module.
    """
    logfile_path = Path(__file__).resolve().parent.parent.parent / 'results' / "idstools.log"
    _logging.default.handlers.file_handler.filename = str(logfile_path)
    logfile_path.parent.mkdir(parents=True, exist_ok=True)

    if env_name and step_name and filename:
        resultfile_path = Path(__file__).resolve().parent.parent.parent / 'results' / env_name / step_name / f"{filename}_results.log"
        _logging.default.handlers.resultfile_handler.filename = str(resultfile_path)
        resultfile_path.parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(_logging.default.to_dict())

    logger = logging.getLogger(module_name)
    return logger

logger = setup_logging(__name__)

def use_decorator(*decorators):
    """
    A decorator that applies multiple decorators to a class.
    
    Args:
        *decorators: The decorators to apply to the class.
    Returns:
        decorator (function): The decorator function.
    """
    def decorator(cls):
        for decorator_func in decorators:
            for name, attr in vars(cls).items():
                if callable(attr):
                    setattr(cls, name, decorator_func(attr))
        return cls
    return decorator

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
def resolve_path(path: str | Path) -> Path:
    """
    This function resolves a path to a Path object.
    
    Args:
        path (str | Path): The path to resolve.
    Returns:
        resolved_path (Path): The resolved path.
    """
    try:
        path = Path(path)
        if path.is_absolute():
            return path
        return Path(__file__).parent.parent.parent / path
    except Exception as e:
        logger.error(f"Error in resolve_path: {e}")
        return None

@emergency_logger
def read_data(file_path: Path, separator: str | None, index: str | None) -> pd.DataFrame | None:
    """
    This function reads data from a file and returns a DataFrame.
    
    Args:
        file_path (Path): The path to the file to read.
        separator (str | None): The separator for the file. If None, the default separator for the file type will be used.
    """
    try:
        extension_methods = {
            '.csv': 'read_csv',
            '.json': 'read_json',
            '.parquet': 'read_parquet',
            '.pickle': 'read_pickle',
            '.xls': 'read_excel',
            '.xlsx': 'read_excel'
        }
        file_type = file_path.suffix.lower()
        method = extension_methods.get(file_type, None)
        if method:
            logger.info(f"Reading data from:\n{file_path.resolve()}")
            if separator:
                data = getattr(pd, method)(file_path, sep=separator, index_col=index)
            else:
                data = getattr(pd, method)(file_path, index_col=index)
            return data
    except AttributeError as e:
        logger.error(f"File type '{file_type}' is not supported: {e}")
        return None
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None
    except Exception as e:
        logger.error(f"Error in read_data: {e}")
        return None
    
@emergency_logger
def write_data(data: pd.DataFrame, output_path: Path):
    """
    This function writes data to a file.
    
    Args:
        data (pd.DataFrame): The data to write to the file.
        output_path (Path): The path to the file to write the data to.
    """
    logger.info(f"Writing data to:\n{output_path}")
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
        )
    data.to_csv(
        output_path,
        index=False
        )