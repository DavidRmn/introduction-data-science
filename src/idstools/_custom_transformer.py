import pandas as pd
from idstools._helpers import setup_logging

logger = setup_logging(__name__)

def replace_dot_with_hyphen(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Replace dot with hyphen in target column.
    
    Args:
        df (pd.DataFrame): DataFrame
        target (str): target column name
        
    Returns:
        pd.DataFrame: DataFrame with target column replaced
    """
    logger.info(f"Replacing dot with hyphen in target column '{target}'.")
    if target in df.columns:
        df[target] = df[target].str.replace('.', '-', regex=False)
    else:
        logger.error(f"Column '{target}' not found in DataFrame.")
    return df

def target_to_datetime(df: pd.DataFrame, target: str, format: str = "%m.%d.%Y") -> pd.DataFrame:
    """
    Convert target column to datetime format.
    
    Args:
        df (pd.DataFrame): DataFrame
        target (str): target column name
    
    Returns:
        pd.DataFrame: DataFrame with target column converted to datetime
    """
    logger.info(f"Converting target column '{target}' to datetime.")
    if target in df.columns:
        try:
            df[target] = pd.to_datetime(df[target], format=format)
        except Exception as e:
            logger.error(f"Error converting column '{target}' to datetime: {e}")
    else:
        logger.error(f"Column '{target}' not found in DataFrame.")
    
    return df