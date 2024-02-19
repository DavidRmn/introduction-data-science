import pandas as pd
from typing import Union
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

def extract_year_from_datetime(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Extract year from datetime target column.
    
    Args:
        df (pd.DataFrame): DataFrame
        target (str): target column name
    
    Returns:
        pd.DataFrame: DataFrame with year extracted from target column
    """
    logger.info(f"Extracting year from target column '{target}'.")
    if target in df.columns:
        df['year'] = df[target].dt.year
    else:
        logger.error(f"Column '{target}' not found in DataFrame.")
    return df

def group_by_value(df: pd.DataFrame, target: str, value: Union[str, int]) -> pd.DataFrame:
    """
    Group DataFrame by target column value.
    
    Args:
        df (pd.DataFrame): DataFrame
        target (str): target column name
        value (Union[str, int]): target column value
    
    Returns:
        pd.DataFrame: DataFrame grouped by target column value
    """
    logger.info(f"Grouping DataFrame by target column '{target}' value '{value}'.")
    if target in df.columns:
        df = df[df[target] == value]
    else:
        logger.error(f"Column '{target}' not found in DataFrame.")
    return df