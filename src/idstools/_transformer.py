import pandas as pd
import numpy as np
from typing import Union
from idstools._helpers import setup_logging

logger = setup_logging(__name__)

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all columns are numeric.
    
    Args:
        df (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame: DataFrame with all columns converted to numeric
    """
    logger.info("Ensuring all columns are numeric.")
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception as e:
                logger.error(f"Error converting column '{col}' to numeric: {e}")
    return df

def remove_outliers(df: pd.DataFrame, target: str, threshold: float = 3) -> pd.DataFrame:
    """
    Drop outliers from target column.
    
    Args:
        df (pd.DataFrame): DataFrame
        target (str): target column name
        threshold (float): threshold value
    
    Returns:
        pd.DataFrame: DataFrame with outliers dropped
    """
    logger.info(f"Dropping outliers from target column '{target}'.")
    if target in df.columns:
        z_scores = (df[target] - df[target].mean()) / df[target].std()
        df = df[(z_scores < threshold) & (z_scores > -threshold)]
    else:
        logger.error(f"Column '{target}' not found in DataFrame.")
    return df

def negative_to_nan(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Replace negative values with NaN in target column.
    
    Args:
        df (pd.DataFrame): DataFrame
        target (str): target column name
        
    Returns:
        pd.DataFrame: DataFrame with negative values replaced with NaN
    """
    logger.info(f"Replacing negative values with NaN in target column '{target}'.")
    if target in df.columns:
        df[target] = df[target].apply(lambda x: x if x > 0.0 else np.nan)
    else:
        logger.error(f"Column '{target}' not found in DataFrame.")
    return df

def process_weekday(df: pd.DataFrame, target: str, date: str) -> pd.DataFrame:
    """
    Maps weekday based on date column values.
    
    Args:
        df (pd.DataFrame): DataFrame
        date (str): date column name (in datetime64 format)
    
    Returns:
        pd.DataFrame: DataFrame with 'weekday' column mapped based on date column values.
    """
    logger.info(f"Mapping weekday based on '{date}' column value.")
    if date in df.columns:
        df[target] = (df[date].dt.dayofweek + 1) % 7
    else:
        logger.error(f"Date column: {date} doesn't exist")
    return df

def impute_season(df: pd.DataFrame, target: str, date: str) -> pd.DataFrame:
    """
    Impute season based on target column value.
    
    Args:
        df (pd.DataFrame): DataFrame
        target (str): target column name
    
    Returns:
        pd.DataFrame: DataFrame with season imputed based on target column value
    """
    def get_season(date):
        day_of_year = date.timetuple().tm_yday
        if 80 <= day_of_year < 172:  #spring, day ranges from 21st March to 20th June
            return 2
        elif 172 <= day_of_year < 265:  #summer, day ranges from 21st June to 21st September
            return 3
        elif 265 <= day_of_year < 355:  #fall, day ranges from 22nd September to 20th December
            return 4
        else:                           #winter
            return 1
    
    logger.info(f"Imputing {target} based on target column '{date}' value.")
    if date in df.columns:
        df[target] = df[date].apply(get_season)
    else:
        logger.error(f"Date column: {date} doesn't exist")
        
    return df

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