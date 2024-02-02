import pandas as pd
from idstools._helpers import setup_logging

logger = setup_logging(__name__)

def replace_dot_with_hyphen(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target in df.columns:
        df[target] = df[target].str.replace('.', '-', regex=False)
    else:
        logger.error(f"Column '{target}' not found in DataFrame.")
    return df