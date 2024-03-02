import pandas as pd
from typing import Union
from sklearn.pipeline import Pipeline
from dataclasses import dataclass, field

@dataclass
class TargetData:
    """
    Data class to store the target data
    
    Args:
    name: str = None
        The name of the target data
    data: pd.DataFrame
        The target data
    index: str  = None
        The index column name
    label: str = None
        The label column name
    features: list[str] = None
        The feature column names
    input_path: str = None
        The input path
    input_delimiter: str = None
        The input delimiter
    filename: str = None
        The filename
    output_path: str = None
        The output path
    env_name: str = None
        The environment name
    step_name: str = None
        The step name
    processed_data: pd.DataFrame = field(default_factory=pd.DataFrame)
        The processed data
    analysis_results: dict = None
        The analysis results
    figures: dict = None
        The figures
    """
    id: Union[int, str]
    data: pd.DataFrame
    index: str = None
    label: str = None
    features: list[str] = None
    input_path: str = None
    input_delimiter: str = None
    filename: str = None
    output_path: str = None
    env_name: str = None
    step_name: str = None
    processed_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    analysis_results: dict = None
    figures: dict = None
    pipeline: Pipeline = field(default_factory=Pipeline)