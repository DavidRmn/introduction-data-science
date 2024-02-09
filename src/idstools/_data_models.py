import pandas as pd
from dataclasses import dataclass, field

@dataclass
class TargetData:
    data: pd.DataFrame
    index: str = None
    label: str = None
    features: list[str] = None
    input_path: str = None
    input_delimiter: str = None
    filename: str = None
    output_path: str = None
    env_name: str = None
    processed_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    analysis_results: dict = None
