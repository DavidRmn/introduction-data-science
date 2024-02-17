import pandas as pd
from idstools._data_models import TargetData
from idstools._helpers import setup_logging
from idstools._helpers import use_decorator, emergency_logger, resolve_path, read_data

logger = setup_logging(__name__)

@use_decorator(emergency_logger)
class Target(TargetData):
    """
    This class is used to load and share the target data.

    Args:
        input_path (str): The path to the input data.
        input_delimiter (str): The delimiter of the input data.
        label (str): The label of the target data.
        index (str): The index of the target data.
        features (list[str]): The features of the target data.
        output_path (str): The path to the output data.
        env_name (str): The name of the environment.
        step_name (str): The name of the step.

    Attributes:
        input_path (str): The path to the input data.
        input_delimiter (str): The delimiter of the input data.
        label (str): The label of the target data.
        index (str): The index of the target data.
        features (list[str]): The features of the target data.
        output_path (str): The path to the output data.
        env_name (str): The name of the environment.
        step_name (str): The name of the step.
        data (pd.DataFrame): The target data.
        filename (str): The filename of the input data.
        processed_data (pd.DataFrame): The processed data.
        analysis_results (dict): The analysis results.  
    """
    def __init__(self,
                 input_path: str,
                 input_delimiter: str = None,
                 label: str = None,
                 index: str = None,
                 features: list[str] = None,
                 output_path: str = None,
                 env_name: str = None,
                 step_name: str = None
                ):
        logger.info("Initializing TargetData object.")
        super().__init__(
            data=pd.DataFrame(),
            index=index,
            label=label,
            input_path=input_path,
            input_delimiter=input_delimiter,
            filename=None,
            output_path=output_path,
            env_name=env_name,
            step_name=step_name,
            features=features,
            processed_data=pd.DataFrame(),
            analysis_results=dict()
        )

        if not input_path:
            logger.error("Please provide an input path.")
            return
        else:
            self.input_path = resolve_path(input_path)
            self.data = read_data(
                file_path=self.input_path,
                separator=input_delimiter,
                index=index
                )
            self.filename = self.input_path.stem

        if not label:
            logger.info(f"No label provided.")
        else:
            logger.info(f"Using label: {self.label}")
        
        if not index:
            logger.info(f"No index provided.")
        else:
            logger.info(f"Using index: {self.index}")

        if not features:
            logger.info(f"No features provided.")
            self.features = self.data.columns.tolist()
        else:
            logger.info(f"Using features: {self.features}")

        if not output_path:
            self.output_path = resolve_path("results")
            logger.info(f"Output path not provided.\nUsing default output path: {self.output_path}")
        else:
            self.output_path = resolve_path(output_path)
            logger.info(f"Using output path: {self.output_path}")
        
        if not env_name:
            logger.info(f"No environment name provided.\nUsing default environment name: SELF_EXECUTED")
            self.env_name = "SELF_EXECUTED"
        else:
            logger.info(f"Using environment name: {self.env_name}")
        
        if not step_name:
            logger.info(f"No step name provided.")
            self.step_name = "STEP"
        else:
            logger.info(f"Using step name: {self.step_name}")
    
    def update_data(self) -> pd.DataFrame:
        """
        Update the data attribute with the processed data.
        
        Returns:
            pd.DataFrame: The processed data.
        """
        try:
            if not self.processed_data.empty:
                self.features = list(set(self.features) & set(self.processed_data.columns.tolist()))
                return self.processed_data
        except Exception as e:
            logger.error(f"Error updating data: {e}")
        return self.data