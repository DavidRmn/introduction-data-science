from idstools._helpers import setup_logging
from idstools._helpers import use_decorator, emergency_logger, log_results, resolve_path, read_data

logger = setup_logging(__name__)

@use_decorator(emergency_logger, log_results)
class TargetData():
    """
    This class is used to process target data.
    """
    def __init__(self, input_path: str, input_delimiter: str = None, label: str = None, output_path: str = None, env_name: str = None):
        logger.info("Initializing TargetData object.")

        self.data = None
        self.label = None
        self.filename = None
        self.input_path = None
        self.processed_data = None
        self.output_path = None
        self.env_name = None
        self.analysis_results = {}

        if not input_path:
            logger.error("Please provide an input path.")
            return
        else:
            self.input_path = resolve_path(input_path)
            self.data = read_data(
                file_path=self.input_path,
                separator=input_delimiter
                )
            self.filename = self.input_path.stem

        if not label:
            logger.info(f"No label provided.")
        else:
            self.label = label
            logger.info(f"Using label: {self.label}")

        if not output_path:
            self.output_path = resolve_path("results")
            logger.info(f"Output path not provided.\nUsing default path: {self.output_path}")
        else:
            self.output_path = resolve_path(output_path)
            logger.info(f"Using output path: {self.output_path}")
        
        if not env_name:
            logger.info(f"No environment name provided. Using default name: 'default'")
            self.env_name = "default"
        else:
            self.env_name = env_name
            logger.info(f"Using environment name: {self.env_name}")

    def run(self):
        pass