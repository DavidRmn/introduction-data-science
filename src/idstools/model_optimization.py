from sklearn.model_selection import TimeSeriesSplit, train_test_split
from idstools._idstools_data import TargetData
from idstools._config import pprint_dynaconf
from idstools._helpers import emergency_logger, setup_logging

logger = setup_logging(__name__)

@emergency_logger
class ModelOptimization():
    """This class is used to optimize the model."""
    def __init__(self, target_data: object, pipeline: dict = None):
        try:
            logger.info("Initializing ModelOptimization")
            self.analysis_results = {}

            self.target_data = target_data
            logger.info(f"Data loaded from {target_data.input_path}.")

            self.data = self.target_data.data
            self.label = self.target_data.label
            self.features = self.target_data.features
            self.filename = self.target_data.filename
            self.env_name = self.target_data.env_name
            self.output_path = self.target_data.output_path

            if not pipeline:
                self.pipeline = {}
                logger.info(f"Please provide a pipeline configuration.")
            else:
                self.pipeline = pipeline
                logger.info(f"Pipeline configuration:\n{pprint_dynaconf(pipeline)}")
            
            self.check_data()

        except Exception as e:
            self.cancel(reason=f"Error in __init__: {e}")

    def check_data(self):
        """Check if data is available."""
        try:
            if self.target_data.processed_data:
                self.data = self.target_data.processed_data
                logger.info(f"Processed data loaded from {self.target_data.input_path}.")
        except Exception as e:
            self.cancel(reason=f"Error in check_data: {e}")

    def train_test_split(self):
        """
        This function is used to split the data into training and testing sets.
        """
        try:
            logger.info("Running train_test_split")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[self.features], self.data[self.label], test_size=0.2, random_state=42)
            logger.info("train_test_split completed successfully")
        except Exception as e:
            self.cancel(reason=f"Error in train_test_split: {e}")
        
    def train_test_validation_split(self):
        """
        This function is used to split the data into training, testing, and validation sets.
        """
        try:
            logger.info("Running train_test_validation_split")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[self.features], self.data[self.label], test_size=0.2, random_state=42)
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)
            logger.info("train_test_validation_split completed successfully")
        except Exception as e:
            self.cancel(reason=f"Error in train_test_validation_split: {e}")

    def split_on_yearly_basis(self):
        """
        This function is used to split the data into training and testing sets on a yearly basis.
        """
        try:
            logger.info("Running split_on_yearly_basis")
            self.X_train, self.X_test = self.data[self.data['year'] < 2019][self.features], self.data[self.data['year'] >= 2019][self.features]
            self.y_train, self.y_test = self.data[self.data['year'] < 2019][self.label], self.data[self.data['year'] >= 2019][self.label]
            logger.info("split_on_yearly_basis completed successfully")
        except Exception as e:
            self.cancel(reason=f"Error in split_on_yearly_basis: {e}")
    
    def time_series_split(self):
        """
        This function is used to split the data into training and testing sets.
        """
        try:
            logger.info("Running time_series_split")
            tscv = TimeSeriesSplit(n_splits=5)
            for train_index, test_index in tscv.split(self.data):
                self.X_train, self.X_test = self.data.iloc[train_index][self.features], self.data.iloc[test_index][self.features]
                self.y_train, self.y_test = self.data.iloc[train_index][self.label], self.data.iloc[test_index][self.label]
                logger.info("time_series_split completed successfully")
        except Exception as e:
            self.cancel(reason=f"Error in time_series_split: {e}")

    def run(self):
        """
        This function is used to run the model optimization.
        """
        try:
            self.check_data()
            logger.info("Running ModelOptimization")
            for model in self.pipeline:
                logger.info(f"Running model optimization for {model}")

            logger.info("ModelOptimization completed successfully")
        except Exception as e:
            self.cancel(reason=f"Error in run: {e}")
    
    def cancel(self, reason):
        logger.info(f"Cancel of data_preparation due to {reason}")
        exit(1)