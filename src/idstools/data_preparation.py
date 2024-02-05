import importlib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from idstools._config import pprint_dynaconf
from idstools._idstools_data import TargetData
from idstools._helpers import use_decorator, emergency_logger, setup_logging, write_data, result_logger

logger = setup_logging(__name__)

@use_decorator(emergency_logger)
class _SimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, config: list):
        self.config = config

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for element in self.config:
            imputer = SimpleImputer(**element["config"])
            X[element["target"]] = imputer.fit_transform(X[[element["target"]]])
        return X
    
@use_decorator(emergency_logger)
class _OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, config: list):
        self.config = config

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for element in self.config:
            ohe_data = pd.get_dummies(X[element["target"]], **element["config"])
            X = pd.concat([X, ohe_data], axis=1)
            X = X.drop([element["target"]], axis=1)
        return X
    
@use_decorator(emergency_logger)   
class _FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, config: list):
        self.config = config

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for element in self.config:
            X = X.drop([element["target"]], **element["config"])
        return X

@use_decorator(emergency_logger)
class _CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, config: list):
        self.config = config

    def fit(self, X, y=None):
        return self

    def _retrieve_function(self, func_name, module=None):
        if callable(func_name):
            return func_name
        try:
            if module:
                module = importlib.import_module(module)
                return getattr(module, func_name)
            else:
                return globals()[func_name]
        except Exception as e:
            raise ImportError(f"Could not import function: {func_name} from module: {module}. Error: {e}")

    def transform(self, X):
        for element in self.config:
            func = self._retrieve_function(element["func"], element.get("module"))
            X = func(X, **element.get("config", {}))
        return X

@use_decorator(emergency_logger)
class DataPreparation():
    """This class is used to prepare the data for the training of the model."""
    def __init__(self, target_data: object = None ,input_path: str = None, input_delimiter: str = None, env_name: str = None, pipeline: dict = None, output_path: str = None):
        try:
            logger.info("Initializing DataPreparation")

            self.analysis_results = {}
            self.processed_data = None

            if target_data is None:
                self.target_data = TargetData(input_path=input_path, input_delimiter=input_delimiter, output_path=output_path, env_name=env_name)
                logger.info(f"Data loaded from {self.target_data.input_path}.")
            else:
                self.target_data = target_data
                logger.info(f"Data loaded from {self.target_data.input_path}.")

            self.data = self.target_data.data
            self.label = self.target_data.label
            self.filename = self.target_data.filename
            self.output_path = self.target_data.output_path
            self.env_name = self.target_data.env_name

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

    def build_pipeline(self, pipeline: dict):
        try:
            self._pipeline = Pipeline(steps=[(transformer, None) for transformer in pipeline])
            for transformer in pipeline:
                self._pipeline.set_params(**{transformer: eval(transformer)(config=pipeline[transformer])})
            logger.info(f"Pipeline created.")
            result_logger.info(f"ENV:{self.env_name} Pipeline created:\n{pprint_dynaconf(pipeline)}")
        except Exception as e:
            logger.error(f"Error in build_pipeline: {e}")

    def run_pipeline(self, pipeline: dict):
        try:
            self.processed_data = self.data.copy()        
            for transformer in pipeline:
                self.processed_data = self._pipeline.named_steps[transformer].transform(self.processed_data)
                logger.info(f"Pipeline step {transformer} has been processed.")
            self.target_data.processed_data = self.processed_data
            result_logger.info(f"ENV:{self.env_name} Processed data:\n{self.processed_data.head().T}")
        except Exception as e:
            logger.error(f"Error in run_pipeline: {e}")

    def write_data(self):
        try:
            path = self.output_path / f"{self.env_name}_{self.filename}_processed.csv"
            write_data(data=self.processed_data, output_path=path)
            logger.info(f"Processed data written to {path}.")
        except Exception as e:
            logger.error(f"Error in write_data: {e}")

    def run(self):
        try:
            self.check_data()
            self.build_pipeline(pipeline=self.pipeline)
            self.run_pipeline(pipeline=self.pipeline)
            self.write_data()
        except Exception as e:
            self.cancel(reason=f"Error in run: {e}")

    def cancel(self, reason):
        logger.info(f"Cancel of data_preparation due to {reason}")
        exit(1)