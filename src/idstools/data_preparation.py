import importlib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from idstools._data_models import TargetData
from idstools._config import pprint_dynaconf
from idstools._helpers import use_decorator, emergency_logger, setup_logging, write_data

logger = setup_logging(__name__)

@use_decorator(emergency_logger)
class _NaNDropper(BaseEstimator, TransformerMixin):
    """This class is used to drop rows with NaN values."""
    def __init__(self, config: list):
        self.config = config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for element in self.config:
            X = X.dropna(subset=[element["target"]])
        return X
@use_decorator(emergency_logger)
class _SimpleImputer(BaseEstimator, TransformerMixin):
    """This class is used to impute NaN values."""
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
    """This class is used to one-hot-encode categorical features."""
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
    """This class is used to drop features."""
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
    """This class is used to apply custom transformations."""
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
    def __init__(self, target: TargetData, pipeline: dict = None):
        try:
            logger.info("Initializing DataPreparation.")
            self.result_logger = setup_logging("data_preparation_results", env_name=target.env_name, step_name=target.step_name, filename="DataPreparation")

            # Initialize class variables
            self._pipeline = None
            self._processed_data = pd.DataFrame()

            # Load data
            self.target = target
            self._data = self.target.update_data()
            logger.info(f"Data loaded from {self.target.input_path}.")
            self.output_path = self.target.output_path / self.target.env_name / self.target.step_name
            self.output_path.mkdir(parents=True, exist_ok=True)

            if not pipeline:
                self.pipeline = {}
                logger.info(f"Please provide a pipeline configuration.")
            else:
                self.pipeline = pipeline
                logger.info(f"Pipeline configuration:\n{pprint_dynaconf(pipeline)}")

        except Exception as e:
            self.cancel(reason=f"Error in __init__: {e}")

    def build_pipeline(self, pipeline: dict):
        try:
            self._pipeline = Pipeline(steps=[(transformer, None) for transformer in pipeline])
            for transformer in pipeline:
                self._pipeline.set_params(**{transformer: eval(transformer)(config=pipeline[transformer])})
            logger.info(f"Pipeline created.")
            self.result_logger.info(f"Pipeline created:\n{pprint_dynaconf(pipeline)}")
        except Exception as e:
            logger.error(f"Error in build_pipeline: {e}")

    def run_pipeline(self, pipeline: dict):
        try:      
            self._processed_data = self._data.copy()
            for transformer in pipeline:
                self._processed_data = self._pipeline.named_steps[transformer].transform(self._processed_data)
                logger.info(f"Pipeline step {transformer} has been processed.")
            self.target.processed_data = self._processed_data
            self.result_logger.info(f"Processed data:\n{self.target.processed_data.head().T}")
        except Exception as e:
            logger.error(f"Error in run_pipeline: {e}")

    def write_data(self):
        try:
            path = self.output_path / f"{self.target.filename}_processed.csv"
            write_data(data=self.target.processed_data, output_path=path)
            logger.info(f"Processed data written to {path}.")
        except Exception as e:
            logger.error(f"Error in write_data: {e}")

    def run(self):
        try:
            self._data = self.target.update_data()
            self.build_pipeline(pipeline=self.pipeline)
            self.run_pipeline(pipeline=self.pipeline)
            self.write_data()
        except Exception as e:
            self.cancel(reason=f"Error in run: {e}")

    def cancel(self, reason):
        logger.info(f"Cancel of data_preparation due to {reason}")
        exit(1)