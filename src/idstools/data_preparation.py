# TODO: Review class
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
            if element["target"] == []:
                X = X.dropna(**element["config"])
                return X
            for feature in element["target"]:
                X = X.dropna(subset=[feature], **element["config"])
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
            for feature in element["target"]:
                X[feature] = imputer.fit_transform(X[[feature]])
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
            for feature in element["target"]:
                X = X.drop([feature], **element["config"])
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
    """This class is used to prepare the data for the training of the model with multiple targets."""
    def __init__(self, targets: list, pipeline: dict = None):
        try:
            logger.info("Initializing DataPreparation for multiple targets.")
            self.targets = targets
            self.pipeline = pipeline if pipeline else {}
            self._pipelines = {}  # Store a pipeline for each target
            
            for target in self.targets:
                self._pipelines[target.env_name] = None
                self.output_path = target.output_path / target.env_name / target.step_name
                self.output_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Pipeline configuration for target {target.env_name}:\n{pprint_dynaconf(self.pipeline)}")
        except Exception as e:
            self.cancel(reason=f"Error in __init__: {e}")

    def build_pipeline(self):
        """This function is used to build the pipeline for each target."""
        for target in self.targets:
            try:
                pipeline_steps = [(transformer, eval(transformer)(config=self.pipeline[transformer])) for transformer in self.pipeline]
                self._pipelines[target.env_name] = Pipeline(steps=pipeline_steps)
                logger.info(f"Pipeline created for target {target.env_name}.")
            except Exception as e:
                logger.error(f"Error in build_pipeline for target {target.env_name}: {e}")

    def run_pipeline(self):
        """This function is used to run the pipeline for each target."""
        for target in self.targets:
            try:
                target_data = target.update_data()
                processed_data = self._pipelines[target.env_name].fit_transform(target_data)
                target.processed_data = processed_data
                logger.info(f"Pipeline processing completed for target {target.env_name}.")
                self.write_data(target)
            except Exception as e:
                logger.error(f"Error in run_pipeline for target {target.env_name}: {e}")

    def write_data(self, target):
        """This function is used to write the processed data to a file."""
        try:
            path = self.output_path / f"{target.filename}_processed.csv"
            write_data(data=target.processed_data, output_path=path)
            logger.info(f"Processed data written to {path} for target {target.env_name}.")
        except Exception as e:
            logger.error(f"Error in write_data for target {target.env_name}: {e}")

    def run(self):
        """This function is used to run the data preparation pipeline."""
        try:
            self.build_pipeline()
            self.run_pipeline()
        except Exception as e:
            self.cancel(reason=f"Error in run: {e}")

    def cancel(self, reason):
        """This function is used to cancel the data preparation."""
        logger.info(f"Cancel of data preparation due to {reason}")
        exit(1)