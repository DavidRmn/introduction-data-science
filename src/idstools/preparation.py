import importlib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
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
            config = element.get("config", {})
            if element["target"] == []:
                X = X.dropna(**config)
                return X
            for feature in element["target"]:
                X = X.dropna(subset=[feature], **config)
            return X
        
@use_decorator(emergency_logger)
class _StandardScaler(BaseEstimator, TransformerMixin):
    """This class is used to scale features."""
    def __init__(self, config: list):
        self.config = config

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for element in self.config:
            config = element.get("config", {})
            scaler = StandardScaler(**config)
            if element["target"] == []:
                logger.info("Scaling all features.")
                X = scaler.fit_transform(X)
                return X
            for feature in element["target"]:
                logger.info(f"Scaling feature {feature}.")
                X[feature] = scaler.fit_transform(X[[feature]])
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
            config = element.get("config", {})
            imputer = SimpleImputer(**config)
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
            config = element.get("config", {})
            ohe_data = pd.get_dummies(X[element["target"]], **config)
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
            config = element.get("config", {})
            for feature in element["target"]:
                X = X.drop([feature], **config)
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
            config = element.get("config", {})
            func = self._retrieve_function(element["func"], element.get("module"))
            X = func(X, **config)
        return X

@use_decorator(emergency_logger)
class DataPreparation():
    """This class is used to prepare the data for the training of the model with multiple targets."""
    def __init__(self, targets: dict, pipeline: dict = None):
        try:
            logger.info("Initializing DataPreparation for multiple targets.")
            self.targets = targets
            self.pipeline = pipeline if pipeline else {}
            self._pipeline = None
            
        except Exception as e:
            self.cancel(reason=f"Error in __init__: {e}")

    def build_pipeline(self):
        """This function is used to build the pipeline."""
        try:
            pipeline_steps = [(transformer, eval(transformer)(config=self.pipeline[transformer])) for transformer in self.pipeline]
            self._pipeline = Pipeline(steps=pipeline_steps)
            logger.info(f"Pipeline created:\n{pprint_dynaconf(self.pipeline)}.")
        except Exception as e:
            logger.error(f"Error in build_pipeline: {e}")

    def run_pipeline(self):
        """This function is used to run the pipeline for each target."""
        for name, target in self.targets.items():
            try:
                target_data = target.update_data()
                processed_data = self._pipeline.fit_transform(target_data)
                target.processed_data = processed_data
                logger.info(f"Pipeline processing completed for target {name} in {target.env_name}:{target.step_name}.")
                self.write_data(target)
            except Exception as e:
                logger.error(f"Error in run_pipeline for target {name} in {target.env_name}:{target.step_name}: {e}")

    def write_data(self, target):
        """This function is used to write the processed data to a file."""
        try:
            path = target.output_path / target.env_name / target.step_name / f"{target.filename}_processed.csv"
            write_data(data=target.processed_data, output_path=path)
            logger.info(f"Processed data written to {path} for target {target.name} in {target.env_name}:{target.step_name}.")
        except Exception as e:
            logger.error(f"Error in write_data for target {target.name} in {target.env_name}:{target.step_name}: {e}")

    def run(self):
        """This function is used to run the data preparation pipeline."""
        try:
            if not self.pipeline:
                self.cancel(reason="No pipeline defined.")
            self.build_pipeline()
            self.run_pipeline()
        except (Exception, KeyboardInterrupt) as e:
            self.cancel(reason=f"Run canceled: {e}")

    def cancel(self, reason):
        """This function is used to cancel the data preparation."""
        logger.info(f"Cancel of data preparation due to <{reason}>.")
        exit(1)