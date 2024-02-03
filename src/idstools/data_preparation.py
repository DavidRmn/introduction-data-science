import importlib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from idstools._config import pprint_dynaconf
from idstools._helpers import emergency_logger, setup_logging, resolve_path, read_data, write_data

logger = setup_logging(__name__)

@emergency_logger
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
    
@emergency_logger
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
    
@emergency_logger    
class _FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, config: list):
        self.config = config

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for element in self.config:
            X = X.drop([element["target"]], **element["config"])
        return X

@emergency_logger
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

@emergency_logger
class DataPreparation():
    """This class is used to prepare the data for the training of the model."""
    def __init__(self, input_path: str, input_delimiter: str = None, output_path: str = None, pipeline: dict = None):
        try:
            logger.info("Initializing DataPreparation")

            if not output_path:
                self.output_path = resolve_path("results")
                logger.info(f"Output path not provided.\nUsing default path: {self.output_path}")
            else:
                self.output_path = resolve_path(output_path)
                logger.info(f"Using output path: {self.output_path}")

            if not pipeline:
                self.transformer = {}
                logger.info(f"Please provide a pipeline configuration.")
            else:
                self.transformer = pipeline
                logger.info(f"Pipeline configuration:\n{pprint_dynaconf(pipeline)}")

            if not input_path:
                logger.error("Please provide an input path.")
                self.data = None
                return
            else:
                self.input_path = resolve_path(input_path)
                self.data = read_data(
                    file_path=self.input_path,
                    separator=input_delimiter
                    )
                self.filename = self.input_path.stem
            
            if self.data is None:
                logger.error(f"Could not read data from {self.input_path}")
                return

        except Exception as e:
            self.cancel(cls=__class__, reason=f"Error in __init__: {e}")

    def build_pipeline(self, config: dict):
        try:
            self.pipeline = Pipeline(steps=[(transformer, None) for transformer in config])
            for transformer in config:
                self.pipeline.set_params(**{transformer: eval(transformer)(config=config[transformer])})
            logger.info(f"Pipeline created.")
            return self.pipeline
        except Exception as e:
            logger.error(f"Error in build_pipeline: {e}")

    def run_pipeline(self, config: dict):
        try:
            self.processed_data = self.data.copy()        
            for transformer in config:
                self.processed_data = self.pipeline.named_steps[transformer].transform(self.processed_data)
                logger.info(f"Pipeline step {transformer} has been processed.")
            return self.processed_data
        except Exception as e:
            logger.error(f"Error in run_pipeline: {e}")

    def write_data(self):
        try:
            path = self.output_path / f"{self.filename}_processed.csv"
            write_data(data=self.processed_data, output_path=path)
        except Exception as e:
            logger.error(f"Error in write_data: {e}")

    def run(self):
        try:
            if self.data is not None:
                _ = self.build_pipeline(config=self.transformer)
                _ = self.run_pipeline(config=self.transformer)
                self.write_data()
        except Exception as e:
            self.cancel(cls=__class__, reason=f"Error in run: {e}")

    def cancel(self, cls, reason):
        logger.info(f"Cancel {cls} of data_preparation due to {reason}")
        exit(1)