import yaml
import importlib
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

import idstools._helpers as helpers

logger = helpers.setup_logging('data_preparation')

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
    
class _FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, config: list):
        self.config = config

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for element in self.config:
            X = X.drop([element["target"]], **element["config"])
        return X

class _GenericDataFrameTransformer(BaseEstimator, TransformerMixin):
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
            func = self._retrieve_function(element["transform_func"], element.get("module"))
            X = func(X, **element.get("config", {}))
        return X

class DataPreparation():
    """This class is used to prepare the data for the training of the model."""
    def __init__(self, config: dict):
        logger.info(f"Start data_preparation with config: \
                    \n{yaml.dump(config, default_flow_style=False)}")
        self.config = config

        if not self.config["input_file"]:
            self.cancel(cls=__class__, reason="No input file specified.")
        self.data = helpers.read_data(file_path=self.config["input_file"]["path"], file_type=self.config["input_file"]["type"], separator=self.config["input_file"]["separator"])

        if not self.config["output_path"]:
            logger.info(f"No output path specified. Using default path:\
                        {Path(__file__).parent.parent.parent}/results")
            self.output_path = Path(__file__).parent.parent.parent / "results"
        else:
            self.output_path = Path(self.config["output_path"])
        
        self.filename = Path(self.config["input_file"]["path"]).stem

    def build_pipeline(self, config: dict):
        try:
            self.pipeline = Pipeline(steps=[(transformer, None) for transformer in config])
            for transformer in config:
                self.pipeline.set_params(**{transformer: eval(transformer)(config=config[transformer])})
            logger.info(f"Pipeline created:\n{str(self.pipeline)}")
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
            helpers.write_data(data=self.processed_data, output_path=self.output_path, filename=f"{self.filename}_processed.csv")
        except Exception as e:
            logger.error(f"Error in write_data: {e}")

    def run(self):
        try:
            _ = self.build_pipeline(config=self.config["pipeline"])
            _ = self.run_pipeline(config=self.config["pipeline"])
            self.write_data()
        except Exception as e:
            self.cancel(cls=__class__, reason=f"Error in run: {e}")

    def cancel(self, cls, reason):
        logger.info(f"Cancel {cls} of data_explorer due to {reason}")
        exit(1)