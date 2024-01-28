import yaml
import importlib
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

import idstools._helpers as helpers

logger = helpers.setup_logging('data_preparation')

class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, config: list):
        self.config = config

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for element in self.config:
            imputer = SimpleImputer(**element["config"])
            X[element["target"]] = imputer.fit_transform(X[[element["target"]]])
        return X
    
class OneHotEncoder(BaseEstimator, TransformerMixin):
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
    
class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, config: list):
        self.config = config

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for element in self.config:
            X = X.drop([element["target"]], **element["config"])
        return X

class GenericDataFrameTransformer(BaseEstimator, TransformerMixin):
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

class data_preparation():
    """This class is used to prepare the data for the training of the model."""
    def __init__(self, config: dict):
        logger.info(f"Start data_preparation with config: \
                    \n{yaml.dump(config, default_flow_style=False)}")
        self.config = config

        if not self.config["input_file"]:
            logger.error(f"Error in run: No input file specified.")
            self.cancel()
        self.data = helpers.read_data(file_config=self.config["input_file"])

        if not self.config["output_path"]:
            logger.info(f"No output path specified. Using default path: {Path(__file__).parent.parent.parent}/results")
            self.output_path = Path(__file__).parent.parent.parent / "results"
        
        self.filename = Path(self.config["input_file"]["path"]).stem

    def build_pipeline(self, config: dict):
        try:
            self.pipeline = Pipeline(steps=[('Imputer', None),
                                            ('OneHotEncoder', None),
                                            ('FeatureDropper', None),
                                            ('GenericDataFrameTransformer', None)])

            if 'Imputer' in config:
                self.pipeline.set_params(Imputer=Imputer(config['Imputer']))
            if 'OneHotEncoder' in config:
                self.pipeline.set_params(OneHotEncoder=OneHotEncoder(config['OneHotEncoder']))
            if 'FeatureDropper' in config:
                self.pipeline.set_params(FeatureDropper=FeatureDropper(config['FeatureDropper']))
            if 'GenericDataFrameTransformer' in config:
                self.pipeline.set_params(GenericDataFrameTransformer=GenericDataFrameTransformer(config['GenericDataFrameTransformer']))
            
            logger.info(f"Pipeline created:\n{str(self.pipeline)}")
            return self.pipeline
        except Exception as e:
            logger.error(f"Error in build_pipeline: {e}")

    def run_pipeline(self, config: dict):
        try:
            self.processed_data = self.data.copy()

            if 'Imputer' in config:
                self.processed_data = self.pipeline.named_steps['Imputer'].transform(self.processed_data)
            if 'OneHotEncoder' in config:    
                self.processed_data = self.pipeline.named_steps['OneHotEncoder'].transform(self.processed_data)
            if 'FeatureDropper' in config:
                self.processed_data = self.pipeline.named_steps['FeatureDropper'].transform(self.processed_data)
            if 'GenericDataFrameTransformer' in config:
                self.processed_data = self.pipeline.named_steps['GenericDataFrameTransformer'].transform(self.processed_data)

            logger.info(f"{self.filename} has been processed by the pipeline.")
            return self.processed_data
        except Exception as e:
            logger.error(f"Error in run_pipeline: {e}")

    def run(self):
        try:
            self.build_pipeline(config=self.config["pipeline"])
            self.run_pipeline(config=self.config["pipeline"])
        except Exception as e:
            logger.error(f"Error in run: {e}")
            self.cancel()

    def cancel(self):
        logger.info(f"Cancel data_preparation")
        exit(1)