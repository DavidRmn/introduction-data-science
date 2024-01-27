import yaml
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import idstools._helpers as helpers

logger = helpers.setup_logging('data_preparation')

class imputer(BaseEstimator, TransformerMixin):
    def fit(self, X, config: dict, y=None):
        return self
    
    def transform(self, X, config: dict):
        for element in config["imputer"]:
            imputer = SimpleImputer(strategy=element["strategy"])
            X[element["variable"]] = imputer.fit_transform(X[[element["variable"]]])
        return X
    
class one_hot_encoder(BaseEstimator, TransformerMixin):
    def fit(self, X, config: dict, y=None):
        return self
    
    def transform(self, X, config: dict):
        for element in config["one_hot_encoder"]:
            ohe_data = pd.get_dummies(X[element["variable"]], prefix=element["prefix"], dtype=element["dtype"])
            X = pd.concat([X, ohe_data], axis=1)
            X = X.drop([element["variable"]], axis=1)
        return X
    
class feature_dropper(BaseEstimator, TransformerMixin):
    def fit(self, X, config: dict, y=None):
        return self
    
    def transform(self, X, config: dict):
        for element in config["feature_dropper"]:
            X = X.drop([element["variable"]], axis=1, errors='ignore')
        return X

class data_preparation():
    """This class is used to prepare the data for the training of the model."""
    def __init__(self, config: dict):
        self.config = config
        self.data = pd.DataFrame()

    def console_output(self):
        logger.info(f"Start data_preparation with config:\n{yaml.dump(self.config, default_flow_style=False)}")
        logger.info(f"Data after preparation:\n{str(self.processed_data[:5].T)}")

    def build_pipeline(self):
        try:
            self.pipeline = Pipeline(steps=[('imputer', None), ('one_hot_encoder', None), ('feature_dropper', None)])
            self.processed_data = self.data.copy()
            if 'imputer' in self.config['pipeline']:
                self.pipeline.set_params(imputer=imputer())
                self.processed_data = self.pipeline.named_steps['imputer'].transform(self.processed_data, self.config["pipeline"])
            if 'one_hot_encoder' in self.config['pipeline']:
                self.pipeline.set_params(one_hot_encoder=one_hot_encoder())
                self.processed_data = self.pipeline.named_steps['one_hot_encoder'].transform(self.processed_data, self.config["pipeline"])
            if 'feature_dropper' in self.config['pipeline']:
                self.pipeline.set_params(feature_dropper=feature_dropper())
                self.processed_data = self.pipeline.named_steps['feature_dropper'].transform(self.processed_data, self.config["pipeline"])

        except Exception as e:
            logger.error(f"Error in build_pipeline: {e}")


    def run(self):
        if self.config["input_file"]:
            self.data = helpers.read_data(file_config=self.config["input_file"])
        else:
            logger.error(f"Error in run: No input file")
            self.cancel()

        if self.config["pipeline"]:
            self.build_pipeline()

        if self.config["console_output"]:
            self.console_output()
    def cancel(self):
        logger.info(f"Cancel data_preparation")
        exit(1)