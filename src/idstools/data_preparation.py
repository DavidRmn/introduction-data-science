from numpy import var
from sklearn.preprocessing import OneHotEncoder
import yaml
import pandas as pd
import sklearn as sk
import idstools._helpers as helpers

logger = helpers.setup_logging('data_preparation')

class data_preparation():
    """This class is used to prepare the data for the training of the model."""
    def __init__(self, config: dict):
        self.config = config

    def console_output(self):
        logger.info(f"Start data_preparation with config:\n{yaml.dump(self.config, default_flow_style=False)}")
        logger.info(f"Data after one hot encoding:\n{str(self.data.head().T)}")

    def one_hot_encode(self, variable: dict):
        try:
            ohe_data = pd.get_dummies(self.data[variable["target"]], prefix=variable["prefix"], dtype=variable["dtype"])
            self.data = pd.concat([self.data, ohe_data], axis=1)
            self.data = self.data.drop([variable["target"]], axis=1)
        except Exception as e:
            logger.error(f"Error in one_hot_encode: {e}")

    def run(self):
        if self.config["input_file"]:
            self.data = helpers.read_data(file_config=self.config["input_file"])
        else:
            logger.error(f"Error in run: No input file")
            self.cancel()

        if self.config["one_hot_encode"]:
            for variable in self.config["one_hot_encode"]:
                self.one_hot_encode(variable=variable)
        if self.config["console_output"]:
            self.console_output()

    def cancel(self):
        logger.info(f"Cancel data_preparation")
        exit(1)