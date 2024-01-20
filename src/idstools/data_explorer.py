import yaml
import pandas as pd
import idstools._helpers as helpers

pd.set_option('display.precision', 2)
logger = helpers.setup_logging('data_explorer')


class data_explorer():
    def __init__(self, config: tuple):
        self.config = config

    def console_output(self):
        logger.info(f"Start data_explorer with config:\n{yaml.dump(self.config, default_flow_style=False)}")
        logger.info(f"Descriptive Analysis of {self.config['input']['file']['path']}\n{self.description.to_string()}")

    def discriptive_analysis(self):
        if self.config["input"]["file"]:
            file = self.config["input"]["file"]
            if file["type"] == "csv":
                df = pd.read_csv(file["path"], sep=file["sep"])
                description = df.describe().T
                return description

    def run(self):
        if self.config["discriptiv_analysis"]:
            self.description = self.discriptive_analysis()
        if self.config["console_output"]:
            self.console_output()