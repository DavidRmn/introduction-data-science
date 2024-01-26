import yaml
import pandas as pd
import idstools._helpers as helpers

pd.set_option('display.precision', 2)
logger = helpers.setup_logging('data_explorer')


class data_explorer():
    def __init__(self, config: dict):
        self.config = config
        self.description = pd.DataFrame

    def console_output(self):
        logger.info(f"Start data_explorer with config:\n{yaml.dump(self.config, default_flow_style=False)}")
        logger.info(f"Descriptive Analysis of {self.config.get('input', {}).get('file').get('path')}\n{str(self.description)}")

    def discriptive_analysis(self):
        try:
            file = self.config.get("input", {}).get("file")
            if file.get("type") == "csv":
                df = pd.read_csv(file.get("path"), sep=file.get("sep"))
                self.description = df.describe().T
        except Exception as e:
            logger.error(f"Error in discriptive_analysis: {e}")

    def run(self):
        if self.config.get("discriptiv_analysis"):
            self.discriptive_analysis()
        if self.config.get("console_output"):
            self.console_output()