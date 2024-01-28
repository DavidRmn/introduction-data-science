import yaml
import pandas as pd
from pathlib import Path
import missingno as msno
import matplotlib.pyplot as plt
import idstools._helpers as helpers

pd.set_option('display.precision', 2)
logger = helpers.setup_logging('data_explorer')


class data_explorer():
    def __init__(self, config: dict):
        self.config = config
        self.filename = Path(self.config["input_file"]["path"]).name.removesuffix(".csv")
        self.output_path = self.config["output_path"]
        self.description = pd.DataFrame

    def discriptive_analysis(self):
        try:
            self.description = self.data.describe().T
            logger.info(f"Descriptive Analysis of {self.filename}\n{str(self.description)}\n")
        except Exception as e:
            logger.error(f"Error in discriptive_analysis: {e}")

    def missing_values(self):
        try:
            plt.figure()
            msno.matrix(self.data)
            plt.savefig(f'{self.output_path}/{self.filename}_matrix_plot.png')
            plt.close()

            plt.figure()
            msno.bar(self.data)
            plt.savefig(f'{self.output_path}/{self.filename}_bar_plot.png')
            plt.close()

            plt.figure()
            msno.heatmap(self.data)
            plt.savefig(f'{self.output_path}/{self.filename}_heatmap_plot.png')
            plt.close()

            logger.info(f"Missing Values plots created for {self.filename}")
            logger.info(f"Plots saved as:\n{self.filename}_matrix_plot.png\n{self.filename}_bar_plot.png\n{self.filename}_heatmap_plot.png")

        except Exception as e:
            logger.error(f"Error in missing_values: {e}")

    def run(self):
        logger.info(f"Start data_explorer with config:\n{yaml.dump(self.config, default_flow_style=False)}")
        if self.config["input_file"]:
            self.data = helpers.read_data(file_config=self.config["input_file"])
        else:
            logger.error(f"Error in run: No input file")
            self.cancel()
        
        if self.config["discriptiv_analysis"]:
            self.discriptive_analysis()
        if self.config["missing_values"]:
            self.missing_values()

    def cancel(self):
        logger.info(f"Cancel data_explorer")
        exit(1)