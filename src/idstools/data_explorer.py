import yaml
import pandas as pd
import seaborn as sns
from pathlib import Path
import missingno as msno
import matplotlib.pyplot as plt
import idstools._helpers as helpers
from idstools._config import idstools_config

pd.set_option('display.precision', 2)
logger = helpers.setup_logging(__name__)

class DataExplorer():
    """This class is used to explore the data."""
    def __init__(self, config: dict = {}):
        if config:
            logger.info(
                f"Config was provided setting config:\n{yaml.dump(
                    config, default_flow_style=False
                    )}")
            idstools_config.set(
                idstools_config.modules.data_explorer,
                config
                )
        logger.info(
            f"Start data_explorer with config:\n{yaml.dump(
                idstools_config.modules.data_explorer,
                default_flow_style=False
                )}")

        if not idstools_config.modules.data_explorer.input_file:
            self.cancel(
                cls=__class__,
                reason="No input file specified."
                )
        self.data = helpers.read_data(
            file_path=idstools_config.modules.data_explorer.input_file.path,
            file_type=idstools_config.modules.data_explorer.input_file.type,
            separator=idstools_config.modules.data_explorer.input_file.separator
            )

        if not idstools_config.modules.data_explorer.output_path:
            logger.info(
                f"No output path specified. Using default path: \
                {Path(__file__).parent.parent.parent}/results"
                )
            idstools_config.set(
                idstools_config.modules.data_explorer.output_path,
                Path(__file__).parent.parent.parent / "results"
            )
        else:
            idstools_config.set(
                idstools_config.modules.data_explorer.output_path,
                Path(idstools_config.modules.data_explorer.output_path)
            )

        self.description = pd.DataFrame
        self.filename = Path(
            idstools_config.modules.data_explorer.input_file.file_path
            ).stem
        self.output_path = idstools_config.modules.data_explorer.output_path
        self.pipeline = idstools_config.modules.data_explorer.pipeline

    def descriptive_analysis(self):
        try:
            self.description = self.data.describe().T
            logger.info(f"Descriptive Analysis of {self.filename}\n{str(self.description)}\n")
        except Exception as e:
            logger.error(f"Error in discriptive_analysis: {e}")

    def missing_value_analysis(self):
        try:
            plt.figure()
            msno.matrix(self.data)
            plt.savefig(f'{self.output_path}/{self.filename}_matrix_plot.png')
            plt.close()

            plt.figure()
            msno.bar(self.data)
            plt.savefig(f'{self.output_path}/{self.filename}_bar_plot.png')
            plt.close()

            logger.info(f"Missing Values plots created for {self.filename}")
            logger.info(f"Plots saved as:\n{self.filename}_matrix_plot.png\n{self.filename}_bar_plot.png\n{self.filename}_heatmap_plot.png")

        except Exception as e:
            logger.error(f"Error in missing_values: {e}")

    def correlation_analysis(self):
        try:
            plt.figure()
            sns.heatmap(self.data.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
            plt.savefig(f'{self.output_path}/{self.filename}_correlation_plot.png')
            plt.close()
        except Exception as e:
            logger.error(f"Error in correlation: {e}")

    def run(self):
        for explorer in self.pipeline:
            try:
                method = getattr(self, explorer)
                method()
            except AttributeError:
                logger.warning(f"{explorer} is not a valid explorer.")
            except Exception as e:
                self.cancel(cls=__class__, reason=f"Error in run: {e}")

    def cancel(self, cls, reason):
        logger.info(f"Cancel {cls} of data_explorer due to {reason}")
        exit(1)