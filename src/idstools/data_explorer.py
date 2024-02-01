import io
import pandas as pd
import seaborn as sns
from pathlib import Path
import missingno as msno
import matplotlib.pyplot as plt
import idstools._helpers as helpers

pd.set_option('display.precision', 2)
logger = helpers.setup_logging(__name__)

@helpers.emergency_logger
class DataExplorer():
    """This class is used to explore the data."""
    def __init__(self, input_path: str, output_path: str, input_type: str = 'csv', input_delimiter: str = ';', pipeline: dict = {}):
        try:
            logger.info("Initializing DataExplorer")
            self.data = helpers.read_data(
                file_path=input_path,
                file_type=input_type,
                separator=input_delimiter,
                )
            self.filename = Path(input_path).stem
            if not output_path:
                self.output_path = Path(__file__).parent.parent.parent / "results"
                logger.info(f"No output path specified.\
                            \nUsing default output path:{self.output_path}")
            else:
                logger.info(f"Using output path: {output_path}")
                self.output_path = Path(output_path).resolve()
                
            if pipeline is None:
                    logger.info("Please provide a pipeline configuration.")
            else:
                logger.info(f"Using pipeline: {pipeline}")
                self.pipeline = pipeline
        except Exception as e:
            self.cancel(cls=__class__, reason=f"Error in __init__: {e}")

    def descriptive_analysis(self):
        try:
            self.head = self.data.head().T
            logger.info(f"Head of {self.filename}\n{str(self.head)}\n")

            buffer = io.StringIO()
            self.data.info(buf=buffer)
            info_str = buffer.getvalue()
            logger.info(f"Info of {self.filename}\n{info_str}")
            buffer.close()
            
            self.types = self.data.dtypes
            logger.info(f"Types of {self.filename}\n{str(self.types)}\n")
            
            self.description = self.data.describe().T
            logger.info(f"Description of {self.filename}\n{str(self.description)}\n")
        except Exception as e:
            logger.error(f"Error in descriptive_analysis: {e}")

    def generate_and_save_plot(self, plot_function):
        try:
            path = self.output_path / Path(self.filename + "_" + str(plot_function).split('.')[1] + ".png")
            plt.figure(figsize=(16, 9))
            plot_function(self.data)
            plt.savefig(path)
            plt.close()
            logger.info(f"Plot saved for {self.filename}\n{path}")
        except Exception as e:
            logger.error(f"Error in generating and saving plot ({self.filename}): {e}")

    def missing_value_matrix_plot(self):
        self.generate_and_save_plot(lambda x: msno.matrix(x),)

    def missing_value_bar_plot(self):
        self.generate_and_save_plot(lambda x: msno.bar(x))

    def correlation_heatmap_plot(self):
        self.generate_and_save_plot(
            lambda x: sns.heatmap(
                x.corr(numeric_only=True),
                annot=True,
                cmap="coolwarm",
                fmt=".2f"
                )
            )

    def run(self):
        for explorer in self.pipeline:
            try:
                logger.info(f"Running {explorer} of data_explorer")
                method = getattr(self, explorer)
                method()
            except AttributeError:
                logger.warning(f"{explorer} is not a valid explorer.")
            except Exception as e:
                self.cancel(cls=__class__, reason=f"Error in run: {e}")

    def cancel(self, cls, reason):
        logger.info(f"Cancel {cls} of data_explorer due to {reason}")
        exit(1)