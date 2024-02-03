import io
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from tqdm import tqdm
from idstools._config import pprint_dynaconf
from idstools._helpers import emergency_logger, log_results, setup_logging, resolve_path, read_data

pd.set_option('display.precision', 2)
logger = setup_logging(__name__)

@emergency_logger
class DataExplorer():
    """This class is used to explore the data."""
    def __init__(self, input_path: str, input_delimiter: str = None, output_path: str = None, label: str = None, pipeline: dict = None):
        try:
            logger.info("Initializing DataExplorer")

            if not label:
                logger.info(f"No label provided.")
            else:
                self.label = label
                logger.info(f"Using label: {self.label}")

            if not output_path:
                self.output_path = resolve_path("results")
                logger.info(f"Output path not provided.\nUsing default path: {self.output_path}")
            else:
                self.output_path = resolve_path(output_path)
                logger.info(f"Using output path: {self.output_path}")

            if not pipeline:
                self.pipeline = {}
                logger.info(f"Please provide a pipeline configuration.")
            else:
                self.pipeline = pipeline
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

    @log_results
    def descriptive_analysis(self):
        """
        Generates descriptive statistics for the dataset.
        """
        try:
            self.head = self.data.head().T
            self.analysis_results["HEAD"] = self.head

            buffer = io.StringIO()
            self.data.info(buf=buffer)
            self.info = buffer.getvalue()
            buffer.close()
            self.analysis_results["INFO"] = self.info
            
            self.dtypes = self.data.dtypes
            self.analysis_results["DTYPES"] = self.dtypes
            
            self.describe = self.data.describe().T
            self.analysis_results["DESCRIBE"] = self.describe

            self.isnull = self.data.isnull().sum()
            self.analysis_results["ISNULL"] = self.isnull
        except Exception as e:
            logger.error(f"Error in descriptive_analysis: {e}")

    def generate_plot(self, lambda_func, plotname: str = None):
        """
        Generates a single plot for the data using seaborn plotting functions.
        
        Args:
            lambda (function): List of lambda functions to generate plots, where each lambda
                            function should expect a DataFrame and an ax as arguments.
            filename (str): Filename to save the combined plot.
        """
        try:
            fig = plt.figure(figsize=(16, 9))
            lambda_func(self.data)
            plt.savefig(self.output_path / f"{self.filename}_{plotname}")
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error in generate_plot: {e}")

    def generate_subplot(self, lambdas, plotname: str = None):
        """
        Generates plots for the data using seaborn plotting functions.

        Args:
            lambdas (list): List of tuples containing lambda functions to generate plots and plot parameters.
            filename (str): Filename to save the combined plot.
        """
        try:
            num_plots = len(lambdas)
            fig, subplots = plt.subplots(num_plots, 1, figsize=(16, 9 * num_plots))

            if num_plots == 1:
                subplots = [subplots]

            for ax, (lambda_func, kwargs) in zip(subplots, lambdas):
                lambda_func(self.data, ax)
                ax.set_title(kwargs.get("title", ""))
                ax.set_ylabel(kwargs.get("ylabel", ""))
                ax.set_xlabel(kwargs.get("xlabel", ""))
                ax.fontsize = 12

            plt.tight_layout()
            plt.savefig(self.output_path / f"{self.filename}_{plotname}")
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error in generate_subplot: {e}")

    def missing_value_analysis(self):
        """
        Generates plots for missing value analysis.
        """
        try:
            self.generate_plot(
                lambda_func=lambda x: msno.bar(x).set_title("Missing Value Barplot"),
                plotname="missing_value_bar.png"
                )
            self.generate_plot(
                lambda_func=lambda x: msno.matrix(x).set_title("Missing Value Matrix"),
                plotname="missing_value_matrix.png"
                )
        except Exception as e:
            logger.error(f"Error in missing_value_analysis: {e}")
                    
    def correlation_analysis(self):
        """
        Generates a heatmap of the correlation matrix.
        """
        try:
            self.generate_plot(
                lambda_func=lambda x: sns.heatmap(
                    x.corr(numeric_only=True),
                    annot=True, cmap="coolwarm",
                    fmt=".2f"
                    ).set_title("Correlation Heatmap"),
                plotname="correlation_heatmap.png"
                )
        except Exception as e:
            logger.error(f"Error in correlation_analysis: {e}")
        
    def outlier_analysis(self):
        """
        Generates boxplots for outlier analysis.
        """
        try:
            lambdas = []
            for column in tqdm(self.data.select_dtypes(include=['float64', 'int64']).columns, desc="Outlier Barplots"):
                lambdas.append(
                    (
                        lambda x, ax, column=column: sns.boxplot(x=x[column], ax=ax),
                            {
                            "title": f"Boxplot of {column}",
                            "ylabel": column,
                            "xlabel": "Distribution"
                            }
                    )
                )
            self.generate_subplot(lambdas, plotname="outlier_boxplots.png")
        except Exception as e:
            logger.error(f"Error in outlier_analysis: {e}")

    def distribution_analysis(self):
        """
        Generates distribution plots for the dataset.
        """
        try:
            lambdas = []
            for column in tqdm(self.data.select_dtypes(include=['float64', 'int64']).columns, desc="Distribution Plots"):
                lambdas.append(
                    (
                    lambda x, ax, column=column: sns.histplot(x[column], kde=True, ax=ax),
                        {
                        "title":f"Skew: {round(self.data[column].skew(), 2)}",
                        "ylabel":column,
                        "xlabel":"Distribution"
                        }
                    )
                )
            self.generate_subplot(lambdas, plotname="distribution_plots.png")
        except Exception as e:
            logger.error(f"Error in distribution_analysis: {e}")

    def scatter_analysis(self):
        """
        Generates scatter plots for the dataset.
        """
        try:
            lambdas = []
            for column in tqdm(self.data.select_dtypes(include=['float64', 'int64']).columns, desc="Scatter Plots"):
                lambdas.append(
                    (
                    lambda x, ax, column=column: sns.scatterplot(x=x[column], y=x[self.label], ax=ax),
                        {
                        "title":f"Scatterplot of {column} vs {self.label}",
                        "ylabel":self.label,
                        "xlabel":column
                        }
                    )
                )
            self.generate_subplot(lambdas, plotname="scatter_plots.png")
        except Exception as e:
            logger.error(f"Error in scatter_analysis: {e}")

    def run(self):
        """
        Runs the pipeline of data exploration methods.
        """
        try:
            if self.data is not None:
                self.analysis_results = {}
                for explorer in self.pipeline:
                    if explorer:
                        method = getattr(self, explorer)
                        method()
                        logger.info(f"Executed {explorer} of data_explorer.")
        except Exception as e:
            logger.error(f"Error in run: {e}")
        except KeyboardInterrupt:
            self.cancel(cls=__class__, reason="KeyboardInterrupt")

    def cancel(self, cls, reason):
        """
        Cancels the data exploration process.
        """
        logger.info(f"Cancel {cls} of data_explorer due to {reason}")
        exit(1)