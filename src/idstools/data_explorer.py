import io
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from tqdm import tqdm
from idstools._config import pprint_dynaconf
from idstools._idstools_data import TargetData
from idstools._helpers import use_decorator, emergency_logger, log_results, setup_logging

pd.set_option('display.precision', 2)
logger = setup_logging(__name__)

@use_decorator(emergency_logger, log_results)
class DataExplorer():
    """This class is used to explore the data."""
    def __init__(self, target_data: object = None ,input_path: str = None, input_delimiter: str = None, output_path: str = None, env_name: str = None, label: str = None, pipeline: dict = None):
        try:
            logger.info("Initializing DataExplorer")

            self.figures = {}
            self.target_data = None

            if target_data is None:
                target_data = TargetData(input_path=input_path, input_delimiter=input_delimiter, label=label, output_path=output_path, env_name=env_name)
            
            self.target_data = target_data
            logger.info(f"Data loaded from {self.target_data.input_path}")

            self.data = self.target_data.data
            self.label = self.target_data.label
            self.filename = self.target_data.filename
            self.output_path = self.target_data.output_path
            self.env_name = self.target_data.env_name

            if not pipeline:
                self.pipeline = {}
                logger.info(f"Please provide a pipeline configuration.")
            else:
                self.pipeline = pipeline
                logger.info(f"Pipeline configuration:\n{pprint_dynaconf(pipeline)}")
            
            self.check_data()

        except Exception as e:
            self.cancel(reason=f"Error in __init__: {e}")

    def check_data(self):
        """Check if data is available."""
        try:
            if self.target_data.processed_data is not None:
                self.data = self.target_data.processed_data
                logger.info(f"Processed data loaded from {self.target_data.input_path}.")
        except Exception as e:
            self.cancel(reason=f"Error in check_data: {e}")

    def descriptive_analysis(self):
        """
        Generates descriptive statistics for the dataset.
        """
        try:
            self.check_data()
            analysis_results = {}
            self.head = self.data.head().T
            analysis_results["HEAD"] = self.head

            buffer = io.StringIO()
            self.data.info(buf=buffer)
            self.info = buffer.getvalue()
            buffer.close()
            analysis_results["INFO"] = self.info
            
            self.dtypes = self.data.dtypes
            analysis_results["DTYPES"] = self.dtypes
            
            self.describe = self.data.describe().T
            analysis_results["DESCRIBE"] = self.describe

            self.isnull = self.data.isnull().sum()
            analysis_results["ISNULL"] = self.isnull
        except Exception as e:
            logger.error(f"Error in descriptive_analysis: {e}")

    def most_correlated_features(self):
        """
        Generates a list of the most correlated features.
        """
        try:
            self.check_data()
            analysis_results = {}
            self.correlation = self.data.select_dtypes(include=['float64', 'int64']).corr()[self.label].abs()
            self.correlation = self.correlation.sort_values(ascending=False)
            self.correlation = self.correlation[(self.correlation < 1) | (self.correlation > -1)]
            self.correlation = self.correlation[(self.correlation >= 0.5) | (self.correlation <= -0.5)]
            analysis_results["CORRELATION"] = self.correlation
        except Exception as e:
            logger.error(f"Error in most_correlated_features: {e}")

    def _generate_plot(self, lambda_func, plotname: str = None, save: bool = True):
        """
        Generates a single plot for the data using seaborn plotting functions.
        
        Args:
            lambda (function): List of lambda functions to generate plots, where each lambda
                            function should expect a DataFrame and an ax as arguments.
            plotname (str): Filename to save the plot.
            save (bool): Whether to save the plot.
        """
        try:
            self.figures[plotname] = plt.figure(figsize=(16, 9))
            lambda_func(self.data)
            if save:
                plt.savefig(self.output_path / f"{self.env_name}_{self.filename}_{plotname}.png")
            plt.close(self.figures[plotname])
        except Exception as e:
            logger.error(f"Error in _generate_plot: {e}")

    def _generate_subplot(self, lambdas, plotname: str = None, save: bool = True):
        """
        Generates plots for the data using seaborn plotting functions.

        Args:
            lambdas (list): List of tuples containing lambda functions to generate plots and plot parameters.
            plotname (str): Filename to save the plot.
            save (bool): Whether to save the plot.
        """
        try:
            num_plots = len(lambdas)
            self.figures[plotname], subplots = plt.subplots(num_plots, 1, figsize=(16, 9 * num_plots))

            if num_plots == 1:
                subplots = [subplots]

            for ax, (lambda_func, kwargs) in zip(subplots, lambdas):
                lambda_func(self.data, ax)
                ax.set_title(kwargs.get("title", ""))
                ax.set_ylabel(kwargs.get("ylabel", ""))
                ax.set_xlabel(kwargs.get("xlabel", ""))
                ax.fontsize = 12

            plt.tight_layout()
            if save:
                plt.savefig(self.output_path / f"{self.env_name}_{self.filename}_{plotname}.png")
            plt.close(self.figures[plotname])
        except Exception as e:
            logger.error(f"Error in _generate_subplot: {e}")

    def missing_value_analysis(self, *args, **kwargs):
        """
        Generates plots for missing value analysis.
        """
        try:
            self.check_data()
            self._generate_plot(
                lambda_func=lambda x: msno.bar(x).set_title("Missing Value Barplot"),
                plotname="missing_value_bar.png"
                )
            self._generate_plot(
                lambda_func=lambda x: msno.matrix(x).set_title("Missing Value Matrix"),
                plotname="missing_value_matrix",
                *args,
                **kwargs
                )
        except Exception as e:
            logger.error(f"Error in missing_value_analysis: {e}")
                    
    def correlation_analysis(self, *args, **kwargs):
        """
        Generates a heatmap of the correlation matrix.
        """
        try:
            self.check_data()
            self._generate_plot(
                lambda_func=lambda x: sns.heatmap(
                    x.corr(numeric_only=True),
                    annot=True, cmap="coolwarm",
                    fmt=".2f"
                    ).set_title("Correlation Heatmap"),
                plotname="correlation_heatmap",
                *args,
                **kwargs
                )
        except Exception as e:
            logger.error(f"Error in correlation_analysis: {e}")
        
    def outlier_analysis(self, *args, **kwargs):
        """
        Generates boxplots for outlier analysis.
        """
        try:
            self.check_data()
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
            self._generate_subplot(lambdas,
                                  plotname="outlier_boxplots",
                                  *args,
                                  **kwargs
                                )
        except Exception as e:
            logger.error(f"Error in outlier_analysis: {e}")

    def distribution_analysis(self, *args, **kwargs):
        """
        Generates distribution plots for the dataset.
        """
        try:
            self.check_data()
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
            self._generate_subplot(lambdas,
                                  plotname="distribution_plots",
                                  *args,
                                  **kwargs
                                )
        except Exception as e:
            logger.error(f"Error in distribution_analysis: {e}")

    def scatter_analysis(self, *args, **kwargs):
        """
        Generates scatter plots for the dataset.
        """
        try:
            self.check_data()
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
            self._generate_subplot(lambdas,
                                  plotname="scatter_plots",
                                  *args,
                                  **kwargs
                                )
        except Exception as e:
            logger.error(f"Error in scatter_analysis: {e}")

    def categorical_analysis(self, *args, **kwargs):
        """
        Generates count plots for categorical columns of the dataset.
        """
        try:
            self.check_data()
            lambdas = []
            for column in tqdm(self.data.select_dtypes(include=['category']).columns, desc="Categorical Plots"):
                lambdas.append(
                    (
                    lambda x, ax, column=column: sns.countplot(x[column], ax=ax),
                        {
                        "title":f"Countplot of {column}",
                        "ylabel":column,
                        "xlabel":"Count"
                        }
                    )
                )
            self._generate_subplot(lambdas,
                                  plotname="categorical_plots",
                                  *args,
                                  **kwargs
                                )
        except Exception as e:
            logger.error(f"Error in categorical_analysis: {e}")

    def time_series_analysis(self, *args, **kwargs):
        """
        Generates time series plots for each column of the dataset.
        """
        try:
            self.check_data()
            lambdas = []
            for time_column in tqdm(self.data.select_dtypes(include=['datetime64']).columns):
                for column in tqdm(self.data.select_dtypes(include=['float64', 'int64']).columns, desc="Time Series Plots"):
                    lambdas.append(
                        (
                        lambda x, ax, column=column, time_column=time_column: sns.lineplot(x=x[time_column], y=x[column], ax=ax),
                            {
                            "title":f"Time Series of {column} vs {time_column}",
                            "ylabel":column,
                            "xlabel":time_column
                            }
                        )
                    )
            self._generate_subplot(lambdas,
                                  plotname="time_series_plots",
                                  *args,
                                  **kwargs
                                )
        except Exception as e:
            logger.error(f"Error in time_series_analysis: {e}")
            
    def over_index_analysis(self, *args, **kwargs):
        """
        Generates plots for over-index analysis.
        """
        try:
            self.check_data()
            lambdas = []
            for column in tqdm(self.data.select_dtypes(include=['float64', 'int64']).columns, desc="Over Index Plots"):
                lambdas.append(
                    (
                    lambda x, ax, column=column: sns.lineplot(x=x.index, y=x[column], ax=ax),
                        {
                        "title":f"Over Index of {column}",
                        "ylabel":column,
                        "xlabel":"Index"
                        }
                    )
                )
            self._generate_subplot(lambdas,
                                  plotname="over_index_plots",
                                  *args,
                                  **kwargs
                                )
        except Exception as e:
            logger.error(f"Error in over_index_analysis: {e}")

    def run(self):
        """
        Runs the pipeline of data exploration methods.
        """
        try:
            for explorer in self.pipeline:
                if explorer:
                    method = getattr(self, explorer)
                    method()
                    logger.info(f"Executed {explorer} of data_explorer.")
        except Exception as e:
            logger.error(f"Error in run: {e}")
        except KeyboardInterrupt:
            self.cancel(reason="KeyboardInterrupt")

    def cancel(self, reason):
        """
        Cancels the data exploration process.
        """
        logger.info(f"Cancel data_explorer due to {reason}")
        exit(1)