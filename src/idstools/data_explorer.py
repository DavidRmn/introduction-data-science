import io
import math
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from tqdm import tqdm
from idstools._data_models import TargetData
from idstools._config import pprint_dynaconf
from idstools._helpers import use_decorator, emergency_logger, setup_logging, result_logger

pd.set_option('display.precision', 2)
logger = setup_logging(__name__)

@use_decorator(emergency_logger)
class DataExplorer():
    """
    This class is used to explore the data.
    
    Available methods:
    - descriptive_analysis: Generates descriptive statistics for the dataset.
    - calculate_correlation: Calculates the correlation matrix for the dataset.
    - missing_value_analysis: Generates plots for missing value analysis.
    - correlation_analysis: Generates a heatmap of the correlation matrix for the dataset.
    - outlier_analysis: Generates boxplots for outlier analysis.
    - distribution_analysis: Generates distribution plots for the dataset.
    - scatter_analysis: Generates scatter plots for the dataset.
    - categorical_analysis: Generates count plots for categorical columns of the dataset.
    - time_series_analysis: Generates time series plots for each column of the dataset.
    - over_index_analysis: Generates plots for over-index analysis.
    - run: Runs the pipeline of data exploration methods.
    - cancel: Cancels the data exploration process.
    """
    def __init__(self, target: TargetData, pipeline: dict = None):
        try:
            logger.info("Initializing DataExplorer")

            # Initialize class variables
            self._data = pd.DataFrame()
            self.figures = {}
            self.head = None
            self.info = None
            self.dtypes = None
            self.describe = None
            self.isnull = None
            self.correlation = None
            self.weak_correlation = None
            self.moderate_correlation = None
            self.strong_correlation = None

            # Load data
            self.target = target
            self._data = self.target.data.copy()
            logger.info(f"Data loaded from {self.target.input_path}.")

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
            if not self.target.processed_data.empty:
                self._data = self.target.processed_data.copy()
                logger.info(f"Processed data loaded from {self.target.input_path}.")
        except Exception as e:
            self.cancel(reason=f"Error in check_data: {e}")

    def descriptive_analysis(self):
        """
        Generates descriptive statistics for the dataset.

        Descriptive statistics include the head, info, dtypes, describe, and isnull attributes.
        """
        try:
            self.check_data()
            self.head = self._data.head().T
            self.target.analysis_results["head"] = self.head
            result_logger.info(f"ENV:{self.target.env_name} HEAD:\n{self.head}")

            buffer = io.StringIO()
            self._data.info(buf=buffer)
            self.info = buffer.getvalue()
            buffer.close()
            self.target.analysis_results["info"] = self.info
            result_logger.info(f"ENV:{self.target.env_name} INFO:\n{self.info}")
            
            self.dtypes = self._data.dtypes
            self.target.analysis_results["dtypes"] = self.dtypes
            result_logger.info(f"ENV:{self.target.env_name} DTYPES:\n{self.dtypes}")
            
            self.describe = self._data.describe().T
            self.target.analysis_results["describe"] = self.describe
            result_logger.info(f"ENV:{self.target.env_name} DESCRIBE:\n{self.describe}")

            self.isnull = self._data.isnull().sum()
            self.target.analysis_results["isnull"] = self.isnull
            result_logger.info(f"ENV:{self.target.env_name} ISNULL:\n{self.isnull}")

        except Exception as e:
            logger.error(f"Error in descriptive_analysis: {e}")

    def calculate_correlation(self, *args, **kwargs):
        """
        Calculates the correlation matrix for the dataset.

        Implements the correlation matrix using the pandas DataFrame.corr() method.
        Performs dtype selection 'select_dtypes(include=['float64', 'int64'])' and 
        correlation filtering to generate low, moderate, and strong correlation results.

        Args:
            args: Positional arguments to pass to the pandas DataFrame.corr() method.
            kwargs: Keyword arguments to pass to the pandas DataFrame.corr() method.
            e.g. method='pearson', method='spearman' or method='kendall'
        """
        try:
            self.check_data()
            method = kwargs.get('method', 'pearson')
            self.correlation = self._data.select_dtypes(include=['float64', 'int64']).corr(*args, **kwargs)[self.target.label].abs()
            self.correlation = self.correlation.sort_values(ascending=False)
            self.correlation = self.correlation[self.correlation <= 1]
            self.target.analysis_results[f"{method}_correlation"] = self.correlation
            self.weak_correlation = self.correlation[self.correlation < 0.1]
            self.target.analysis_results[f"{method}_weak_correlation"] = self.weak_correlation
            self.moderate_correlation = self.correlation[(self.correlation >= 0.1) & (self.correlation < 0.5)]
            self.target.analysis_results[f"{method}_moderate_correlation"] = self.moderate_correlation
            self.strong_correlation = self.correlation[self.correlation >= 0.5]
            self.target.analysis_results[f"{method}_strong_correlation"] = self.strong_correlation
            result_logger.info(f"ENV:{self.target.env_name} STRONG CORRELATION:{self.strong_correlation}")
            result_logger.info(f"ENV:{self.target.env_name} MEDIUM CORRELATION:{self.moderate_correlation}")
            result_logger.info(f"ENV:{self.target.env_name} WEAK CORRELATION:\n{self.weak_correlation}")
        except Exception as e:
            logger.error(f"Error in calculate_correration: {e}")

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
            lambda_func(self._data)
            if save:
                plt.savefig(self.target.output_path / f"{self.target.env_name}_{self.target.filename}_{plotname}.png")
            plt.close(self.figures[plotname])
        except Exception as e:
            logger.error(f"Error in _generate_plot: {e}")

    def _calculate_grid_size(self, num_plots):
        cols = math.ceil(math.sqrt(num_plots))
        rows = math.ceil(num_plots / cols)
        return rows, cols

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
            rows, cols = self._calculate_grid_size(num_plots)
            figsize = (cols * 4, rows * 3)
            self.figures[plotname], subplots = plt.subplots(rows, cols, figsize=figsize)
            subplots = subplots.flatten() if num_plots > 1 else [subplots]

            for ax, (lambda_func, kwargs) in zip(subplots, lambdas):
                lambda_func(self._data, ax)
                ax.set_title(kwargs.get("title", ""), fontsize=10)
                ax.set_ylabel(kwargs.get("ylabel", ""), fontsize=10)
                ax.set_xlabel(kwargs.get("xlabel", ""), fontsize=10)

            plt.tight_layout()
            if save:
                plt.savefig(self.target.output_path / f"{self.target.env_name}_{self.target.filename}_{plotname}.png")
            plt.close(self.figures[plotname])
        except Exception as e:
            logger.error(f"Error in _generate_subplot: {e}")

    def missing_value_analysis(self, *args, **kwargs):
        """
        Generates plots for missing value analysis.

        The missing value analysis includes a barplot and matrix of missing values.
        For visualization, the missingno library is used to generate the plots.
        msno.bar() and msno.matrix() are used to generate the barplot and matrix, respectively.
        """
        try:
            self.check_data()
            self._generate_plot(
                lambda_func=lambda x: msno.bar(x).set_title("Missing Value Barplot"),
                plotname="missing_value_bar",
                *args,
                **kwargs
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
        Generates a heatmap of the correlation matrix for the dataset.

        For visualization, the seaborn library is used to generate the heatmap.
        sns.heatmap() is used to generate the heatmap.
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

        The boxplots are generated for each column of the dataset.
        For visualization, the seaborn library is used to generate the plots.
        sns.boxplot() is used to generate the boxplots.
        """
        try:
            self.check_data()
            lambdas = []
            for column in tqdm(self._data.select_dtypes(include=['float64', 'int64']).columns, desc="Outlier Barplots"):
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

        The distribution plots are generated for each column of the dataset.
        For visualization, the seaborn library is used to generate the plots.
        sns.histplot() is used to generate the distribution plots.
        """
        try:
            self.check_data()
            lambdas = []
            for column in tqdm(self._data.select_dtypes(include=['float64', 'int64']).columns, desc="Distribution Plots"):
                lambdas.append(
                    (
                    lambda x, ax, column=column: sns.histplot(x[column], kde=True, ax=ax),
                        {
                        "title":f"Skew: {round(self._data[column].skew(), 2)}",
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

        The scatter plots are generated for each column of the dataset.
        For visualization, the seaborn library is used to generate the plots.
        sns.scatterplot() is used to generate the scatter plots.
        """
        try:
            self.check_data()
            lambdas = []
            for column in tqdm(self._data.select_dtypes(include=['float64', 'int64']).columns, desc="Scatter Plots"):
                lambdas.append(
                    (
                    lambda x, ax, column=column: sns.scatterplot(x=x[column], y=x[self.target.label], ax=ax),
                        {
                        "title":f"Scatterplot of {column} vs {self.target.label}",
                        "ylabel":self.target.label,
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

        The count plots are generated for each column of the dataset.
        For visualization, the seaborn library is used to generate the plots.
        sns.countplot() is used to generate the count plots.
        """
        try:
            self.check_data()
            lambdas = []
            for column in tqdm(self._data.select_dtypes(include=['category']).columns, desc="Categorical Plots"):
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

        The time series plots are generated for each column of the dataset.
        For visualization, the seaborn library is used to generate the plots.
        sns.lineplot() is used to generate the time series plots.
        """
        try:
            self.check_data()
            lambdas = []
            for time_column in tqdm(self._data.select_dtypes(include=['datetime64']).columns):
                for column in tqdm(self._data.select_dtypes(include=['float64', 'int64']).columns, desc="Time Series Plots"):
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

        The over-index analysis includes line plots for each column of the dataset.
        For visualization, the seaborn library is used to generate the plots.
        sns.lineplot() is used to generate the line plots.
        """
        try:
            self.check_data()
            lambdas = []
            for column in tqdm(self._data.select_dtypes(include=['float64', 'int64']).columns, desc="Over Index Plots"):
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