import io
import math
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from idstools._data_models import TargetData
from idstools._helpers import use_decorator, emergency_logger, setup_logging

pd.set_option('display.precision', 2)
logger = setup_logging(__name__)

@use_decorator(emergency_logger)
class DataExplorer():
    """
    This class is used to explore the data.
    """
    def __init__(self, targets: list, pipeline: dict = None):
        try:
            logger.info("Initializing DataExplorer")
            self.targets = targets
            self.pipeline = pipeline if pipeline else {}

        except Exception as e:
            self.cancel(reason=f"Error in __init__: {e}")

    def _generate_plot(self, lambda_func, target: TargetData,  plotname: str = None, save: bool = True) -> None:
        """
        Generates a single plot for the data using seaborn plotting functions.
        
        Args:
            lambda (function): List of lambda functions to generate plots, where each lambda
                            function should expect a DataFrame and an ax as arguments.
            plotname (str): Filename to save the plot.
            save (bool): Whether to save the plot.
        """
        try:
            target_data = target.update_data()
            target.figures[f"{target.filename}_{plotname}"] = plt.figure(figsize=(len(target_data.columns), len(target_data.columns)/2))
            lambda_func(target_data)
            if save:
                path = target.output_path / target.env_name / target.step_name
                path.mkdir(parents=True, exist_ok=True)
                plt.savefig(path / f"{target.filename}_{plotname}.png")
            plt.close(target.figures[f"{target.filename}_{plotname}"])
        except Exception as e:
            logger.error(f"Error in _generate_plot: {e}")

    def _calculate_grid_size(self, num_plots) -> tuple:
        """
        Calculates the number of rows and columns for the grid of plots.

        Args:
            num_plots (int): The number of plots to generate.

        Returns:
            tuple: The number of rows and columns for the grid of plots.
        """
        if num_plots <= 1:
            return 1, 1
        cols = math.ceil(math.sqrt(num_plots))
        rows = math.ceil(num_plots / cols)
        return rows, cols

    def _generate_subplot(self, lambdas, target: TargetData, plotname: str = None, save: bool = True) -> None:
        """
        Generates plots for the data using seaborn plotting functions.

        Args:
            lambdas (list): List of tuples containing lambda functions to generate plots and plot parameters.
            plotname (str): Filename to save the plot.
            save (bool): Whether to save the plot.
        """
        try:
            target_data = target.update_data()
            num_plots = len(lambdas)
            rows, cols = self._calculate_grid_size(num_plots)
            figsize = (cols * 4, rows * 3)
            target.figures[f"{target.filename}_{plotname}"], subplots = plt.subplots(rows, cols, figsize=figsize)
            subplots = subplots.flatten() if num_plots > 1 else [subplots]

            for ax, (lambda_func, kwargs) in zip(subplots, lambdas):
                lambda_func(target_data, ax)

                title = kwargs.get("title", "")
                ylabel = kwargs.get("ylabel", "")
                xlabel = kwargs.get("xlabel", "")

                ax.set_title(title, fontsize=10)
                ax.set_ylabel(ylabel, fontsize=10)
                ax.set_xlabel(xlabel, fontsize=10)

            plt.tight_layout()
            if save:
                path = target.output_path / target.env_name / target.step_name
                path.mkdir(parents=True, exist_ok=True)
                plt.savefig(path / f"{target.filename}_{plotname}.png")
            plt.close(target.figures[f"{target.filename}_{plotname}"])
        except Exception as e:
            logger.error(f"Error in _generate_subplot: {e}")

    def missing_values_plot(self, target: TargetData, **kwargs) -> None:
        """
        Generates plots for missing value analysis.

        The missing value analysis includes a barplot and matrix of missing values.
        For visualization, the missingno library is used to generate the plots.
        msno.bar() and msno.matrix() are used to generate the barplot and matrix, respectively.
        """
        try:
            self._generate_plot(
                lambda_func=lambda x: msno.bar(x).set_title("Missing Value Barplot"),
                target=target,
                plotname="missing_value_bar",
                **kwargs
                )
            self._generate_plot(
                lambda_func=lambda x: msno.matrix(x).set_title("Missing Value Matrix"),
                target=target,
                plotname="missing_value_matrix",
                **kwargs
                )
        except Exception as e:
            logger.error(f"Error in missing_value_analysis: {e}")

    def correlation_plot(self, target: TargetData, **kwargs) -> None:
        """
        Generates a heatmap of the correlation matrix for the dataset.

        For visualization, the seaborn library is used to generate the heatmap.
        sns.heatmap() is used to generate the heatmap.
        """
        try:
            self._generate_plot(
                lambda_func=lambda x: sns.heatmap(
                    x.corr(numeric_only=True),
                    annot=True, cmap="coolwarm",
                    fmt=".2f"
                    ).set_title("Correlation Heatmap"),
                target=target,
                plotname="correlation_heatmap",
                **kwargs
                )
        except Exception as e:
            logger.error(f"Error in correlation_analysis: {e}")

    def outlier_plot(self, target: TargetData, **kwargs) -> None:
        """
        Generates boxplots for outlier analysis.

        The boxplots are generated for each column of the dataset.
        For visualization, the seaborn library is used to generate the plots.
        sns.boxplot() is used to generate the boxplots.
        """
        try:
            target_data = target.update_data()
            lambdas = []
            for column in target_data.select_dtypes(include=['float64', 'int64']).columns:
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
            self._generate_subplot(
                lambdas,
                target=target,
                plotname="outlier_boxplots",
                **kwargs
                ) if lambdas else logger.warning("No numeric columns found for outlier analysis.")
        except Exception as e:
            logger.error(f"Error in outlier_analysis: {e}")

    def distribution_plot(self, target: TargetData, **kwargs) -> None:
        """
        Generates distribution plots for the dataset.

        The distribution plots are generated for each column of the dataset.
        For visualization, the seaborn library is used to generate the plots.
        sns.histplot() is used to generate the distribution plots.
        """
        try:
            target_data = target.update_data()
            lambdas = []
            for column in target_data.select_dtypes(include=['float64', 'int64']).columns:
                lambdas.append(
                    (
                    lambda x, ax, column=column: sns.histplot(x[column], kde=True, ax=ax),
                        {
                        "title":f"Skew: {round(target_data[column].skew(), 2)}",
                        "ylabel":column,
                        "xlabel":"Distribution"
                        }
                    )
                )
            self._generate_subplot(
                lambdas,
                target=target,
                plotname="distribution_plots",
                **kwargs
                ) if lambdas else logger.warning("No numeric columns found for distribution analysis.")
        except Exception as e:
            logger.error(f"Error in distribution_analysis: {e}")

    def scatter_plot(self, target: TargetData, **kwargs):
        """
        Generates scatter plots for the dataset.

        The scatter plots are generated for each column of the dataset.
        For visualization, the seaborn library is used to generate the plots.
        sns.scatterplot() is used to generate the scatter plots.
        """
        try:
            target_data = target.update_data()
            lambdas = []
            for column in target_data.select_dtypes(include=['float64', 'int64']).columns:
                lambdas.append(
                    (
                    lambda x, ax, column=column: sns.scatterplot(x=x[column], y=x[target.label], ax=ax),
                        {
                        "title":f"Scatterplot of {column} vs {target.label}",
                        "ylabel":target.label,
                        "xlabel":column
                        }
                    )
                )
            self._generate_subplot(
                lambdas,
                target=target,
                plotname="scatter_plots",
                **kwargs
                ) if lambdas else logger.warning("No numeric columns found for scatter analysis.")
        except Exception as e:
            logger.error(f"Error in scatter_analysis: {e}")

    def categorical_plot(self, target: TargetData, **kwargs):
        """
        Generates count plots for categorical columns of the dataset.

        The count plots are generated for each column of the dataset.
        For visualization, the seaborn library is used to generate the plots.
        sns.countplot() is used to generate the count plots.
        """
        try:
            target_data = target.update_data()
            lambdas = []
            for column in target_data.select_dtypes(include=['category']).columns:
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
            self._generate_subplot(
                lambdas,
                target=target,
                plotname="categorical_plots",
                **kwargs
                ) if lambdas else logger.warning("No categorical columns found for categorical analysis.")
        except Exception as e:
            logger.error(f"Error in categorical_analysis: {e}")

    def time_series_plot(self, target: TargetData, **kwargs):
        """
        Generates time series plots for each column of the dataset.

        The time series plots are generated for each column of the dataset.
        For visualization, the seaborn library is used to generate the plots.
        sns.lineplot() is used to generate the time series plots.
        """
        try:
            target_data = target.update_data()
            lambdas = []
            for time_column in target_data.select_dtypes(include=['datetime64']).columns:
                for column in target_data.select_dtypes(include=['float64', 'int64']).columns:
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
            self._generate_subplot(
                lambdas,
                target=target,
                plotname="time_series_plots",
                **kwargs
                ) if lambdas else logger.warning("No time series columns found for time series analysis.")
        except Exception as e:
            logger.error(f"Error in time_series_analysis: {e}")

    def over_index_plot(self, target: TargetData, **kwargs):
        """
        Generates plots for over-index analysis.

        The over-index analysis includes line plots for each column of the dataset.
        For visualization, the seaborn library is used to generate the plots.
        sns.lineplot() is used to generate the line plots.
        """
        try:
            target_data = target.update_data()
            lambdas = []
            for column in target_data.select_dtypes(include=['float64', 'int64']).columns:
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
            self._generate_subplot(
                lambdas,
                target=target,
                plotname="over_index_plots",
                **kwargs
                ) if lambdas else logger.warning("No numeric columns found for over index analysis.")
        except Exception as e:
            logger.error(f"Error in over_index_analysis: {e}")

    def descriptive_analysis(self, target: TargetData, **kwargs) -> None:
        """
        Performs a descriptive analysis of the data.

        Args:
        - target: The target data to analyze.
        - args: Arguments to pass to the pandas.DataFrame.head() and pandas.DataFrame.tail() methods.
        - kwargs: Keyword arguments to pass to the pandas.DataFrame.head() and pandas.DataFrame.tail() methods.
        """
        try:
            target_data = target.update_data()
            results = self._add_result_category(target.analysis_results['DataExplorer'], "descriptive_analysis")

            head = kwargs.get('head', 5)
            tail = kwargs.get('tail', 5)

            results[f"head_{head}"] = target_data.head(head).T
            results[f"tail_{tail}"] = target_data.tail(tail).T

            with io.StringIO() as buffer:
                target_data.info(buf=buffer)
                results["info"] = buffer.getvalue()

            results["dtypes"] = target_data.dtypes
            results["describe"] = target_data.describe().T
            results["isnull"] = target_data.isnull().sum()
        except Exception as e:
            logger.error(f"Error in descriptive_analysis: {e}")

    def correlation_analysis(self, target: TargetData, **kwargs) -> None:
        """
        Calculates the correlation between the features and the target variable.

        Args:
        - target: The target data to analyze.
        - kwargs: Keyword arguments to pass to the pandas.DataFrame.corr() method.
        """
        try:
            target_data = target.update_data()
            results = self._add_result_category(target.analysis_results['DataExplorer'], "correlation_analysis")

            method = kwargs.get('method', 'pearson')
            min_periods= kwargs.get('min_periods', 1)
            numeric_only= kwargs.get('numeric_only', False)

            correlation = target_data.select_dtypes(include=['float64', 'int64']).corr(method=method, min_periods=min_periods, numeric_only=numeric_only)[target.label].abs()
            correlation = correlation.sort_values(ascending=False)
            
            correlation = correlation[correlation <= 1]
            results[f"{method}_correlation"] = correlation
            
            strong_correlation = correlation[correlation >= 0.5]
            results[f"{method}_strong_correlation"] = strong_correlation

            moderate_correlation = correlation[(correlation >= 0.1) & (correlation < 0.5)]
            results[f"{method}_moderate_correlation"] = moderate_correlation
            
            weak_correlation = correlation[correlation < 0.1]
            results[f"{method}_weak_correlation"] = weak_correlation
        except Exception as e:
            logger.error(f"Error in calculate_correration: {e}")

    def vif_analysis(self, target: TargetData, **kwargs) -> None:
        """
        Calculates the variance inflation factor for the dataset, handling potential numerical issues gracefully.

        The variance inflation factor is calculated using the statsmodels library.
        This version includes handling for divide by zero warnings and improves data appending efficiency.
        """
        try:
            target_data = target.update_data()
            results = self._add_result_category(target.analysis_results['DataExplorer'], "vif_analysis")

            vif_data = add_constant(target_data.select_dtypes(include=['float64', 'int64']))
            vif_data.drop(target.label, axis=1, inplace=True)
            
            vif_records = []
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                for i in range(vif_data.shape[1]):
                    try:
                        vif_value = variance_inflation_factor(vif_data.values, i)
                    except Exception as e:
                        logger.error(f"VIF calculation failed for feature at index {i} due to: {e}")
                        vif_value = np.nan  # Assign NaN or another placeholder value indicating failure
                    
                    vif_records.append({"VIF Factor": vif_value, "features": vif_data.columns[i]})
                
                # Convert the list of dictionaries to a DataFrame
                vif = pd.DataFrame(vif_records)
                
                # Log any warnings captured
                for warning in w:
                    logger.warning(f"VIF warning: {warning.message}")
                
            results["variance_inflation_factor"] = vif
        except Exception as e:
            logger.error(f"Error in variance_inflation_factor method: {e}")

    def _add_result_category(self, target: dict, category: str = "DataExplorer") -> dict:
        """
        Prepares the analysis results for the target.
        """
        try:
            return target.setdefault(category, {})
        except Exception as e:
            logger.error(f"Error in _prepare_analysis_results: {e}")

    def _run_pipeline(self, target: TargetData):
        """
        Runs the pipeline for each target.
        """
        for step in self.pipeline:
            explorer = step.get("explorer")
            config = step.get("config") if step.get("config") else {}

            try:
                method = getattr(self, explorer)
            except AttributeError:
                logger.error(f"Error in run_pipeline: No explorer method named '{explorer}'")
                continue

            try:
                method(target, **config)
                logger.info(f"Executed explorer '{explorer}' with config {config}")
            except Exception as e:
                logger.error(f"Error executing explorer '{explorer}': {e}")

    def _log_results(self, target: TargetData):
        """
        Logs the results of the data exploration.
        """
        try:
            result_logger = setup_logging("data_explorer_results", env_name=target.env_name, step_name=target.step_name, filename=f"DataExplorer_{target.filename}")
            result_logger.info(f"Logging results for target {target.filename} in {target.env_name}:{target.step_name}.")

            for result_category, results in target.analysis_results['DataExplorer'].items():
                result_logger.info(f"Results for {result_category}:")

                log_results = {key: (value.to_string(float_format='{:.3f}'.format) if hasattr(value, 'to_string') else str(value)) for key, value in results.items()}

                for key, value in log_results.items():
                    result_logger.info(f"{key}:\n{value}\n")
        except Exception as e:
            logger.error(f"Error in _log_results: {e}")

    def run(self):
        """
        Runs the pipeline of data exploration methods.
        """
        try:
            if not self.pipeline:
                logger.warning("No pipeline provided. Running descriptive_analysis only.")
                self.pipeline = {"descriptive_analysis": None}
            for target in self.targets:
                self.current_target = target
                self._add_result_category(target.analysis_results)
                self._run_pipeline(target)
                self._log_results(target)
                logger.info(f"Data exploration completed for target {target.filename} in {target.env_name}:{target.step_name}.")
        except (Exception, KeyboardInterrupt) as e:
            self.cancel(reason=f"Run canceled: {e}")

    def cancel(self, reason: str):
        """
        Cancels the data exploration process.
        """
        logger.info(f"Cancel data_explorer due to <{reason}>.")
        exit(1)