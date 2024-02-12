import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from idstools._data_models import TargetData
from idstools._config import pprint_dynaconf
from idstools._helpers import emergency_logger, setup_logging, result_logger

logger = setup_logging(__name__)

@emergency_logger
class ModelOptimization():
    """This class is used to optimize the model."""
    def __init__(self, target: TargetData, pipeline: dict = None):
        try:
            logger.info("Initializing ModelOptimization")

            # initialize variables
            self._models = {}
            self.X_train = None
            self.X_test = None
            self.X_val = None
            self.y_train = None
            self.y_test = None
            self.y_val = None

            # load data
            self.target = target
            self._data = self.target.data.copy()
            logger.info(f"Data loaded from {target.input_path}.")

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

    def train_test_split(self):
        """
        This function is used to split the data into training and testing sets.
        """
        try:
            self.check_data()
            logger.info("Running train_test_split")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self._data[self.target.features], self._data[self.target.label], test_size=0.2, random_state=42)
            logger.info("train_test_split completed successfully")
        except Exception as e:
            self.cancel(reason=f"Error in train_test_split: {e}")
        
    def train_test_validation_split(self):
        """
        This function is used to split the data into training, testing, and validation sets.
        """
        try:
            self.check_data()
            logger.info("Running train_test_validation_split")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self._data[self.target.features], self._data[self.target.label], test_size=0.2, random_state=42)
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)
            logger.info("train_test_validation_split completed successfully")
        except Exception as e:
            self.cancel(reason=f"Error in train_test_validation_split: {e}")

    def split_on_yearly_basis(self):
        """
        This function is used to split the data into training and testing sets on a yearly basis.
        """
        try:
            self.check_data()
            logger.info("Running split_on_yearly_basis")
            self.X_train, self.X_test = self._data[self._data['year'] < 2019][self.target.features], self._data[self._data['year'] >= 2019][self.target.features]
            self.y_train, self.y_test = self._data[self._data['year'] < 2019][self.target.label], self._data[self._data['year'] >= 2019][self.target.label]
            logger.info("split_on_yearly_basis completed successfully")
        except Exception as e:
            self.cancel(reason=f"Error in split_on_yearly_basis: {e}")
    
    def time_series_split(self):
        """
        This function is used to split the data into training and testing sets.
        """
        try:
            self.check_data()
            logger.info("Running time_series_split")
            tscv = TimeSeriesSplit(n_splits=5)
            for train_index, test_index in tscv.split(self._data):
                self.X_train, self.X_test = self._data.iloc[train_index][self.target.features], self._data.iloc[test_index][self.target.features]
                self.y_train, self.y_test = self._data.iloc[train_index][self.target.label], self._data.iloc[test_index][self.target.label]
                logger.info("time_series_split completed successfully")
        except Exception as e:
            self.cancel(reason=f"Error in time_series_split: {e}")

    def linear_regression(self):
        """
        This function is used to run the linear regression model.
        """
        try:
            self.check_data()
            logger.info("Running linear_regression")
            reg = LinearRegression()
            reg.fit(self.X_train, self.y_train)
            reg.predict(self.X_test)
            self._models['linear_regression'] = reg
            logger.info("linear_regression completed successfully")
        except Exception as e:
            self.cancel(reason=f"Error in linear_regression: {e}")

    def lazy_regressor(self):
        """
        This function is used to run the lazy regressor model.
        """
        try:
            self.check_data()
            logger.info("Running lazy_regressor")
            reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
            models = reg.fit(self.X_train, self.X_test, self.y_train, self.y_test)
            self._models['lazy_regressor'] = models
            logger.info("lazy_regressor completed successfully")
        except Exception as e:
            self.cancel(reason=f"Error in lazy_regressor: {e}")

    def validation(self):
        """
        This function is used to validate the model.
        """
        try:
            self.check_data()

            logger.info("Running validation")
            for model in self._models:
                logger.info(f"Validation for {model}")
                prediction = self._models[model].predict(self.X_test)
                logger.info(f"Prediction for {model} completed successfully")
                r2 = r2_score(self.y_test, prediction).round(2)
                self.target.analysis_results[f"{model}_r2"] = r2
                adjusted_r2 = self.calculate_adjusted_r_squared(r2, len(self.y_test), len(self.X_test.columns))
                self.target.analysis_results[f"{model}_adjusted_r2"] = adjusted_r2
                mae = mean_absolute_error(self.y_test, prediction).round(2)
                self.target.analysis_results[f"{model}_mae"] = mae
                logger.info(f"R2 score for {model}: {r2}")
                logger.info(f"Adjusted R2 score for {model}: {adjusted_r2}")
                logger.info(f"Mean Absolute Error for {model}: {mae}")
            logger.info("Validation completed successfully")
        except Exception as e:
            self.cancel(reason=f"Error in validation: {e}")

    def calculate_adjusted_r_squared(self, r_squared, n, p):
        """
        Calculates the adjusted R-squared value for the model.

        Args:
            r_squared (float): The R-squared value of the model.
            n (int): The number of observations in the dataset.
            p (int): The number of predictors in the model (excluding the constant term).

        Returns:
            float: The adjusted R-squared value.
        """
        adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
        return adjusted_r_squared

    def run(self):
        """
        This function is used to run the model optimization.
        """
        try:
            self.check_data()
            logger.info("Running ModelOptimization")
            for model in self.pipeline:
                logger.info(f"Running model optimization for {model}")
                method = getattr(self, model)
                method()
                logger.info(f"Executed {model} of model_optimization.")
            logger.info("ModelOptimization completed successfully")
        except AttributeError as e:
            self.cancel(reason=f"Error in run: {e}")
        except Exception as e:
            self.cancel(reason=f"Error in run: {e}")
    
    def cancel(self, reason):
        logger.info(f"Cancel of model_optimization due to {reason}")
        exit(1)