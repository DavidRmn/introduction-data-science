import pandas as pd
from seaborn import residplot
from lazypredict.Supervised import LazyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split, learning_curve, LearningCurveDisplay
from idstools._data_models import TargetData
from idstools._config import pprint_dynaconf
from idstools._helpers import emergency_logger, setup_logging

logger = setup_logging(__name__)

@emergency_logger
class ModelOptimization():
    """This class is used to optimize the model."""
    def __init__(self, target: TargetData, validation_target: TargetData = None, pipeline: dict = None):
        try:
            logger.info("Initializing ModelOptimization")
            self.result_logger = setup_logging("model_optimization_results", env_name=target.env_name, step_name=target.step_name, filename="ModelOptimization")
            
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
            self._data = self.target.update_data()
            self.target.analysis_results[self.target.env_name] = {self.target.step_name: {"ModelOptimization": {}}}
            self.output_path = self.target.output_path / self.target.env_name / self.target.step_name
            self.output_path.mkdir(parents=True, exist_ok=True)

            if validation_target:
                self.validation_target = validation_target
                self.y_validation = self.validation_target.data[self.validation_target.label]
                self.X_validation = self.validation_target.data[self.validation_target.features]
            else:
                self.validation_target = None
                logger.info(f"No validation data provided.")

            if not pipeline:
                self.pipeline = {}
                logger.info(f"Please provide a pipeline configuration.")
            else:
                self.pipeline = pipeline
                logger.info(f"Pipeline configuration:\n{pprint_dynaconf(pipeline)}")

        except Exception as e:
            self.cancel(reason=f"Error in __init__: {e}")

    # ------------------------ Data Splitting Functions ------------------------

    def train_test_split(self):
        """
        This function is used to split the data into training and testing sets.
        """
        try:
            self._data = self.target.update_data()
            logger.info("Running train_test_split")
            if not self.target.label:
                self.cancel(reason="Target label not provided.")
            suitable_dtypes = ['int', 'float', 'object', 'category']
            suitable_features = self._data.select_dtypes(include=suitable_dtypes).columns.intersection(self.target.features)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self._data[suitable_features], self._data[self.target.label], test_size=0.2, random_state=42)
            logger.info("train_test_split completed successfully")
        except Exception as e:
            self.cancel(reason=f"Error in train_test_split: {e}")

    # ------------------------ Model Functions ------------------------

    def linear_regression(self):
        """
        This function is used to run the linear regression model.
        """
        try:
            logger.info("Running linear_regression")
            reg = LinearRegression()
            reg.fit(self.X_train, self.y_train)
            self._models['linear_regression'] = reg
            logger.info("linear_regression completed successfully")
        except Exception as e:
            self.cancel(reason=f"Error in linear_regression: {e}")

    def lazy_regressor(self):
        """
        This function is used to run the lazy regressor model.
        """
        try:
            self._data = self.target.update_data()
            logger.info("Running lazy_regressor")
            reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
            models = reg.fit(self.X_train, self.X_test, self.y_train, self.y_test)
            self._models['lazy_regressor'] = models
            logger.info("lazy_regressor completed successfully")
        except Exception as e:
            self.cancel(reason=f"Error in lazy_regressor: {e}")

    # ------------------------ Model Validation Functions ------------------------

    def validate_models(self):
        """
        This function is used to validate the models.
        """
        try:
            self._data = self.target.update_data()

            logger.info("Running validation")
            for model in self._models:
                logger.info(f"Validation for {model}")
                if self.validation_target:
                    logger.info(f"Validation for {model} using validation data")
                    prediction = self._models[model].predict(self.X_validation)
                    r2 = r2_score(self.y_validation, prediction).round(2)
                    self.target.analysis_results[f"{model}_r2"] = r2
                    adjusted_r2 = self.calculate_adjusted_r_squared(r2, len(self.y_validation), len(self.X_validation.columns))
                    self.target.analysis_results[self.target.env_name][self.target.step_name]["ModelOptimization"][f"{model}_adjusted_r2"] = adjusted_r2
                    mae = mean_absolute_error(self.y_validation, prediction).round(2)
                    self.target.analysis_results[self.target.env_name][self.target.step_name]["ModelOptimization"][f"{model}_mae"] = mae
                    self.result_logger.info(f"R2 score for {model}: {r2}")
                    self.result_logger.info(f"Adjusted R2 score for {model}: {adjusted_r2}")
                    self.result_logger.info(f"Mean Absolute Error for {model}: {mae}")
                else:
                    logger.info(f"Validation for {model} using test data")
                    prediction = self._models[model].predict(self.X_test)
                    r2 = r2_score(self.y_test, prediction).round(2)
                    self.target.analysis_results[f"{model}_r2"] = r2
                    adjusted_r2 = self.calculate_adjusted_r_squared(r2, len(self.y_test), len(self.X_test.columns))
                    self.target.analysis_results[self.target.env_name][self.target.step_name]["ModelOptimization"][f"{model}_adjusted_r2"] = adjusted_r2
                    mae = mean_absolute_error(self.y_test, prediction).round(2)
                    self.target.analysis_results[self.target.env_name][self.target.step_name]["ModelOptimization"][f"{model}_mae"] = mae
                    self.result_logger.info(f"R2 score for {model}: {r2}")
                    self.result_logger.info(f"Adjusted R2 score for {model}: {adjusted_r2}")
                    self.result_logger.info(f"Mean Absolute Error for {model}: {mae}")
                for coefficient, feature in zip(self._models[model].coef_, self.X_train.columns):
                    self.result_logger.info(f"Feature: {feature} - Coefficient: {abs(round(coefficient, 2))}")
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
    
    def residual_analysis(self):
        """
        This function is used to analyze the residuals of the model.
        """
        try:
            self._data = self.target.update_data()
            logger.info("Running residual_analysis")
            for model in self._models:
                logger.info(f"Residual analysis for {model}")
                prediction = self._models[model].predict(self.X_test)
                self.target.analysis_results[self.target.env_name][self.target.step_name]["ModelOptimization"][f"{model}_prediction"] = prediction
                residuals = self.y_test - prediction
                logger.info(f"Residuals for {model} completed successfully")
                self.target.analysis_results[self.target.env_name][self.target.step_name]["ModelOptimization"][f"{model}_residuals"] = residuals
                logger.info(f"Residual analysis for {model} completed successfully")
            logger.info("Residual analysis completed successfully")
        except Exception as e:
            self.cancel(reason=f"Error in residual_analysis: {e}")

    def plot_residuals(self, save: bool = False):
        """
        This function is used to plot the residuals of the model.
        """
        try:
            self._data = self.target.update_data()
            logger.info("Running plot_residuals")
            for model in self._models:
                logger.info(f"Plotting residuals for {model}")
                residuals = self.target.analysis_results[self.target.env_name][self.target.step_name]["ModelOptimization"][f"{model}_residuals"]
                prediction = self.target.analysis_results[self.target.env_name][self.target.step_name]["ModelOptimization"][f"{model}_prediction"]
                residuals = residplot(x=prediction, y=residuals)
                residuals.set(xlabel='Fitted values', ylabel='Residuals')
                residuals.set_title(f"Residuals for {model}")
                self.target.analysis_results[self.target.env_name][self.target.step_name]["ModelOptimization"][f"{model}_residuals_plot"] = residuals
                if save:
                    residuals.get_figure().savefig(f"{self.target.output_path}/residuals_{model}.png")
                logger.info(f"Residuals for {model} plotted successfully")
            logger.info("Plot_residuals completed successfully")
        except Exception as e:
            self.cancel(reason=f"Error in plot_residuals: {e}")

    def feature_importance(self):
        """
        This function is used to calculate the feature importance of the model.
        """
        try:
            self._data = self.target.update_data()
            logger.info("Running feature_importance")
            for model in self._models:
                logger.info(f"Feature importance for {model}")
                if model == 'linear_regression':
                    feature_importance = self._models[model].coef_
                else:
                    feature_importance = self._models[model].feature_importances_
                self.target.analysis_results[self.target.env_name][self.target.step_name]["ModelOptimization"][f"{model}_feature_importance"] = feature_importance
                logger.info(f"Feature importance for {model} completed successfully")
            logger.info("Feature importance completed successfully")
        except Exception as e:
            self.cancel(reason=f"Error in feature_importance: {e}")

    def plot_feature_importance(self):
        """
        This function is used to plot the feature importance of the model.
        """
        try:
            self._data = self.target.update_data()
            logger.info("Running plot_feature_importance")
            for model in self._models:
                logger.info(f"Plotting feature importance for {model}")
                feature_importance = self.target.analysis_results[self.target.env_name][self.target.step_name]["ModelOptimization"][f"{model}_feature_importance"]
                pd.Series(feature_importance, index=self.X_train.columns).nlargest(10).plot(kind='barh')
                logger.info(f"Feature importance for {model} plotted successfully")
            logger.info("Plot_feature_importance completed successfully")
        except Exception as e:
            self.cancel(reason=f"Error in plot_feature_importance: {e}")

    def learning_curve(self):
        """
        This function is used to plot the learning curve of the model.
        """
        try:
            self._data = self.target.update_data()
            logger.info("Running learning_curve")
            for model in self._models:
                logger.info(f"Plotting learning curve for {model}")
                if model == 'linear_regression':
                    train_sizes, train_scores, test_scores = learning_curve(self._models[model], self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error')
                    display = LearningCurveDisplay(train_sizes=train_sizes, train_scores=train_scores, test_scores=test_scores, score_name='neg_mean_squared_error')
                    display.plot()
                else:
                    train_sizes, train_scores, test_scores = learning_curve(self._models[model], self.X_train, self.y_train, cv=5, scoring='r2')
                self.target.analysis_results[self.target.env_name][self.target.step_name]["ModelOptimization"][f"{model}_learning_curve"] = (train_sizes, train_scores, test_scores)
                logger.info(f"Learning curve for {model} plotted successfully")
            logger.info("Learning_curve completed successfully")
        except Exception as e:
            self.cancel(reason=f"Error in learning_curve: {e}")

    def run(self):
        """
        This function is used to run the model optimization.
        """
        try:
            self._data = self.target.update_data()
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