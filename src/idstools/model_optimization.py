import pickle
import importlib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
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

    def train_test_split(self, model: str, test_size: float = 0.2, random_state: int = 42):
        """This function is used to split the data into training and testing sets."""
        try:
            logger.info("Splitting data into training and testing sets.")
            X_train, X_test, y_train, y_test = train_test_split(self._data[self.target.features], self._data[self.target.label], test_size=test_size, random_state=random_state)
            logger.info(f"Training set: {len(X_train)} samples\nTesting set: {len(X_test)} samples")
            self._models[model]['X_train'] = X_train
            self._models[model]['X_test'] = X_test
            self._models[model]['y_train'] = y_train
            self._models[model]['y_test'] = y_test
        except Exception as e:
            self.cancel(reason=f"Error in train_test_split: {e}")

    def _prepare_data(self, model: dict):
        """This function is used to prepare the data for the model."""
        try:
            if not model['split']:
                logger.info(f"No split method provided for model {model['model']}.")
                return
            logger.info("Preparing data for the model.")
            split = getattr(self, model['split']['method'])
            split(model=model['model'], test_size=model['split']['config']['test_size'], random_state=model['split']['config']['random_state'])      
        except Exception as e:
            self.cancel(reason=f"Error in _prepare_data: {e}")

    def _retrieve_model(self, model: dict):
        try:
            module = importlib.import_module(model['module'])
            model_class = getattr(module, model['model'])
            self._models[model['model']] = {'model': model_class(**model['config'])}
        except AttributeError as e:
            self.cancel(reason=f"Error in _retrieve_model: {e}")

    def _prepare_validation(self, model: dict):
        """This function is used to prepare the validation for the model."""
        try:
            if not model['validation']:
                logger.info(f"No validation method provided for model {model['model']}.")
                return
            logger.info("Preparing validation for the model.")
            self._models[model['model']]['validation'] = model['validation']
            if model['validation']['use_validation_target'] and self.validation_target is not None:
                self._models[model['model']]['X_test'] = self.X_validation
                self._models[model['model']]['y_test'] = self.y_validation
            else:
                logger.info(f"No validation data provided for model {model['model']}. Using target data.")
        except Exception as e:
            self.cancel(reason=f"Error in _prepare_validation: {e}")

    def _train_model(self, model: dict):
        """This function is used to fit the models."""
        try:
            logger.info(f"Fitting model {model['model']}.")
            if model['model'] == 'LazyRegressor':
                self._models[model['model']]['models'], self._models[model['model']]['predictions'] = self._models[model['model']]['model'].fit(X_train=self._models[model['model']]['X_train'], X_test=self._models[model['model']]['X_test'], y_train=self._models[model['model']]['y_train'], y_test=self._models[model['model']]['y_test'])
                self.result_logger.info(f"Performance of {model['model']} models:\n{self._models[model['model']]['models']}.")
            else:
                self._models[model['model']]['model'].fit(self._models[model['model']]['X_train'], self._models[model['model']]['y_train'])
        except KeyError as e:
            self.cancel(reason=f"Error in train_models: {e}")

    def prepare_models(self):
        """This function is used to prepare the models for optimization."""
        try:
            logger.info("Preparing models for optimization.")
            for model in self.pipeline:
                if model['load_model']:
                    logger.info(f"Loading model {model['model']}.")
                    model_path = Path(model['model']).absolute()
                    self._models[model['model']] = {'model': pickle.load(open(model_path, 'rb'))}
                else:
                    logger.info(f"Preparing model {model['model']}.")
                    self._retrieve_model(model=model)
                    self._prepare_data(model=model)
                    self._train_model(model=model)
                self._prepare_validation(model=model)
        except Exception as e:
            self.cancel(reason=f"Error in prepare_models: {e}")

    def predict_models(self):
        """This function is used to predict the models."""
        try:
            logger.info("Predicting models.")
            for model in self._models:
                if model == 'LazyRegressor':
                    logger.info(f"Not predicting model {model}.")
                    continue
                self._models[model]['predictions'] = self._models[model]['model'].predict(self._models[model]['X_test'])
        except Exception as e:
            self.cancel(reason=f"Error in predict_models: {e}")

    def r2_score(self, model: str):
        """This function is used to calculate the R2 score."""
        try:
            logger.info(f"Calculating R2 score for model {model}.")
            self._models[model]['r2_score'] = r2_score(self._models[model]['y_test'], self._models[model]['predictions'])
            self.result_logger.info(f"R2 score for model {model} is {self._models[model]['r2_score']}.")
        except Exception as e:
            self.cancel(reason=f"Error in r2_score: {e}")
    
    def mae(self, model: str):
        """This function is used to calculate the mean absolute error."""
        try:
            logger.info(f"Calculating mean absolute error for model {model}.")
            self._models[model]['mae'] = mean_absolute_error(self._models[model]['y_test'], self._models[model]['predictions'])
            self.result_logger.info(f"Mean absolute error for model {model} is {self._models[model]['mae']}.")
        except Exception as e:
            self.cancel(reason=f"Error in mae: {e}")

    def validate_models(self):
        """This function is used to validate the models."""
        try:
            logger.info("Validating models.")
            for model in self._models:
                if model == 'LazyRegressor':
                    logger.info(f"Not validating model {model}.")
                    continue
                if not model['validation']:
                    logger.info(f"No validation method provided for model {model}.")
                    return
                for method in model['validation']['methods']:
                        method_instance = getattr(self, method)
                        method_instance(model=model)
        except Exception as e:
                self.cancel(reason=f"Error in validate_models: {e}")

    def run(self):
        """This function is used to run the model optimization pipeline."""
        try:
            logger.info("Running model optimization pipeline.")
            self.prepare_models()
            self.predict_models()
            self.validate_models()
        except Exception as e:
            self.cancel(reason=f"Error in run: {e}")

    def cancel(self, reason):
        logger.info(f"Cancel of model_optimization due to {reason}")
        exit(1)