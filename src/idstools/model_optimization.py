# TODO: Add docstrings to all functions
# TODO: Add type hints to all functions
# TODO: Change execution of pipeline to one model at a time
# TODO: Better cofnig handling e.g. save_model not in _perpare_models()
import joblib
import importlib
import pandas as pd
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
            self.validation_target = None
            self.X_validation = None
            self.y_validation = None

            # load data
            self.target = target
            self._data = self.target.update_data()
            self.target.analysis_results[self.target.env_name] = {self.target.step_name: {"ModelOptimization": {}}}
            self.output_path = self.target.output_path / self.target.env_name / self.target.step_name
            self.output_path.mkdir(parents=True, exist_ok=True)

            if validation_target:
                self.validation_target = validation_target
                validation_data = self.validation_target.update_data()
                self.y_validation = validation_data[self.validation_target.label]
                self.X_validation = validation_data[self.validation_target.features]
                self.X_validation = self.X_validation.reindex(columns=self.target.features, fill_value=0)
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
            if not 'save_model' in model.keys():
                logger.info(f"No save_model configuration provided for model {model['model']}.")
                self._models[model['model']]['save_model'] = None
            else:
                self._models[model['model']]['save_model'] = {'path': Path(model['save_model']['path']).absolute()}
            if not 'validation' in model.keys():
                logger.info(f"No validation configuration provided for model {model['model']}.")
                self._models[model['model']]['validation'] = None
                return
            if not 'methods' in model['validation'].keys():
                logger.info(f"No validation method provided for model {model['model']}.")
                self._models[model['model']]['validation'] = None
                return
            logger.info("Preparing validation for the model.")
            self._models[model['model']]['validation'] = model['validation']
            if model['validation']['use_validation_target'] and self.validation_target is not None:
                self._models[model['model']]['X_test'] = pd.DataFrame(self.X_validation, columns=self._models[model['model']]['features'])
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
                    filename = model_path.name.removesuffix(".pkl") + "_features.pkl"
                    feature_path = Path(model_path.parent / filename).absolute()
                    self._models[model['model']] = {'model': joblib.load(open(model_path, 'rb'))}
                    self._models[model['model']]['features'] = joblib.load(open(feature_path, 'rb'))
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
            for model, config in self._models.items():
                if model == 'LazyRegressor':
                    logger.info(f"Not predicting model {model}.")
                else:
                    self._models[model]['predictions'] = self._models[model]['model'].predict(config['X_test'])
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
            for model, config in self._models.items():
                if model == 'LazyRegressor':
                    logger.info(f"Not validating model {model}.")
                    continue
                if config['validation'] is None:
                    logger.info(f"No validation method provided for model {model}.")
                    return
                for method in config['validation']['methods']:
                        method_instance = getattr(self, method)
                        method_instance(model=model)
        except Exception as e:
                self.cancel(reason=f"Error in validate_models: {e}")

    def save_models(self):
        """This function is used to save the models."""
        try:
            logger.info("Saving models.")
            for model, config in self._models.items():
                if config['save_model'] is not None and 'path' in config['save_model'].keys():
                    logger.info(f"Saving model {model}.")
                    model_path = config['save_model']['path']
                    feature_path = model_path.parent / f"{model_path.name.removesuffix('.pkl')}_features.pkl"
                    joblib.dump(self._models[model]['model'], open(model_path, 'wb'))
                    joblib.dump(self.target.features, open(feature_path, 'wb'))
                else:
                    logger.error(f"Please provide a path to save the model {model}.")
        except Exception as e:
            self.cancel(reason=f"Error in save_models: {e}")

    def run(self):
        """This function is used to run the model optimization pipeline."""
        try:
            logger.info("Running model optimization pipeline.")
            self.prepare_models()
            self.predict_models()
            self.validate_models()
            self.save_models()
        except Exception as e:
            self.cancel(reason=f"Error in run: {e}")

    def cancel(self, reason):
        logger.info(f"Cancel of model_optimization due to {reason}")
        exit(1)