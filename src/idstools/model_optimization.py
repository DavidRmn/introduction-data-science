# TODO: Add docstrings to all functions
# TODO: Add type hints to all functions
# TODO: Change execution of pipeline to one model at a time
# TODO: Better cofnig handling e.g. save_model not in _perpare_models()
import joblib
from importlib import import_module
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from idstools._data_models import TargetData
from idstools._helpers import emergency_logger, setup_logging, add_category

logger = setup_logging(__name__)

@emergency_logger
class ModelOptimization():
    """This class is used to optimize the model."""
    def __init__(self, targets: dict, pipeline: list[dict] = None):
        try:
            logger.info("Initializing ModelOptimization")
            
            self.models = {}
            self.targets = targets
            self.pipeline = pipeline if pipeline else {}

        except Exception as e:
            self.cancel(reason=f"Error in __init__: {e}")
    
    def _retrieve_function(self, func_name, module=None) -> callable:
        if callable(func_name):
            return func_name
        try:
            if module:
                module = import_module(module)
                return getattr(module, func_name)
            else:
                return globals()[func_name]
        except Exception as e:
            raise ImportError(f"Could not import function: {func_name} from module: {module}. Error: {e}")

    def _retrieve_model(self, model_config: dict[dict]):
        """This function is used to retrieve the model."""
        try:
            model = model_config.get("model")
            module = model_config.get("module")
            config = model_config.get("config", {})
            logger.info(f"Retrieving model {model}.")
            model_object = add_category(self.models, model)

            model_class = self._retrieve_function(func_name=model, module=module)
            model_object["model"] = model_class(**config)
        except Exception as e:
            self.cancel(reason=f"Error in _retrieve_model: {e}")

    def _prepare_data(self, model_config: dict[dict]):
        """This function is used to prepare the data for the model."""
        try:
            split = model_config.get("split", None)
            for name, target in self.targets.items() if name in split.get("targets") else None:
                logger.info(f"Preparing target for {model_config.get('module')}.{model_config.get('model')}.")
                model_target = add_category(self.models[model_config.get("model")], name)

                target_data = target.update_data()

                test_size = split.get("config").get("test_size", 0.2)
                random_state = split.get("config").get("random_state", 42)

                X_train, X_test, y_train, y_test = train_test_split(target_data[target.features], target_data[target.label], test_size=test_size, random_state=random_state)
                model_target["X_train"] = X_train
                model_target["X_test"] = X_test
                model_target["y_train"] = y_train
                model_target["y_test"] = y_test
                logger.info(f"Training set: {len(X_train)} samples\nTesting set: {len(X_test)} samples")
        except Exception as e:
            self.cancel(reason=f"Error in _prepare_data: {e}")

    def _fit_model(self, model_config: dict[dict]):
        """This function is used to fit the model."""
        try:
            model = model_config.get("model")
            model = self.models[model]
            for target in model_config.get("targets", {}):
                logger.info(f"Fitting {target} for {model['model']}.")
                target = self.targets[target]
                model['model'].fit(model[target]["X_train"], model[target]["y_train"])
        except Exception as e:
            self.cancel(reason=f"Error in _fit_model: {e}")

    def _prepare_validation(self, model_config: dict[dict]):
        """This function is used to validate the model."""
        try:
            model = model_config.get("model")
            model = self.models[model]
            validation = model_config.get("validation")
            for target in validation.get("targets"):
                logger.info(f"Validating {target} for {model['model']}.")
                model = add_category(model, target)
                
                target = self.targets[target]
                target_data = target.update_data()
                
                model[target]["X_test"] = target_data[target.features]
                model[target]["y_test"] = target_data[target.label]
        except Exception as e:
            self.cancel(reason=f"Error in _prepare_validation: {e}")

    def _predict_model(self, model_config: dict[dict]):
        """This function is used to predict the model."""
        try:
            model = model_config.get("model")
            model = self.models[model]
            validation = model_config.get("validation")
            for target in self.validation_targets if 'validation_targets' in validation.get("targets") else None:
                logger.info(f"Predicting {target} for {model['model']}.")
                model["y_pred"] = model['model'].predict(model[target]["X_test"])
            for target in self.targets if 'targets' in validation.get("targets") else None:
                logger.info(f"Predicting {target} for {model['model']}.")
                model["y_pred"] = model['model'].predict(model[target]["X_test"])
        except Exception as e:
            self.cancel(reason=f"Error in _predict_model: {e}")       

    def _validate_model(self, model_config: dict[dict]):
        """This function is used to validate the model."""
        try:
            validation = model_config.get("validation")
            for target in model_config.get("targets"):
                self.models[target] = add_category(self.models[target], "validation")
                logger.info(f"Validating model for target {target}.")
                for target in validation.get("targets"):
                    pass
        except Exception as e:
            self.cancel(reason=f"Error in _validate_model: {e}")

    def run(self):
        """This function is used to run the model optimization pipeline."""
        try:
            logger.info("Running model optimization pipeline.")
            for model_config in self.pipeline:
                self._retrieve_model(model_config)
                self._prepare_data(model_config)
                self._fit_model(model_config)
                self._prepare_validation(model_config)
                self._predict_model(model_config)
                #self._validate_model(model_config)
                #self._save_model(model_config)
                #self._log_results()
        except Exception as e:
            self.cancel(reason=f"Error in run: {e}")

    def cancel(self, reason):
        logger.info(f"Cancel of model_optimization due to {reason}")
        exit(1)