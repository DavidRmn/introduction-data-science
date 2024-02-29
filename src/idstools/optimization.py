#TODO: Implement log results function
#TODO: Implement GridSearch class
import joblib
from pathlib import Path
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

    def _load_model(self, model_config: dict[dict]):
        """This function is used to load the model."""
        try:
            id = model_config.get("id")
            load = model_config.get("load", {})
            model = load.get("model", None)
            features = load.get("features", None)
            if model:
                logger.info(f"Loading model {model}.")
                model_object = add_category(self.models, id)
                model_path = Path(model).absolute()
                model_object['model'] = joblib.load(open(model_path, 'rb'))
            if features:
                logger.info(f"Loading features {features}.")
                model_object = add_category(self.models, id)
                feature_path = Path(features).absolute()
                model_object['features'] = joblib.load(open(feature_path, 'rb'))
        except Exception as e:
            self.cancel(reason=f"Error in _load_model: {e}")
    
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
            id = model_config.get("id")
            model = model_config.get("model")
            module = model_config.get("module", None)
            load = model_config.get("load", None)
            config = model_config.get("config", {})
            if load:
                self._load_model(model_config)
                return
            elif model:
                logger.info(f"Retrieving model {model}.")
                model_target = add_category(self.models, id)
                model = self._retrieve_function(model, module)
                model_target['model'] = model(**config)
                model_target['config'] = config
            else:
                logger.info(f"No model provided for {model}.")

        except Exception as e:
            self.cancel(reason=f"Error in _retrieve_model: {e}")

    def _prepare_data(self, model_config: dict[dict]):
        """This function is used to prepare the data for the model."""
        try:
            id = model_config.get("id")
            model = model_config.get("model")
            split = model_config.get("split", None)
            for target in model_config.get("targets", []):
                target = self.targets[target] if target in self.targets.keys() else None
                if target is None:
                    logger.info(f"No target provided for model {model}.")
                    return
                logger.info(f"Preparing target for {model}.")
                model_target = add_category(self.models[id], target.name)

                target_data = target.update_data()

                test_size = split.get("config").get("test_size", 0.2)
                random_state = split.get("config").get("random_state", 42)

                X_train, X_test, y_train, y_test = train_test_split(target_data[target.features], target_data[target.label], test_size=test_size, random_state=random_state)
                model_target["X_train"] = X_train
                model_target["X_test"] = X_test
                model_target["y_train"] = y_train
                model_target["y_test"] = y_test
                logger.info(f"Performed splitting for {target.name}\nTraining set: {len(X_train)} samples\nTesting set: {len(X_test)} samples")
        except Exception as e:
            self.cancel(reason=f"Error in _prepare_data: {e}")

    def _fit_model(self, model_config: dict[dict]):
        """This function is used to fit the model."""
        try:
            id = model_config.get("id", None)
            model = model_config.get("model")
            for target in model_config.get("targets", {}):
                target = self.targets[target] if target in self.targets.keys() else None
                if target is None:
                    logger.info(f"No target provided for model {model}.")
                    return

                logger.info(f"Fitting {target.name} for {model}.")
                target = self.models[id][target.name]
                if model == 'LazyRegressor':
                    self.models[id]['model'].fit(
                        X_train=target['X_train'],
                        y_train=target['y_train'],
                        X_test=target['X_test'],
                        y_test=target['y_test']
                        )
                else:
                    self.models[id]['model'].fit(
                        X=target['X_train'],
                        y=target['y_train']
                        )

        except Exception as e:
            self.cancel(reason=f"Error in _fit_model: {e}")

    def _prepare_validation(self, model_config: dict[dict]):
        """This function is used to validate the model."""
        try:
            id = model_config.get("id", None)
            validation = model_config.get("validation", {})
            for target in validation.get("targets", {}):
                target = self.targets[target] if target in self.targets.keys() else None
                if target is None:
                    logger.info(f"No validation target provided for model {id}.")
                    return
                logger.info(f"Prepare validation data {target.name} for {id}.")

                model = self.models[id]

                model_validation = add_category(self.models[id], "validation")
                model_validation = add_category(model_validation, target.name)

                target_data = target.update_data()

                model_validation["X_test"] = target_data[model["features"]] if model.get("features", None) else target_data[target.features]
                model_validation["y_test"] = target_data[target.label]
        except Exception as e:
            self.cancel(reason=f"Error in _prepare_validation: {e}")

    def _predict_model(self, model_config: dict[dict]):
        """This function is used to predict the model."""
        try:
            id = model_config.get("id", None)
            validation = model_config.get("validation", {})
            for target in validation.get("targets", {}):
                target = self.targets[target] if target in self.targets.keys() else None
                if not target:
                    logger.info(f"No validation target provided for model {id}.")
                    return
                logger.info(f"Predicting {target.name} for {id}.")
                model = self.models[id]["model"]
                model_target = self.models[id]["validation"][target.name]
                if hasattr(model, "predict"):
                    model_target["y_pred"] = model.predict(model_target["X_test"])
                else:
                    logger.info(f"No predict method found for model {id}.")
        except Exception as e:
            self.cancel(reason=f"Error in _predict_model: {e}")

    def r2_score(self, model_config: dict[dict]):
        """This function is used to calculate the R2 score."""
        try:
            id = model_config.get("id")
            logger.info(f"Calculating R2 score for model {id}.")
            for target in model_config.get("validation", {}).get("targets", []):
                target = self.targets[target] if target in self.targets.keys() else None
                if not target:
                    logger.info(f"No validation target provided for model {id}.")
                    return
                model_target = self.models[id]["validation"][target.name]
                self.models[id]["validation"][target.name]["r2_score"] = r2_score(model_target["y_test"], model_target["y_pred"])
                logger.info(f"R2 score for model {id} with validation target {target.name} is {self.models[id]['validation'][target.name]['r2_score']}.")
        except Exception as e:
            self.cancel(reason=f"Error in r2_score: {e}")

    def mae(self, model_config: dict[dict]):
        """This function is used to calculate the mean absolute error."""
        try:
            id = model_config.get("id")
            logger.info(f"Calculating mean absolute error for model {id}.")
            for target in model_config.get("validation", {}).get("targets", []):
                target = self.targets[target] if target in self.targets.keys() else None
                if not target:
                    logger.info(f"No validation target provided for model {id}.")
                    return
                model_target = self.models[id]["validation"][target.name]
                self.models[id]["validation"][target.name]["mae"] = mean_absolute_error(model_target["y_test"], model_target["y_pred"])
                logger.info(f"Mean absolute error for model {id} with validation target {target.name} is {self.models[id]['validation'][target.name]['mae']}.")
        except Exception as e:
            self.cancel(reason=f"Error in mae: {e}")

    def _validate_model(self, model_config: dict[dict]):
        """This function is used to validate the model."""
        try:
            id = model_config.get("id")
            validation = model_config.get("validation", {})
            for method in validation.get("methods", []):
                logger.info(f"Validating model {id} with method {method}.")
                method = getattr(self, method)
                method(model_config)
        except Exception as e:
            self.cancel(reason=f"Error in _validate_model: {e}")

    def _save_model(self, model_object: dict):
        """This function is used to save the model."""
        try:
            id = model_object.get("id")
            save = model_object.get("save", {})
            model = save.get("model", None)
            features = save.get("features", None)
            for target in model_object.get("targets", []):
                model_object = self.models[id]
                target = self.targets[target] if target in self.targets.keys() else None
                if not target:
                    logger.info(f"No target provided for model {id}.")
                    return
                if model:
                    logger.info(f"Saving model to {model}.")
                    model_path = Path(model).absolute()
                    with open(model_path, 'wb') as f:
                        joblib.dump(model_object['model'], f)
                if features:
                    logger.info(f"Saving features to {features}.")
                    feature_path = Path(features).absolute()
                    features = model_object['features'] if model_object.get("features", None) else self.targets[target.name].features
                    with open(feature_path, 'wb') as f:
                        joblib.dump(features, f)
        except Exception as e:
            self.cancel(reason=f"Error in _save_model: {e}")

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
                self._validate_model(model_config)
                self._save_model(model_config)
                #self._log_results()
        except Exception as e:
            self.cancel(reason=f"Error in run: {e}")

    def cancel(self, reason):
        logger.info(f"Cancel of model_optimization due to {reason}")
        exit(1)

@emergency_logger
class GridSearch():
    def __init__(self, targets: dict, pipeline: list[dict] = None):
        try:
            logger.info("Initializing GridSearch")
            
            self.models = {}
            self.targets = targets
            self.pipeline = pipeline if pipeline else {}

        except Exception as e:
            self.cancel(reason=f"Error in __init__: {e}")

    def run(self):
        """This function is used to run the Grid Search pipeline."""
        try:
            pass
        except Exception as e:
            self.cancel(reason=f"Error in run: {e}")

    def cancel(self, reason):
        logger.info(f"Cancel of GridSearch due to {reason}")
        exit(1)