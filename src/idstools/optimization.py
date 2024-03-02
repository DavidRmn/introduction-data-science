import joblib
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from importlib import import_module
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from idstools._helpers import emergency_logger, setup_logging, add_category

logger = setup_logging(__name__)

@emergency_logger
class ModelOptimization():
    """This class is used to optimize the model."""
    def __init__(self, targets: list, pipeline: list[dict] = None):
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

    def _add_estimator_config(self, config: dict, estimator: str):
        """This function is used to add the estimator config."""
        if not config:
            config = {}
        logger.info(f"Adding estimator config for {estimator}.")
        model = self.models[estimator]['model']
        param_grid = self.models[estimator]['param_grid']
        config["estimator"] = model
        config["param_grid"] = param_grid
        return config

    def _retrieve_model(self, model_config: dict[dict]):
        """This function is used to retrieve the model."""
        try:
            id = model_config.get("id")
            model = model_config.get("model")
            module = model_config.get("module", None)
            load = model_config.get("load", None)
            param_grid = model_config.get("param_grid", None)
            estimators = model_config.get("estimators", None)
            config = model_config.get("config", {})
            if load:
                self._load_model(model_config)
                return
            elif param_grid:
                logger.info(f"Retrieving model {model} with param grid {param_grid}.")
                model_target = add_category(self.models, id)
                model = self._retrieve_function(model, module)
                model_target['model'] = model()
                model_target['config'] = config
                model_target['param_grid'] = param_grid
                return
            elif estimators:
                for estimator in estimators:
                    logger.info(f"Retrieving model {estimator}.")
                    model_target = add_category(self.models,id)
                    model_target = add_category(model_target, estimator)
                    model = self._retrieve_function(model, module)
                    config = self._add_estimator_config(config, estimator)
                    model_target['model'] = model(**config)
                    model_target['config'] = config
                return
            elif model:
                logger.info(f"Retrieving model {model}.")
                model_target = add_category(self.models, id)
                model = self._retrieve_function(model, module)
                model_target['model'] = model(**config)
                model_target['config'] = config
                return
            else:
                logger.info(f"No model provided for {id}.")
                return

        except Exception as e:
            self.cancel(reason=f"Error in _retrieve_model: {e}")

    def _prepare_data(self, model_config: dict[dict]):
        """This function is used to prepare the data for the model."""
        try:
            id = model_config.get("id")
            model = model_config.get("model")
            split = model_config.get("split", None)
            for target in model_config.get("targets", []):
                target = next((t for t in self.targets if target == t.id), None)
                if target is None:
                    logger.info(f"No target provided for model {model}.")
                    return
                logger.info(f"Preparing target for {model}.")
                model_target = add_category(self.models[id], target.filename)

                target_data = target.update_data()

                test_size = split.get("config").get("test_size", 0.2)
                random_state = split.get("config").get("random_state", 42)

                X_train, X_test, y_train, y_test = train_test_split(target_data[target.features], target_data[target.label], test_size=test_size, random_state=random_state)
                model_target["X_train"] = X_train
                model_target["X_test"] = X_test
                model_target["y_train"] = y_train
                model_target["y_test"] = y_test
                logger.info(f"Performed splitting for {target.filename}\nTraining set: {len(X_train)} samples\nTesting set: {len(X_test)} samples")
        except Exception as e:
            self.cancel(reason=f"Error in _prepare_data: {e}")

    def _fit_model(self, model_config: dict[dict]):
        """This function is used to fit the model."""
        try:
            id = model_config.get("id", None)
            model = model_config.get("model")
            validation = model_config.get("validation", {})
            estimators = model_config.get("estimators", None)
            for target in model_config.get("targets", {}):
                target = next((t for t in self.targets if target == t.id), None)
                if target is None:
                    logger.info(f"No target provided for model {model}.")
                    return

                logger.info(f"Fitting {target.filename} for {model}.")
                results = add_category(target.analysis_results, id)
                target_data = self.models[id][target.filename]
                validation_result = add_category(self.models[id], "validation")
                validation_result = add_category(validation_result, target.filename)
                if model == 'LazyRegressor':
                    for validation_file in validation.get("targets", {}):
                        validation_file = next((t for t in self.targets if validation_file == t.id), None)
                        if not validation_file:
                            logger.info(f"No validation target provided for model {id}.")
                            return
                        validation_file.update_data()
                        models, predictions = self.models[id]['model'].fit(
                            X_train=target_data['X_train'],
                            y_train=target_data['y_train'],
                            X_test=self.models[id]['validation'][validation_file.filename]['X_test'],
                            y_test=self.models[id]['validation'][validation_file.filename]['y_test']
                            )
                        results["LazyRegressor"] = [models]
                        for model in predictions:
                            results[f"LazyRegressor_{model}"] = predictions[model]
                            validation_file.figures[f"LazyRegressor_{model}_actual_vs_predicted"] = self.actual_vs_predicted(
                                self.models[id]['validation'][validation_file.filename]['y_test'],
                                predictions[model], model=f"LazyRegressor_{model}"
                                )
                elif estimators:
                    for estimator in estimators:
                        result = self.models[id][estimator]['model'].fit(
                            X=target_data['X_train'],
                            y=target_data['y_train']
                            )
                        results[f'{estimator}_best_estimator'] = result.best_estimator_
                        results[f'{estimator}_best_params'] = result.best_params_
                        results[f'{estimator}_best_score'] = result.best_score_
                else:
                    self.models[id]['model'].fit(
                        X=target_data['X_train'],
                        y=target_data['y_train']
                        )

        except Exception as e:
            self.cancel(reason=f"Error in _fit_model: {e}")

    def _prepare_validation(self, model_config: dict[dict]):
        """This function is used to validate the model."""
        try:
            id = model_config.get("id", None)
            validation = model_config.get("validation", {})
            for target in validation.get("targets", {}):
                target = next((t for t in self.targets if target == t.id), None)
                if target is None:
                    logger.info(f"No validation target provided for model {id}.")
                    return
                logger.info(f"Prepare validation data {target.filename} for {id}.")

                model = self.models[id]

                model_validation = add_category(self.models[id], "validation")
                model_validation = add_category(model_validation, target.filename)

                target_data = target.update_data()

                model_validation["X_test"] = target_data[model["features"]] if model.get("features", None) else target_data[target.features]
                model_validation["y_test"] = target_data[target.label]
        except Exception as e:
            self.cancel(reason=f"Error in _prepare_validation: {e}")

    def _predict_model(self, model_config: dict[dict]):
        """This function is used to predict the model."""
        try:
            id = model_config.get("id", None)
            estimators = model_config.get("estimators", None)
            validation = model_config.get("validation", {})
            for target in validation.get("targets", {}):
                target = next((t for t in self.targets if target == t.id), None)
                if not target:
                    logger.info(f"No validation target provided for model {id}.")
                    return
                logger.info(f"Predicting target {target.filename} for {id}.")
                if estimators:
                    for estimator in estimators:
                        model = self.models[id][estimator]["model"]
                        model_result = add_category(self.models[id][estimator], target.filename)
                        model_target = self.models[id]["validation"][target.filename]
                        if hasattr(model, "predict"):
                            model_result["y_pred"] = model.predict(model_target["X_test"])
                            for model in estimators:
                                results = add_category(target.analysis_results, id)
                                results[f"{estimator}_{model}"] = model_result["y_pred"]
                                target.figures[f"GS_{estimator}_actual_vs_predicted"] = self.actual_vs_predicted(
                                    model_target["y_test"],
                                    model_result["y_pred"],
                                    model=f"GS_{estimator}"
                                    )
                        else:
                            logger.info(f"No predict method found for model {id}.")
                else:
                    model = self.models[id]["model"]
                    model_target = self.models[id]["validation"][target.filename]
                    if hasattr(model, "predict"):
                        model_target["y_pred"] = model.predict(model_target["X_test"])
                        results = add_category(target.analysis_results, id)
                        results["y_pred"] = model_target["y_pred"]
                        target.figures[f"{id}_actual_vs_predicted"] = self.actual_vs_predicted(
                            model_target["y_test"],
                            model_target["y_pred"],
                            model=id
                            )
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
                target = next((t for t in self.targets if target == t.id), None)
                results = add_category(target.analysis_results, id)
                if not target:
                    logger.info(f"No validation target provided for model {id}.")
                    return
                model_target = self.models[id]["validation"][target.filename]
                self.models[id]["validation"][target.filename]["r2_score"] = r2_score(model_target["y_test"], model_target["y_pred"])
                results["r2_score"] = self.models[id]["validation"][target.filename]["r2_score"]
                logger.info(f"R2 score for model {id} with validation target {target.filename} is {self.models[id]['validation'][target.filename]['r2_score']}.")
                # workarround
                print(f"R2 score for model {id} with validation target {target.filename} is {self.models[id]['validation'][target.filename]['r2_score']}.")
        except Exception as e:
            self.cancel(reason=f"Error in r2_score: {e}")

    def mae(self, model_config: dict[dict]):
        """This function is used to calculate the mean absolute error."""
        try:
            id = model_config.get("id")
            logger.info(f"Calculating mean absolute error for model {id}.")
            for target in model_config.get("validation", {}).get("targets", []):
                target = next((t for t in self.targets if target == t.id), None)
                results = add_category(target.analysis_results, id)
                if not target:
                    logger.info(f"No validation target provided for model {id}.")
                    return
                model_target = self.models[id]["validation"][target.filename]
                self.models[id]["validation"][target.filename]["mae"] = mean_absolute_error(model_target["y_test"], model_target["y_pred"])
                results["mae"] = self.models[id]["validation"][target.filename]["mae"]
                logger.info(f"Mean absolute error for model {id} with validation target {target.filename} is {self.models[id]['validation'][target.filename]['mae']}.")
                # workarround 
                print(f"Mean absolute error for model {id} with validation target {target.filename} is {self.models[id]['validation'][target.filename]['mae']}.")
        except Exception as e:
            self.cancel(reason=f"Error in mae: {e}")

    def actual_vs_predicted(self, y_test, y_pred, model: str = ""):
        data = {"Actual": y_test, "Predicted": y_pred}
        df = pd.DataFrame(data)
        sns.set_theme(style="whitegrid")
        plot = plt.figure(figsize=(10, 6))

        sns.lineplot(data=df, markers=False)

        plt.title(f"{model} Actual vs. Predicted Values")
        plt.xlabel("Data Points")
        plt.ylabel("Values")
        return plot

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
                target = next((t for t in self.targets if target == t.id), None)
                if not target:
                    logger.info(f"No target provided for model {id}.")
                    return
                if model:
                    logger.info(f"Saving model to {model}.")
                    model_path = Path(model).absolute()
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(model_path, 'wb') as f:
                        joblib.dump(model_object['model'], f)
                if features:
                    logger.info(f"Saving features to {features}.")
                    feature_path = Path(features).absolute()
                    feature_path.parent.mkdir(parents=True, exist_ok=True)
                    features = model_object['features'] if model_object.get("features", None) else target.features
                    with open(feature_path, 'wb') as f:
                        joblib.dump(features, f)
        except Exception as e:
            self.cancel(reason=f"Error in _save_model: {e}")

    def _log_results(self, model_config: dict) -> None:
        """
        This function is used to log the results of the model optimization.
        """
        try:
            id = model_config.get("id")
            for target in model_config.get("targets", []):
                target = next((t for t in self.targets if target == t.id), None)
                if not target:
                    logger.info(f"No target provided for model {model_config.get('id')}.")
                    return
                result_logger = setup_logging("optimization_results", env_name=target.env_name, step_name=target.step_name, filename=f"ModelOptimization_{target.filename}")
                result_logger.info(f"Logging results for target {target.filename} in {target.env_name}:{target.step_name}.")
                result_logger.info(f"Results for model {id}:")

                for result_category, results in target.analysis_results[id].items():
                    result_logger.info(f"Results for {result_category}:")
                    if type(results) == dict:
                        result_logger.info(f"{results}:\n")
                        for sub_result, sub_value in results.items():
                            result_logger.info(f"  {sub_result}: {sub_value}")
                    else:
                        result_logger.info(f"{result_category}:\n{results}")
        except Exception as e:
            logger.error(f"Error in _log_results: {e}")
    def run(self):
        """This function is used to run the model optimization pipeline."""
        try:
            logger.info("Running model optimization pipeline.")
            for model_config in self.pipeline:
                self._retrieve_model(model_config)
                self._prepare_data(model_config)
                self._prepare_validation(model_config)
                self._fit_model(model_config)
                self._predict_model(model_config)
                self._validate_model(model_config)
                self._save_model(model_config)
                self._log_results(model_config)
        except Exception as e:
            self.cancel(reason=f"Error in run: {e}")

    def cancel(self, reason):
        logger.info(f"Cancel of model_optimization due to {reason}")
        exit(1)