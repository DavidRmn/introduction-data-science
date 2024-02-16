import importlib
from tqdm import tqdm
from idstools._config import PrettyDynaconf
from idstools._helpers import setup_logging

logger = setup_logging(__name__)

class Wrapper:
    """
    Wrapper class for executing multiple environments and modules.
    
    Args:
        config (PrettyDynaconf): Configuration object.

    Attributes:
        config (PrettyDynaconf): Configuration object.
        current_target_data (None): Placeholder for current TargetData instance.
        environments (dict): Dictionary of environments and their steps.
    
    Methods:
        _prepare_classes: Prepare modules from configuration.
        _prepare_modules: Prepare environment from configuration.
        _prepare_environments: Prepare configuration for the wrapper.
        _instantiate_and_run_class: Instantiate and run a class.
        _process_modules: Processing modules in a step.
        _process_steps: Process steps in an environment.
        run: Process all environments and their modules.
    """
    def __init__(self, config: PrettyDynaconf):
        self.config = config
        self.current_target = None
        self.environments = self._prepare_environments()

    def _prepare_classes(self, env_name, step_config) -> dict:
        """
        Prepare modules from configuration.
        
        Args:
            step_config (dict): Configuration for the step.

        Returns:
            dict: Dictionary of module names and their classes.
        """
        module_classes = {}
        for module_name, module_config in step_config.items():
            try:
                class_name = next(iter(module_config))
                module_classes[module_name] = (class_name, module_config[class_name])
                logger.debug(f"Instantiated {class_name} for module {module_name} in environment {env_name}")
            except Exception as e:
                logger.error(f"Error instantiating module {module_name} in environment {env_name}: {e}")
        return module_classes

    def _prepare_modules(self, env_name, env_config) -> dict:
        """
        Prepare environment from configuration.
        
        Args:
            env_config (dict): Configuration for the environment.

        Returns:
            dict: Dictionary of steps and their modules.
        """
        environment_steps = {}
        for step_name, step_config in env_config.items():
            try:
                environment_steps[step_name] = self._prepare_classes(env_name, step_config)
                logger.debug(f"Prepared step {step_name} in environment {env_name}")
            except Exception as e:
                logger.error(f"Error preparing step {step_name} in environment {env_name}: {e}")
        return environment_steps

    def _prepare_environments(self):
        """
        Prepare configuration for the wrapper.
        """
        environments = {}
        logger.info("Reading environments from configuration.")
        for env_name, env_config in self.config.to_dict().items():
            if not env_config:
                logger.error(f"No configuration found for environment: {env_name}")
                return
            logger.debug(f"Preparing environment: {env_name}")
            environment_steps = self._prepare_modules(env_name, env_config)
            environments[env_name] = environment_steps
        logger.info(f"Completed preparation of environments: {list(environments.keys())}")
        return environments
    
    def _instantiate_and_run_class(self, module_name, class_name, class_config, env_name=None, target=None, is_target=False):
        """
        Instantiate and run a class.
        
        Args:
            module_name (str): Name of the module.
            class_name (str): Name of the class.
            class_config (dict): Configuration for the class.
            env_name (str): Name of the environment.
            target (None): Placeholder for current TargetData instance.
            is_target (bool): Flag to indicate if the module is TargetData.
        """
        try:
            logger.debug(f"Importing module {module_name} for class {class_name}")
            module_path = f"idstools.{module_name.lower()}"
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            if is_target is False and target is None:
                logger.error(f"Error instantiating class {class_name}: No TargetData instance available.")
                return
            if is_target is False:
                instance = cls(**class_config, target=target)
                logger.debug(f"Running class {class_name}")
                instance.run()
                return
            instance = cls(**class_config, env_name=env_name)
            return instance
        except Exception as e:
            logger.error(f"Error instantiating class {class_name}: {e}")

    def _process_modules(self, modules, step_name, env_name):
        """
        Processing modules in a step.

        Args:
            modules (dict): Dictionary of modules and their classes.
            step_name (str): Name of the step.
            env_name (str): Name of the environment.
        """
        try:
            for module_name, (class_name, class_config) in tqdm(modules.items(), desc=f"Modules in {step_name}"):
                logger.info(f"Processing module: {module_name}")
                if class_name == "Target":
                    self.current_target = self._instantiate_and_run_class(module_name=module_name, class_name=class_name, class_config=class_config, env_name=env_name, is_target=True)
                else:
                    self._instantiate_and_run_class(module_name=module_name, class_name=class_name, class_config=class_config, target=self.current_target)
        except Exception as e:
            logger.error(f"Error processing module {module_name} in environment {env_name}: {e}")
            
    def _process_steps(self, env_name, steps):
        """
        Process steps in an environment.

        Args:
            env_name (str): Name of the environment.
            steps (dict): Dictionary of steps and their modules.
        """
        try:
            for step_name, modules in tqdm(steps.items(), desc=f"Steps in {env_name}"):
                logger.info(f"Processing step: {step_name}")
                self._process_modules(modules, step_name, env_name)
        except Exception as e:
            logger.error(f"Error processing step {step_name} in environment {env_name}: {e}")        
    
    def run(self):
        """
        Process all environments and their modules.
        """
        logger.info("Start processing of environments.")
        for env_name, steps in tqdm(self.environments.items(), desc="Environments"):
            logger.info(f"Processing environment: {env_name}")
            self._process_steps(env_name, steps)
        logger.info("Finished processing of all environments.")