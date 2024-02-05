import importlib
from tqdm import tqdm
from idstools._config import PrettyDynaconf
from idstools._helpers import setup_logging, result_logger

logger = setup_logging(__name__)

class Wrapper:
    def __init__(self, config: PrettyDynaconf):
        self.config = config
        self.current_target_data = None
        self.environments = self.__instantiate_modules()

    def __instantiate_modules(self):
        environments = {}
        logger.info("Instantiating environments from configuration.")
        for env_name, env_config in self.config.to_dict().items():
            logger.debug(f"Processing environment: {env_name}")
            module_classes = {}
            if env_config:
                for module_name, module_config in env_config.items():
                    try:
                        class_name = next(iter(module_config))
                        module_classes[module_name] = (class_name, module_config[class_name])
                        logger.debug(f"Configured {class_name} for module {module_name} in environment {env_name}")
                    except Exception as e:
                        logger.error(f"Error processing module {module_name} in environment {env_name}: {e}")
            environments[env_name] = module_classes
        logger.info(f"Completed instantiation of environments: {list(environments.keys())}")
        return environments

    def run(self):
        logger.info("Starting execution of environments.")
        for env_name, modules in tqdm(self.environments.items(), desc="Environments"):
            self.current_target_data = None  # Reset for each environment
            logger.info(f"Executing environment: {env_name}")
            for module_name, (class_name, class_config) in tqdm(modules.items(), desc=f"Modules in {env_name}"):
                try:
                    logger.info(f"Processing environment: {env_name}")
                    # Check for TargetData configuration
                    if class_name == "TargetData":
                        self.current_target_data = self.initialize_and_run_module(module_name, class_name, class_config, env_name=env_name, is_target_data=True)
                        continue  # Skip further processing in this iteration
                    # For other modules, pass the current TargetData instance if available
                    self.initialize_and_run_module(module_name, class_name, class_config, target_data=self.current_target_data)
                except Exception as e:
                    logger.error(f"Error executing module {module_name} in environment {env_name}: {e}")
        logger.info("Finished execution of all environments.")

    def initialize_and_run_module(self, module_name, class_name, class_config, env_name=None, target_data=None, is_target_data=False):
        try:
            logger.debug(f"Loading module {module_name} for class {class_name}")
            module_path = f"idstools.{module_name.lower()}"
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)

            # If this is the TargetData module or another module requires the current TargetData instance
            if is_target_data or target_data:
                if is_target_data:
                    instance = cls(**class_config, env_name=env_name)
                else:
                    # Modify class_config to use the existing TargetData instance
                    # Assuming other modules can accept a TargetData instance through some parameter
                    modified_config = {**class_config, 'target_data': target_data}
                    instance = cls(**modified_config, env_name=env_name)
            else:
                instance = cls(**class_config, env_name=env_name)

            if hasattr(cls, 'run'):
                logger.info(f"Instantiating and executing {class_name} in module {module_name}")
                instance.run()
                if is_target_data:
                    return instance  # Return the TargetData instance for reuse
            else:
                logger.warning(f"Class {class_name} in module {module_name} does not have a 'run' method")
        except Exception as e:
            logger.error(f"Exception during execution of module {module_name}: {e}")