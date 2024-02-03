import importlib
from tqdm import tqdm
from idstools._config import _idstools
from idstools._helpers import setup_logging

logger = setup_logging(__name__)

class Wrapper:
    def __init__(self):
        self.environments = self.__instantiate_modules()

    def __instantiate_modules(self):
        environments = {}
        logger.info("Instantiating environments from configuration.")
        for env_name, config in _idstools.to_dict().items():
            logger.debug(f"Processing environment: {env_name}")
            module_classes = {}
            for module_name, module_config in config.items():
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
            logger.info(f"Executing environment: {env_name}")
            for module_name, (class_name, class_config) in tqdm(modules.items(), desc=f"Modules in {env_name}"):
                try:
                    self.initialize_and_run_module(module_name, class_name, class_config)
                except Exception as e:
                    logger.error(f"Error executing module {module_name} in environment {env_name}: {e}")
        logger.info("Finished execution of all environments.")

    def initialize_and_run_module(self, module_name, class_name, class_config):
        try:
            logger.debug(f"Loading module {module_name} for class {class_name}")
            module_path = f"idstools.{module_name.lower()}"
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)

            if hasattr(cls, 'run'):
                logger.info(f"Instantiating and executing {class_name} in module {module_name}")
                instance = cls(**class_config)
                instance.run()
            else:
                logger.warning(f"Class {class_name} in module {module_name} does not have a 'run' method")
        except Exception as e:
            logger.error(f"Exception during execution of module {module_name}: {e}")