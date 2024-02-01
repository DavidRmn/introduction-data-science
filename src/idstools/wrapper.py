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
        logger.debug(f"Found environments: {_idstools.to_dict()}")

        for env_name, config in _idstools.to_dict().items():
            logger.debug(f"Found environment: {env_name}.")
            module_classes = {}

            for module_name, module_config in config.items():
                logger.debug(f"Found module: {module_name} with config: {module_config}")

                try:
                    class_name = next(iter(module_config))
                    module_classes[module_name] = (class_name, module_config[class_name])
                    logger.debug(f"Instantiated module: {module_name} with class: {class_name}")
                except KeyError as e:
                    logger.error(f"Error in module configuration {module_name}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error instantiating module {module_name}: {e}")
                    raise

            environments[env_name] = module_classes

        return environments


    def run(self):
        logger.debug(f"Running environments: {self.environments}")
        for env_name, modules in tqdm(self.environments.items()):
            logger.debug(f"Running environment: {env_name}")
            for module_name, (class_name, class_config) in tqdm(modules.items()):
                logger.debug(f"Running module: {module_name} with class: {class_name} and config: {class_config}")
                
                try:
                    module_path = f"idstools.{module_name.lower()}"
                    module = importlib.import_module(module_path)
                    cls = getattr(module, class_name)
                    
                    if hasattr(cls, 'run'):
                        instance = cls(**class_config)
                        instance.run()
                    else:
                        logger.error(f"No 'run' method found in class {class_name} of module {module_path}")

                except ImportError:
                    logger.error(f"Module not found: {module_name}")
                except AttributeError:
                    logger.error(f"Class not found: {class_name} in module {module_name}")
                except Exception as e:
                    logger.error(f"Error running module {module_name}: {e}")

