import importlib
from idstools._config import _idstools
from idstools._helpers import setup_logging

logger = setup_logging(__name__)

class Wrapper:
    def __init__(self):
        self.module_classes = self.__instantiate_modules()

    def __instantiate_modules(self):
        module_classes = {}
        for config_name, config in _idstools.default.items():
            logger.info(f"Found config: {config_name}")
            for module_name, module_config in config.items():
                try:
                    class_name = next(iter(module_config))
                    logger.debug(f"Found module: {module_name} with class: {class_name}")
                    module_classes[module_name] = (class_name, module_config[class_name])
                except Exception as e:
                    logger.error(f"Error instantiating module {module_name}: {e}")
        return module_classes

    def run(self):
        for module_name, (class_name, class_config) in self.module_classes.items():
            try:
                module = importlib.import_module(f"idstools.{module_name}")
                cls = getattr(module, class_name)
                logger.debug(f"Running module: {module_name} with class: {class_name}")
                logger.debug(f"Class config: {class_config.to_dict()}")
                instance = cls(**class_config.to_dict())
                instance.run()
            except ImportError:
                logger.error(f"Module {module_name} not found")
            except AttributeError:
                logger.error(f"Class {class_name} not found in module {module_name}")
            except Exception as e:
                logger.error(f"Error running module {module_name}: {e}")
