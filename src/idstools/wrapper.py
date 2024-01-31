import importlib
from idstools._config import idstools_config

class Wrapper:
    def __init__(self):
        self.modules = self.__instantiate_modules()

    def __instantiate_modules(self):
        instantiated_modules = {}
        for module_name, module_info in idstools_config.modules.items():
            if module_info.class_name:
                module_path = f"idstools.{module_name}"
                module = importlib.import_module(module_path)
                class_name = getattr(module, module_info.class_name)
                instantiated_modules[module_name] = class_name()
        return instantiated_modules

    def run(self):
        for module in self.modules.values():
            module.run()