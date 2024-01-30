import yaml
from pathlib import Path
#TODO: use settings from idstools._config
from idstools._config import settings
import idstools.data_explorer as idsde
import idstools.data_preparation as idsdp

def read_yaml(file: Path):
    with open(file, 'r') as f:
        content = yaml.load(f, Loader=yaml.SafeLoader)
    return content

class wrapper():
    def __init__(self, config_file: Path):
        self.config = read_yaml(file=config_file)

    def __instantiate_modules(self):
        self.data_explorer = idsde.data_explorer(self.config['default']["data_explorer"])
        self.data_preparation = idsdp.data_preparation(self.config['default']["data_preparation"])

    def run(self):
        self.__instantiate_modules()
        self.data_explorer.run()
        self.data_preparation.run()