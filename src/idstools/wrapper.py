import yaml
from pathlib import Path

import idstools.data_explorer as idsde

def read_yaml(file: Path):
    with open(file, 'r') as file:
        content = yaml.load(file, Loader=yaml.SafeLoader)
    return content

class wrapper():
    def __init__(self, config_file: Path):
        self.config = read_yaml(file=config_file)

    def __instantiate_modules(self):
        self.data_explorer = idsde.data_explorer(self.config["data_explorer"])

    def run(self):
        self.__instantiate_modules()
        self.data_explorer.run()