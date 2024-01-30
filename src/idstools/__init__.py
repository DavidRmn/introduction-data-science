from pathlib import Path

from idstools.wrapper import wrapper

config_path=Path(__file__).parent.parent.parent / 'config' / 'idstools' / 'config.yml'

def main():
    """Entry point for the application script"""
    w = wrapper(config_file=config_path)
    w.run()