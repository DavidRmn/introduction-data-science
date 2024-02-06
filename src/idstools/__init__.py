from argparse import ArgumentParser
from idstools.wrapper import Wrapper
from idstools._config import load_config

def main():
    """Entry point for the application script"""
    parser = ArgumentParser(description="IDSTools: A collection of tools for data science.")
    parser.add_argument(
        "--config",
        help="Path to the configuration file.",
        default="idstools/config.yml"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    w = Wrapper(config=config)
    w.run()