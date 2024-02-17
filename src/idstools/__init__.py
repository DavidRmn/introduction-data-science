import shutil
from pathlib import Path
from argparse import ArgumentParser
from idstools.wrapper import Wrapper
from idstools._config import load_config

def main():
    """Entry point for the application script"""
    parser = ArgumentParser(description="IDSTools: A collection of tools for data science.")
    parser.add_argument(
        "--config",
        help="Path to the configuration file.",
        default="idstools/config.yaml"
    )
    parser.add_argument(
        "--clear-results",
        help="Clear the results directory before running the pipeline.",
        default=False,
    )
    args = parser.parse_args()
    config = load_config(args.config)
    result_path = Path("results").resolve()
    if result_path.exists() and args.clear_results:
        shutil.rmtree(result_path)
    w = Wrapper(config=config)
    w.run()