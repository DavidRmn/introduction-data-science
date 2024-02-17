import shutil
from pathlib import Path
from argparse import ArgumentParser
from idstools.wrapper import Wrapper
from idstools._config import load_config

def main():
    """
    Entry point for the application script
    
    This function is the entry point for the application script.\n
    It parses the command line arguments and runs the pipeline.

    Command line arguments:
        --config: Path to the configuration file.\n
        Default: idstools/config.yaml

        --clear-results: Clear the results directory before running the pipeline.\n
        Default: False

    Example:
        python -m idstools --config idstools/config.yaml --clear-results True
    """
    parser = ArgumentParser(description="IDSTools: A collection of tools for data science.")
    parser.add_argument(
        "--config",
        help="Path to the configuration file.",
        default="idstools/config.yaml",
    )
    parser.add_argument(
        "--clear-results",
        help="(True/False) Wether to clear the results directory before running the pipeline.",
        default=False,
    )
    args = parser.parse_args()
    config = load_config(args.config)
    result_path = Path("results").resolve()
    if result_path.exists() and args.clear_results:
        shutil.rmtree(result_path)
    w = Wrapper(config=config)
    w.run()