"""Configuration module for idstools package."""
from pathlib import Path
from dynaconf import Dynaconf

config_root = Path(__file__).parent.parent.parent / "config"

logging_config = Dynaconf(
    envvar_prefix="LOGGING",
    root_path=str(config_root),
    settings_files=[
        'logging/config.yml',
        ]
)

idstools_config = Dynaconf(
    envvar_prefix="IDSTOOLS",
    root_path=str(config_root),
    settings_files=[
        'idstools/config.yml',
        ]
)
