"""Configuration module for idstools package."""
from pathlib import Path
from dynaconf import Dynaconf

config_root = Path(__file__).parent.parent.parent / "config"

settings = Dynaconf(
    envvar_prefix="IDSTOOLS",
    settings_files=[
        f'{str(config_root)}/idstools/config.yml',
        f'{str(config_root)}/logging/config.yml',
        f'{str(config_root)}/.secrets.yml'
        ],
    environments=[
        "default",
        "development"
        ],
    env_switcher="IDSTOOLS_ENV",
)
