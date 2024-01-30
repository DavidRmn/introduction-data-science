"""Configuration module for idstools package."""
from pathlib import Path
from dynaconf import Dynaconf

project_root = Path(__file__).parent.parent.parent

settings = Dynaconf(
    envvar_prefix="IDSTOOLS",
    settings_files=[
        f'{str(project_root)}/config/idstools/config.yml',
        f'{str(project_root)}/config/logging/config.yaml',
        f'{str(project_root)}/config/.secrets.yml'
        ],
    environments=[
        "default",
        "development"
        ],
    env_switcher="IDSTOOLS_ENV",
)
