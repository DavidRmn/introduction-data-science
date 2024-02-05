"""Configuration module for idstools package."""
import yaml
from box import Box
from pathlib import Path
from dynaconf import Dynaconf
from IPython.display import display, Markdown

config_root = Path(__file__).parent.parent.parent / "config"

def to_plain_dict(obj):
    """Recursively convert Box objects and other iterables to plain dictionaries."""
    if isinstance(obj, Box):
        # Convert Box to dict and recursively process its items
        return {k: to_plain_dict(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        # Recursively process dictionary items
        return {k: to_plain_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Recursively process list or tuple elements
        return [to_plain_dict(v) for v in obj]
    else:
        # Return the object itself if it's not a Box, dict, list, or tuple
        return obj

def pprint_dynaconf(dynaconf, notebook=False):
    """Convert Dynaconf or Box objects to plain dict and dump as YAML."""
    try:
        settings_dict = dynaconf.as_dict()
    except AttributeError:
        settings_dict = to_plain_dict(dynaconf)

    formatted_yaml = yaml.safe_dump(settings_dict, default_flow_style=False, sort_keys=False)        

    if notebook:
        display(Markdown(f"```yaml\n{formatted_yaml}\n```"))
    else:
        return formatted_yaml
    
class PrettyDynaconf(Dynaconf):
    """A Dynaconf subclass that pretty-prints its configuration."""
    def _repr_markdown_(self):
        pprint_dynaconf(self, notebook=True)   

def create_config(config_file: str = None):
    """
    Initializes and returns a PrettyDynaconf object with specified settings.

    Parameters:
    - config_root: The root path where the configuration files are located.

    Returns:
    - A configured PrettyDynaconf object.
    """
    if config_file is None:
        config_file = str(config_root / "idstools" / "config.yml")
    else:
        config_file = str(config_file)

    _idstools = PrettyDynaconf(
        envvar_prefix='IDSTOOLS',
        settings_files=[
            config_file,
        ]
    )
    return _idstools

_template = PrettyDynaconf(
        envvar_prefix='IDSTOOLS',
        settings_files=[
            str(config_root / "idstools" / "template.yml"),
        ]
    )

_logging = PrettyDynaconf(
    envvar_prefix='LOGGING',
    root_path=str(config_root),
    settings_files=[
        'logging/config.yml',
        ]
)