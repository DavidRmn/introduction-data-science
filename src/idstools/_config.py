"""Configuration module for idstools package."""
import yaml
from box import Box
from pathlib import Path
from dynaconf import Dynaconf
from IPython.display import display, Markdown

config_root = Path(__file__).parent.parent.parent / "config"

def to_plain_dict(obj):
    """
    Recursively convert Box objects and other iterables to plain dictionaries.
    
    Args:
        obj: The object to convert to a plain dictionary.
    
    Returns:
        dict: The plain dictionary representation of the object.
    """
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
    """
    Convert Dynaconf or Box objects to plain dict and dump as YAML.
    
    Args:
        dynaconf (Dynaconf): The Dynaconf object to pretty-print.
        notebook (bool): Whether to display the output in a Jupyter notebook.
    
    Returns:
        str: The formatted YAML string if notebook is False.
    """
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
    """
    A Dynaconf subclass that pretty-prints its configuration.
    
    Args:
        Dynaconf: The Dynaconf class to subclass.

    Returns:
        PrettyDynaconf: A PrettyDynaconf object with the specified settings.
    """
    def _repr_markdown_(self):
        """Pretty-print the configuration as Markdown."""
        pprint_dynaconf(self, notebook=True)   

def load_config(config_file: str = None):
    """
    Initializes and returns a PrettyDynaconf object with specified settings.

    Args:
        config_file (str): The path to the configuration file.

    Returns:
        PrettyDynaconf: A PrettyDynaconf object with the specified settings.
    """
    if config_file is None:
        config_file = str(config_root / "idstools" / "config.yaml")
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
        str(config_root / "idstools" / "template.yaml"),
    ]
    )

_logging = PrettyDynaconf(
    envvar_prefix='LOGGING',
    root_path=str(config_root),
    settings_files=[
        'logging/config.yaml',
        ]
)