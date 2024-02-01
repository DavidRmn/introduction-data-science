from idstools._config import pprint_dynaconf
from idstools._helpers import emergency_logger, setup_logging

logger = setup_logging(__name__)

@emergency_logger
class ModelOptimization():
    """This class is used to optimize the model."""
    def __init__(self, **kwargs):
        logger.info("Initializing ModelOptimization")
        logger.debug(f"Parameters:\n{pprint_dynaconf(kwargs)}")
        pass

    def console_output(self):
        pass

    def run(self):
        pass