import idstools._helpers as helpers

logger = helpers.setup_logging(__name__)

@helpers.emergency_logger
class ModelOptimization():
    """This class is used to optimize the model."""
    def __init__(self, **kwargs):
        logger.info("Initializing ModelOptimization")
        logger.debug(f"{kwargs}")
        pass

    def console_output(self):
        pass

    def run(self):
        pass