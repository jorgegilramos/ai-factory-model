from src.ai_factory_model.logger import info, debug, error, warning, critical
from src.ai_factory_model.logger.logger import AppLogger


def test_logger_init():
    logger1 = AppLogger(__name__)
    # Avoid log double initialization
    logger1.__init__(__name__)
    AppLogger.__init__(logger1, __name__)


def test_logger_functions():
    message = "Log test message"
    info(message)
    debug(message)
    error(message)
    warning(message)
    critical(message)
