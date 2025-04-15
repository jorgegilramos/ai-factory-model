from .logger import logger


# Shortcut to logger functions
def info(message, *args, **kwargs):
    logger.info(message, *args, **kwargs)


def debug(message, *args, **kwargs):
    logger.debug(message, *args, **kwargs)


def error(message, *args, **kwargs):
    logger.error(message, *args, **kwargs)


def warning(message, *args, **kwargs):
    logger.warning(message, *args, **kwargs)


def critical(message, *args, **kwargs):
    logger.critical(message, *args, **kwargs)
