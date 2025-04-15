import os
from decouple import config

_logging_dir: str = config("LOGGING_DIR", default="logs", cast=str)
# Add directory separator at the end
LOGGING_DIR = _logging_dir if _logging_dir.endswith("/") else f"{_logging_dir}{os.sep}"
LOGGING_FILE = config("LOGGING_FILE", default="ai-model-factoy.log", cast=str)
LOGGING_WHEN = config("LOGGING_WHEN", default="midnight", cast=str)
LOGGING_INTERVAL: int = int(config("LOGGING_INTERVAL", default="1", cast=str))
LOGGING_TITLE = config("LOGGING_TITLE", "ai-model-factoy")
LOGGING_LEVEL = config("LOGGING_LEVEL", "INFO")
LOGGING_HANDLERS = config("LOGGING_HANDLERS", "console,file_handler")
LOGGING_FORMATTER = config("LOGGING_FORMATTER", "%(asctime)s - [%(name)s] - %(levelname)-5s - %(message)s")

# Debugging mode
FORCE_LOG_DEBUG = config("FORCE_LOG_DEBUG", default=False, cast=bool)
