import os
from decouple import config

_logging_dir: str = config("AI_FACTORY_MODEL_LOGGING_DIR", default="logs", cast=str)
# Add directory separator at the end
LOGGING_DIR: str = _logging_dir if _logging_dir.endswith("/") else f"{_logging_dir}{os.sep}"
LOGGING_FILE: str = config("AI_FACTORY_MODEL_LOGGING_FILE", default="ai-model-factoy.log", cast=str)
LOGGING_WHEN: str = config("AI_FACTORY_MODEL_LOGGING_WHEN", default="midnight", cast=str)
LOGGING_INTERVAL: int = int(config("AI_FACTORY_MODEL_LOGGING_INTERVAL", default="1", cast=str))
LOGGING_TITLE: str = config("AI_FACTORY_MODEL_LOGGING_TITLE", "ai-model-factoy", cast=str)
LOGGING_LEVEL: str = config("AI_FACTORY_MODEL_LOGGING_LEVEL", "INFO")
LOGGING_HANDLERS: str = config("AI_FACTORY_MODEL_LOGGING_HANDLERS", "console,file_handler")
LOGGING_FORMATTER: str = config("AI_FACTORY_MODEL_LOGGING_FORMATTER",
                                "%(asctime)s - [%(name)s] - %(levelname)-5s - %(message)s")

# Debugging mode
FORCE_LOG_DEBUG: bool = config("AI_FACTORY_MODEL_FORCE_LOG_DEBUG", default=False, cast=bool)
