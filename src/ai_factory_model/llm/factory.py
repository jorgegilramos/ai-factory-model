from ..logger import error
from ..config import MODELS_CONFIG_FILE

# Models
from .model_base import BaseModel
from .model_AzureOpenAIChat import AzureOpenAIChatModel
from .model_AzureOpenAIEmbedding import AzureOpenAIEmbeddingModel
from .model_LMStudioChat import LMStudioChat
from .model_OpenAIChat import OpenAIChatModel
from .model_OpenAIEmbedding import OpenAIEmbeddingModel
from .model_AzureAIChat import AzureAIChatModel
from .model_GoogleAIChat import GoogleAIChatModel
from .model_GoogleAIEmbedding import GoogleAIEmbeddingModel
from .model_OllamaChat import OllamaChatModel
from .model_utils import load_from_file


cache = dict()


class ModelFactory:

    MODEL_CLASSES = {
        "AzureOpenAIChat": AzureOpenAIChatModel,
        "AzureOpenAIEmbedding": AzureOpenAIEmbeddingModel,
        "OpenChatAIChat": OpenAIChatModel,
        "OpenAIEmbedding": OpenAIEmbeddingModel,
        "AzureAIChat": AzureAIChatModel,
        "GoogleAIChat": GoogleAIChatModel,
        "GoogleAIEmbedding": GoogleAIEmbeddingModel,
        "OllamaChat": OllamaChatModel,
        "LMStudioChat": LMStudioChat
    }

    @staticmethod
    def get_model(alias: str) -> BaseModel:

        cached = cache.get(alias)
        if cached is not None:
            return cached

        config = load_from_file(MODELS_CONFIG_FILE)

        try:
            model_def = config.get(alias)
            if model_def is not None:
                model_class = ModelFactory.MODEL_CLASSES.get(model_def.get("connection_type"))
                model: BaseModel = model_class(model_def)
                model.initialize_model(alias)
                cache[alias] = model
                return model
        except Exception as e:
            error_msg = f"Error in ModelFactory.create {e}"
            error(f"{error_msg}")
            raise Exception(error_msg)

    @staticmethod
    def create_model(alias: str, model_def: dict) -> BaseModel:
        """
        Creates a model instance using the provided configuration dictionary.

        This method does not use the cache to retrieve an existing model,
        but it updates the cache with the newly created and initialized model.

        Args:
            alias: Alias for the model.
            model_def: Dictionary with model configuration.

        Returns:
            An initialized instance of the model.
        """
        try:
            model_class = ModelFactory.MODEL_CLASSES.get(model_def.get("connection_type"))
            model: BaseModel = model_class(model_def)
            model.initialize_model(alias)
            cache[alias] = model
            return model
        except Exception as e:
            error_msg = f"Error in ModelFactory.create {e}"
            error(f"{error_msg}")
            raise Exception(error_msg)
