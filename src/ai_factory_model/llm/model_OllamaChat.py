try:
    from langchain_ollama import ChatOllama  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover
    ChatOllama = None
from .model_base import BaseModel

# https://python.langchain.com/docs/integrations/chat/google_generative_ai/
# Model Names: Updated to typical Ollama models (llama2, mistral, codellama, neural-chat)


class OllamaChatModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, alias):
        if ChatOllama is None:
            raise ImportError("langchain_ollama is not installed. Please install it to use OllamaChatModel.")
        self.client = ChatOllama(
            azure_deployment=self.model_name,
            **self.params
        )
        self.alias = alias
        return self
