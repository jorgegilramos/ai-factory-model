from langchain_ollama import ChatOllama
from .model_base import BaseModel

# https://python.langchain.com/docs/integrations/chat/google_generative_ai/


class OllamaChatModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, alias):

        self.client = ChatOllama(
            azure_deployment=self.model_name,
            **self.params
        )
        self.alias = alias
        return self
