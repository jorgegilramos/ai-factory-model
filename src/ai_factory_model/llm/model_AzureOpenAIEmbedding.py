from langchain_openai import AzureOpenAIEmbeddings
from .model_base_embedding import BaseModelEmbedding


class AzureOpenAIEmbeddingModel(BaseModelEmbedding):

    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, alias):
        self.client = AzureOpenAIEmbeddings(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            azure_deployment=self.model_name,
            api_version=self.version,
            **self.params
        )
        self.alias = alias
        return self
