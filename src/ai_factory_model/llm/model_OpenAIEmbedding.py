from langchain_openai import OpenAIEmbeddings
from .model_base_embedding import BaseModelEmbedding


class OpenAIEmbeddingModel(BaseModelEmbedding):

    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, alias):
        self.client = OpenAIEmbeddings(
            openai_api_base=self.endpoint,
            openai_api_key=self.api_key,
            model=self.model_name,
            **self.params
        )
        self.alias = alias
        return self
