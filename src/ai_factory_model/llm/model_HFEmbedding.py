from langchain_community.embeddings import HuggingFaceEmbeddings
from .model_base_embedding import BaseModelEmbedding


class HFEmbedding(BaseModelEmbedding):

    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, alias):
        self.client = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.params
        )
        self.alias = alias
        return self
