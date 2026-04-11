try:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # pyright: ignore[reportMissingImports]
except ImportError:
    HuggingFaceEmbeddings = None

from .model_base_embedding import BaseModelEmbedding


class HFEmbedding(BaseModelEmbedding):

    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, alias):
        if HuggingFaceEmbeddings is None:
            raise ImportError("langchain_community.embeddings is not installed. Please install it to use HFEmbedding.")

        self.client = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.params
        )
        self.alias = alias
        return self
