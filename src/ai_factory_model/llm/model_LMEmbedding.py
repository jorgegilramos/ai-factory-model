try:
    from openai import OpenAI  # pyright: ignore[reportMissingImports]
except ImportError:
    OpenAI = None  # pragma: no cover

from .model_base_embedding import BaseModelEmbedding


class LMEmbedding(BaseModelEmbedding):

    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, alias):
        if OpenAI is None:
            raise ImportError("openai is not installed. Please install it to use LMEmbedding.")

        self.client = OpenAI(
            base_url=self.endpoint,
            api_key=self.api_key or "lmstudio",
            model=self.model_name,
            **self.params
        )
        self.alias = alias
        return self

    def embed_query(self, text: str):
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(
            input=[text],
            model=self.model_name,
            **self.params
        )
        return response.data[0].embedding


# from openai import OpenAI
# client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
# x = client.embeddings.create(input = [text], model="nomic-ai/nomic-embed-text-v1.5-GGUF").data[0].embedding
