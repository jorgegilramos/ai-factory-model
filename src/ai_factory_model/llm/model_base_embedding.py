from .model_base import BaseModel


class BaseModelEmbedding(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def embedding(self, text: str):

        text = text.replace("\n", " ")
        return self.client.embed_query(text)
