from langchain_google_genai import GoogleGenerativeAIEmbeddings
from .model_base import BaseModel
# https://python.langchain.com/v0.1/docs/integrations/text_embedding/google_generative_ai/


class GoogleAIEmbeddingModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, alias):
        # import os
        # os.environ["GOOGLE_API_KEY"] = self.api_key

        self.client = GoogleGenerativeAIEmbeddings(
            model=self.model_name,
            google_api_key=self.api_key,
            **self.params
        )
        self.alias = alias
        return self
