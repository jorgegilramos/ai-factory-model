from langchain_google_genai import ChatGoogleGenerativeAI
from .model_base import BaseModel
# https://python.langchain.com/docs/integrations/chat/google_generative_ai/


class GoogleAIChatModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, alias):
        # import os
        # os.environ["GOOGLE_API_KEY"] = self.api_key

        self.client = ChatGoogleGenerativeAI(
            # azure_deployment=self.model_name,
            model=self.model_name,
            google_api_key=self.api_key,
            **self.params
        )
        self.alias = alias
        return self
