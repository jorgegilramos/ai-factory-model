from langchain_openai import ChatOpenAI
from .model_base import BaseModel


class OpenAIChatModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, alias):
        self.client = ChatOpenAI(
            openai_api_base=self.endpoint,
            openai_api_key=self.api_key,
            model=self.model_name,
            **self.params
        )
        self.alias = alias
        return self
