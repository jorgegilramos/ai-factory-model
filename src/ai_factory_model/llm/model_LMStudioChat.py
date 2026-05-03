try:
    from langchain_openai import ChatOpenAI  # pyright: ignore[reportMissingImports]
except ImportError:
    ChatOpenAI = None  # pragma: no cover

import re
from .model_base import BaseModel


class LMStudioChat(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, alias):
        if ChatOpenAI is None:
            raise ImportError("langchain_openai is not installed. Please install it to use LMStudioChat.")

        self.client = ChatOpenAI(
            openai_api_base=self.endpoint,
            openai_api_key="lmstudio",
            model=self.model_name,
            **self.params
        )
        self.alias = alias
        return self

    def prompt(self, params):
        original_response = super().prompt(params)
        clean_response = re.sub(r"<think>.*?</think>", "", original_response, flags=re.DOTALL).strip()

        return clean_response
