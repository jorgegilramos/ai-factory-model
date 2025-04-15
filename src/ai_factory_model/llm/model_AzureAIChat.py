try:
    # langchain_azure_ai is not compatible with langchainwith python 3.13 (Numpy version conflict)
    from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
except Exception:
    None
from .model_base import BaseModel

# https://python.langchain.com/docs/integrations/chat/azure_ai/


class AzureAIChatModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, alias):
        # TODO: Change this!
        import os
        os.environ["AZURE_INFERENCE_CREDENTIAL"] = self.api_key
        os.environ["AZURE_INFERENCE_ENDPOINT"] = self.endpoint

        self.client = AzureAIChatCompletionsModel(
            azure_deployment=self.model_name,
            **self.params
        )
        self.alias = alias
        return self
