from langchain_openai import AzureChatOpenAI

from .model_base import BaseModel, VALUE_SERVICE_PRINCIPAL, VALUE_API_KEY
from .auth_clients import AzureAuthClient

# https://github.com/openai/openai-python/blob/main/examples/azure_ad.py


class AzureOpenAIChatModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, alias):
        if self.api_auth == VALUE_SERVICE_PRINCIPAL:
            # Azure Service Principal
            self.auth_client = AzureAuthClient()

            self.client = AzureChatOpenAI(
                azure_endpoint=self.endpoint,
                # Deprecated:
                # azure_ad_token=self.auth_client.get_token(),
                azure_ad_token_provider=self.auth_client.get_token_provider(),
                azure_deployment=self.model_name,
                api_version=self.version,
                **self.params
            )
        elif self.api_auth == VALUE_API_KEY:
            # Azure API Token
            self.client = AzureChatOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                azure_deployment=self.model_name,
                api_version=self.version,
                **self.params
            )
        else:
            raise ValueError("Authorization should be \"service_principal\" or \"api_key\"")
        self.alias = alias
        return self
