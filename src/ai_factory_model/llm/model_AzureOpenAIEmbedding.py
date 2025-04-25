from langchain_openai import AzureOpenAIEmbeddings
from .model_base import VALUE_SERVICE_PRINCIPAL, VALUE_API_KEY
from .model_base_embedding import BaseModelEmbedding
from .auth_clients import AzureAuthClient


class AzureOpenAIEmbeddingModel(BaseModelEmbedding):

    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, alias):

        if self.api_auth == VALUE_SERVICE_PRINCIPAL:
            # Azure Service Principal
            self.auth_client = AzureAuthClient()

            self.client = AzureOpenAIEmbeddings(
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
            self.client = AzureOpenAIEmbeddings(
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
