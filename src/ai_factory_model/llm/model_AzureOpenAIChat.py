from langchain_openai import AzureChatOpenAI
from .model_base import BaseModel

# from .auth_clients import AZURE_OPENAI_API_KEY, AzureAuthClient, AWSAuthClient
from .auth_clients import AzureAuthClient


class AzureOpenAIChatModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, alias):
        if self.api_auth == "service_principal":
            # Azure Service Principal
            self.auth_client = AzureAuthClient()
            self.client = AzureChatOpenAI(
                azure_endpoint=self.endpoint,
                azure_ad_token=self.auth_client.get_token(),
                azure_deployment=self.model_name,
                api_version=self.version,
                **self.params
            )
        elif self.api_auth == "api_key":
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
