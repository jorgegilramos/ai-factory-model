import time
from azure.identity import ClientSecretCredential, get_bearer_token_provider
from openai.lib.azure import AzureADTokenProvider
from azure.core.credentials import AccessTokenInfo

from ...config import AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, \
    AZURE_TOKEN_URL


AZURE_TOKEN_DEFAULT_REFRESH_OFFSET = 300
AZURE_TOKEN_DEFAULT_REFRESH_RETRY_DELAY = 30


# Azure Authentication Client
class AzureAuthClient:

    credential: ClientSecretCredential = None
    token_provider: AzureADTokenProvider = None
    token: AccessTokenInfo = None
    _last_request_time: int = 0

    def __init__(self):

        # self.credential = ClientSecretCredential(
        #     tenant_id=AZURE_TENANT_ID,
        #     client_id=AZURE_CLIENT_ID,
        #     client_secret=AZURE_CLIENT_SECRET
        # )

        self.credential = self._create_credential(
            tenant_id=AZURE_TENANT_ID,
            client_id=AZURE_CLIENT_ID,
            client_secret=AZURE_CLIENT_SECRET
        )

        # Azure token provider
        self.token_provider = self._create_token_provider(self.credential)

    def _create_credential(self, tenant_id, client_id, client_secret) -> ClientSecretCredential:
        return ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret
        )

    def _should_refresh(self) -> bool:
        if self.token is not None:
            now = int(time.time())
            if self.token.refresh_on is not None and now >= self.token.refresh_on:
                return True
            if self.token.expires_on - now > AZURE_TOKEN_DEFAULT_REFRESH_OFFSET:
                return False
            if now - self._last_request_time < AZURE_TOKEN_DEFAULT_REFRESH_RETRY_DELAY:
                return False
        return True

    def _create_token_provider(self, credential) -> AzureADTokenProvider:
        # Azure token provider
        return get_bearer_token_provider(credential, AZURE_TOKEN_URL)

    def get_token(self) -> str:
        # Get token
        if self._should_refresh():
            self._last_request_time = int(time.time())
            self.token = self.credential.get_token_info(AZURE_TOKEN_URL)
        return self.token.token

    def get_token_provider(self) -> AzureADTokenProvider:

        # if self.token_provider is None:
        #     raise Exception("Token provider must have a value")
        return self.token_provider

    def get_credential(self) -> ClientSecretCredential:
        return self.credential
